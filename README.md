# Synthetic Pre-Training for Bitcoin Distributional Prediction (3–7 Day Horizon)

_Adapting the methodology from [JointFM-0.1 (Hackmann, 2026)](https://arxiv.org/abs/2603.20266) — a foundation model pre-trained on synthetic SDE trajectories — to single-asset probabilistic forecasting of BTC/USD._

---

## 1. Core Idea

JointFM shows that a transformer trained on **millions of synthetic stochastic trajectories** (sampled from SDEs with randomly varied parameters) can learn a general "grammar" of time-series dynamics and produce calibrated distributional forecasts at inference time — without ever fitting to the target data.

We adapt this to a narrower, harder problem: predict the **full probability distribution** of Bitcoin's log-return (or price) over a 3–7 day forward window, given a context window of recent history.

The key modifications from JointFM:

- **Single-target** (univariate BTC), not multi-target joint distribution
- **Crypto-specific SDE families** instead of generic/equity-focused samplers
- **Short horizon** (3–7 days) instead of quarterly
- **Mixed training**: synthetic majority + real BTC data minority

---

## 2. Choosing the Right SDE Family

Bitcoin returns exhibit heavy tails, volatility clustering, asymmetric jumps (crashes are sharper than rallies), and occasional regime shifts. No single SDE captures all of this, so we build a **mixture of generative models** and sample from them during training.

### 2.1 Geometric Brownian Motion (Baseline)

The simplest model. Useful as a "boring" component in the training mix so the network learns to recognize low-complexity dynamics too.

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t
$$

- $\mu$: drift
- $\sigma$: constant volatility
- $W_t$: standard Wiener process

Log-returns over interval $\Delta t$:

$$
\ln\!\left(\frac{S_{t+\Delta t}}{S_t}\right) \sim \mathcal{N}\!\left(\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t,\; \sigma^2 \Delta t\right)
$$

### 2.2 Merton Jump-Diffusion

Adds Poisson-driven jumps to GBM. Captures sudden crashes/pumps.

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t + S_{t^-}\, (e^{J} - 1)\, dN_t
$$

- $N_t$: Poisson process with intensity $\lambda$ (expected jumps per unit time)
- $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$: log-jump size

### 2.3 Kou Double-Exponential Jump-Diffusion

Like Merton but with **asymmetric double-exponential** jump sizes — a much better fit for crypto, where downward jumps tend to be larger and faster than upward ones.

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t + S_{t^-}\, (e^{J} - 1)\, dN_t
$$

where the jump size $J$ has density:

$$
f_J(x) = p \cdot \eta_1 \, e^{-\eta_1 x}\, \mathbf{1}_{x \geq 0} \;+\; (1-p) \cdot \eta_2 \, e^{\eta_2 x}\, \mathbf{1}_{x < 0}
$$

- $p$: probability of an upward jump
- $\eta_1 > 1$: rate of upward jump decay (larger = smaller jumps)
- $\eta_2 > 0$: rate of downward jump decay
- Asymmetry: typically $p < 0.5$ and $\eta_2 < \eta_1$ for BTC (crashes are bigger)

### 2.4 Bates Model (Stochastic Volatility + Jumps)

Combines Heston stochastic volatility with Merton jumps. The most expressive single model.

$$
dS_t = \mu \, S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S + S_{t^-}(e^J - 1)\, dN_t
$$

$$
dv_t = \kappa\,(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^v
$$

$$
\text{Corr}(dW_t^S, dW_t^v) = \rho
$$

- $v_t$: instantaneous variance (stochastic)
- $\kappa$: mean-reversion speed of variance
- $\theta$: long-run variance level
- $\xi$: vol-of-vol
- $\rho$: leverage effect (typically negative — vol rises when price drops)
- $J, N_t$: same jump components as Merton

### 2.5 Regime-Switching GBM

Captures crypto's tendency to alternate between bull/bear/sideways regimes.

$$
dS_t = \mu_{Z_t}\, S_t\, dt + \sigma_{Z_t}\, S_t\, dW_t
$$

where $Z_t \in \{1, 2, \ldots, K\}$ is a continuous-time Markov chain with generator matrix $Q$.

Each regime $k$ has its own $(\mu_k, \sigma_k)$. Transition rates $q_{ij}$ control how often the process switches. For BTC, 2–3 regimes work well (e.g., low-vol sideways, high-vol bull, crash).

### 2.6 (Optional) CGMY / Tempered Stable

For even heavier tails than jump-diffusion, replace the Brownian component with a tempered stable (CGMY) process. This is more exotic and harder to simulate, but captures the empirical observation that BTC returns have near-infinite variance in some periods.

---

## 3. Parameter Sampling Strategy

The central insight from JointFM: **don't fit one set of parameters — sample many plausible ones and train the model to handle all of them.**

For each training batch, we:

1. **Sample an SDE type** from {GBM, Merton, Kou, Bates, Regime-Switching} with weights (e.g., 5%, 20%, 30%, 30%, 15%)
2. **Sample parameters** from broad priors calibrated to be "BTC-plausible"
3. **Simulate a trajectory** of length `context_window + forecast_horizon`
4. **Train** the model to predict the distributional forecast given the context

### Recommended Parameter Priors

These ranges are informed by historical BTC calibrations but deliberately broadened so the model generalizes:

| Parameter                   | Symbol     | Prior Distribution                               | Rationale                                         |
| --------------------------- | ---------- | ------------------------------------------------ | ------------------------------------------------- |
| **Annualized drift**        | $\mu$      | $\text{Uniform}(-0.5, 1.5)$                      | BTC has ranged from severe bear to 10x bull years |
| **Base volatility**         | $\sigma$   | $\text{Uniform}(0.3, 1.5)$                       | BTC annualized vol typically 40–120%, allow wider |
| **Jump intensity**          | $\lambda$  | $\text{Uniform}(0.5, 50)$ /yr                    | From rare (1/yr) to frequent (weekly) jumps       |
| **Jump mean (up)**          | $1/\eta_1$ | $\text{Uniform}(0.01, 0.15)$                     | Avg upward jump 1–15%                             |
| **Jump mean (down)**        | $1/\eta_2$ | $\text{Uniform}(0.02, 0.25)$                     | Avg downward jump 2–25% (asymmetric)              |
| **Up-jump probability**     | $p$        | $\text{Beta}(2, 3)$                              | Centered ~0.4, slight downward bias               |
| **Mean-reversion speed**    | $\kappa$   | $\text{Uniform}(0.5, 10)$                        | Vol mean-reverts over days to months              |
| **Long-run variance**       | $\theta$   | $\text{Uniform}(0.1, 2.0)$                       | Squared, so covers wide vol range                 |
| **Vol-of-vol**              | $\xi$      | $\text{Uniform}(0.2, 2.0)$                       | BTC vol itself is volatile                        |
| **Leverage correlation**    | $\rho$     | $\text{Uniform}(-0.9, -0.1)$                     | Negative: vol rises on drops                      |
| **Regime transition rates** | $q_{ij}$   | $\text{Exponential}(\text{mean}=30\text{ days})$ | Avg regime duration ~1 month                      |

### Simulation Details

- **Time step**: $\Delta t = 1/365$ (daily) or $1/(365 \times 24)$ (hourly) for finer resolution
- **Context window**: 30–90 days of history (the model sees this)
- **Forecast horizon**: 3, 5, or 7 days (randomly chosen per sample during training)
- **Paths per parameter set**: 1 (the diversity comes from parameter variation, not repeated paths)
- **Transform**: work in **log-return space** $r_t = \ln(S_t / S_{t-1})$, not raw prices

---

## 4. Model Architecture

Following JointFM's approach but simplified for the univariate case:

```
Input: [r_{t-L}, r_{t-L+1}, ..., r_{t-1}, r_t]  (context window of L log-returns)
  │
  ▼
Patching & Embedding (group returns into patches of ~5 days)
  │
  ▼
Transformer Encoder (6–8 layers, causal or bidirectional over context)
  │
  ▼
Forecast Head: outputs distributional parameters for horizon h ∈ {3,5,7} days
  │
  ▼
Output: parameters of predicted distribution for cumulative log-return r_{t+1:t+h}
```

### 4.1 Output Distribution (Forecast Head)

The forecast head parameterizes a **flexible distribution** over the cumulative h-day log-return. Good choices:

**Option A — Mixture of Gaussians:**

$$
p(r_{t:t+h}) = \sum_{k=1}^{K} \pi_k \; \mathcal{N}(r \mid \mu_k, \sigma_k^2)
$$

The head outputs $3K$ values: $K$ mixture weights (softmax), $K$ means, $K$ log-standard-deviations. Use $K = 5$–$10$ components.

**Option B — Quantile regression (non-parametric):**

Output $Q$ quantile predictions $\hat{q}_{\tau_1}, \ldots, \hat{q}_{\tau_Q}$ at fixed probability levels (e.g., $\tau \in \{0.01, 0.05, 0.10, \ldots, 0.90, 0.95, 0.99\}$).

**Option C — Normalizing flow head:**

A small conditional normalizing flow (e.g., 4-layer affine coupling) that maps a base Gaussian to the target distribution. Most expressive, but more complex to train.

**Recommendation**: Start with Mixture of Gaussians ($K=8$). It's expressive enough for heavy tails and multimodality, easy to sample from, and has a closed-form likelihood for the loss function.

### 4.2 Horizon Conditioning

Encode the forecast horizon $h$ (3, 5, or 7 days) as a learned embedding and add/concatenate it to the transformer's output before the forecast head. This lets one model handle all horizons.

---

## 5. Loss Function

### 5.1 Primary: Negative Log-Likelihood (NLL)

For Mixture of Gaussians with $K$ components:

$$
\mathcal{L}_{\text{NLL}} = -\ln \sum_{k=1}^{K} \pi_k \; \phi\!\left(\frac{r^* - \mu_k}{\sigma_k}\right) \frac{1}{\sigma_k}
$$

where $r^* = \ln(S_{t+h}/S_t)$ is the realized cumulative log-return and $\phi$ is the standard normal PDF.

For quantile regression, use the **pinball loss**:

$$
\mathcal{L}_{\text{pinball}} = \sum_{i=1}^{Q} \rho_{\tau_i}(r^* - \hat{q}_{\tau_i}), \quad \rho_\tau(u) = u\,(\tau - \mathbf{1}_{u<0})
$$

### 5.2 Auxiliary: CRPS (Continuous Ranked Probability Score)

CRPS directly measures how well the predicted CDF matches reality. For a mixture of Gaussians it has a closed form. It's a strictly proper scoring rule — the true distribution uniquely minimizes it.

$$
\text{CRPS}(F, r^*) = \mathbb{E}_F[|X - r^*|] - \tfrac{1}{2}\mathbb{E}_F[|X - X'|]
$$

where $X, X' \sim F$ are independent draws from the predicted distribution.

### 5.3 Combined Training Loss

$$
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \alpha \cdot \text{CRPS}
$$

with $\alpha \approx 0.1$–$0.5$. The NLL drives sharp density estimation; CRPS regularizes calibration.

---

## 6. Mixing in Real Data

Pure synthetic training risks a domain gap. We propose a **curriculum with real data fine-tuning**:

### Phase 1: Synthetic Pre-training (80% of training)

Train on purely synthetic trajectories as described above. This teaches the model the geometry of stochastic processes.

### Phase 2: Mixed Fine-tuning (20% of training)

Mix synthetic and real data:

- **70% synthetic** (same as phase 1)
- **30% real BTC data** — historical daily/hourly returns from 2015–present

For real data augmentation:

- **Rolling windows**: slide the context window across all available history
- **Multi-exchange**: use data from Bitstamp, Coinbase, Binance for slight distributional variety
- **Bootstrap**: resample contiguous blocks to increase effective dataset size

### Why Not 100% Real Data?

BTC has ~3,800 daily observations (2015–2025). For a 60-day context + 7-day horizon, that's only ~3,700 non-overlapping-ish training examples. Far too few to train a transformer. The synthetic data provides the volume; the real data provides the fine-tuning signal.

---

## 7. Evaluation

### 7.1 Calibration (PIT Histogram)

Apply the **Probability Integral Transform**: if the model is well-calibrated, $F(r^*) \sim \text{Uniform}(0,1)$. Plot the histogram of PIT values across the test set — it should be flat.

### 7.2 CRPS on Held-Out Real Data

Compute CRPS on a rolling out-of-sample window (e.g., 2023–2025 real BTC data). Compare against baselines:

- Historical simulation (empirical distribution of past 60-day windows)
- GARCH(1,1) with Student-t innovations
- Calibrated Kou jump-diffusion (classical, fitted daily)

### 7.3 Coverage

Check prediction interval coverage:

$$
\text{Coverage}(\alpha) = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[r_i^* \in \left[q_{\alpha/2}^{(i)},\; q_{1-\alpha/2}^{(i)}\right]\right]
$$

A well-calibrated model should have ~90% coverage at the 90% level, ~95% at 95%, etc.

### 7.4 Tail Accuracy

Specifically check the 1st and 99th percentile predictions. For BTC risk management, the tails matter more than the center. Report the **tail-weighted CRPS** (weight extreme quantiles more heavily).

### 7.5 Sharpness

Calibration alone isn't enough — a model predicting "uniform over all possible returns" would be perfectly calibrated but useless. Also report the average **width of the 90% prediction interval**. Narrower is better, conditional on maintaining calibration.

### Key Libraries

- **SDE simulation**: `sdeint`, `diffrax` (JAX), or custom Euler-Maruyama
- **Transformer**: PyTorch with standard `nn.TransformerEncoder`
- **Data**: `yfinance` or `ccxt` for real BTC data
- **Evaluation**: `properscoring` (CRPS), `scipy.stats` (PIT)

---

## 8. Risks & Mitigations

| Risk                                  | Severity | Mitigation                                                               |
| ------------------------------------- | -------- | ------------------------------------------------------------------------ |
| Synthetic-real distribution gap       | High     | Mixed fine-tuning (Phase 2); validate PIT on real data                   |
| SDE family too narrow                 | Medium   | Include 5+ SDE types; add CGMY if tails are still underweight            |
| Overfitting to real data in Phase 2   | Medium   | Keep synthetic majority (70%); early stopping on real validation set     |
| Model learns "average SDE" not BTC    | Medium   | Weight Kou + Bates higher in sampling; condition on recent realized vol  |
| 3-day vs 7-day horizon conflict       | Low      | Horizon embedding; or train separate lightweight heads                   |
| Non-stationarity of BTC dynamics      | High     | Use rolling retraining (e.g., quarterly); expand SDE priors over time    |
| Evaluation on too-small real test set | Medium   | Use expanding-window backtest over 2+ years; report confidence intervals |

---

## 9. Why This Should Work (and Where It Might Not)

**Why it should work:**

The JointFM paper demonstrates that transformers can learn a surprisingly general mapping from "recent trajectory shape" → "plausible future distribution" when exposed to enough variety during training. By narrowing the SDE family to crypto-plausible dynamics and adding real BTC fine-tuning, we get the best of both worlds: the model sees enough stochastic variety to generalize, but is also grounded in real market data.

**Where it might fail:**

Bitcoin is driven partly by **exogenous, non-price signals** — regulatory news, exchange hacks, ETF flows, macro events, social media sentiment — that leave no trace in the price history alone. A purely price-based distributional model will produce well-calibrated intervals _on average_, but will be poorly calibrated _conditional on_ major news events. It will likely underestimate tail probabilities during calm periods (when a shock is brewing but hasn't hit yet) and overestimate them during already-volatile periods.

To partially address this, you could extend the context to include auxiliary features (on-chain metrics, funding rates, options implied vol, VIX) — but this moves away from the clean synthetic pre-training paradigm and into more traditional feature engineering territory.

---

## References

- **JointFM-0.1**: Hackmann, S. (2026). _JointFM-0.1: A Foundation Model for Multi-Target Joint Distributional Prediction._ [arXiv:2603.20266](https://arxiv.org/abs/2603.20266)
- **Kou Jump-Diffusion**: Kou, S. G. (2002). _A Jump-Diffusion Model for Option Pricing._ Management Science, 48(8), 1086–1101.
- **Bates Model**: Bates, D. S. (1996). _Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options._ Review of Financial Studies, 9(1), 69–107.
- **CRPS**: Gneiting, T. & Raftery, A. E. (2007). _Strictly Proper Scoring Rules, Prediction, and Estimation._ JASA, 102(477), 359–378.
- **Time-Series Foundation Models**: Das et al. (2024). _A Decoder-Only Foundation Model for Time-Series Forecasting._ [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- **Crypto Volatility Modeling**: Scaillet, O., Treccani, A., & Trevisan, C. (2020). _High-Frequency Jump Analysis of the Bitcoin Market._ Journal of Financial Econometrics, 18(2), 209–232.
