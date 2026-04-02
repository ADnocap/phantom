# Phantom v2: Improvement Plan

Literature-informed improvements to push Phantom beyond the current 3x naive baseline.
All changes are **backward-compatible** — toggled via config flags so the original model can still be run as-is.

---

## Option 1: Mixture of Student-t (MoT) Output Head

**Problem**: K=5 Gaussian components have exponentially decaying tails. BTC empirically has power-law tails (exponent ~2.87). The model structurally underestimates crash/rally probabilities.

**Fix**: Add a learnable degrees-of-freedom parameter `nu_k` per component. Student-t tails decay as |x|^{-(nu+1)}, which is power-law. When nu -> infinity it recovers the Gaussian, so the model can *learn* when heavy tails are appropriate.

**Changes**:
- `MixtureHead` outputs 4K params: (log_pi, mu, log_sigma, log_nu)
- `log_nu` clamped to [log(2.01), log(100)] — nu > 2 for finite variance
- NLL uses Student-t log-pdf; energy distance samples via Student-t reparameterization
- Closed-form CRPS exists for Student-t (Grimit et al., 2006)

**Config**: `use_student_t: bool = False`

**Evidence**: JointFM uses "fat-tailed multivariate distributions." Moirai 1.0 uses mixture of Student-t + Normal. BTC tail risk papers consistently find Student-t outperforms Gaussian.

**Impact**: HIGH | **Effort**: LOW (~50 lines in model.py + losses.py)

---

## Option 2: Multi-Scale Patching

**Problem**: Fixed 5-day patches create an inductive bias toward weekly patterns. The model can't distinguish 3-day momentum from 15-day mean-reversion — everything is discretized into uniform 5-day chunks.

**Fix**: Parallel patch embeddings at multiple resolutions. From a 75-day context with patch sizes [3, 5, 15]:
- 3-day patches: 25 tokens (short-term momentum)
- 5-day patches: 15 tokens (weekly patterns)
- 15-day patches: 5 tokens (multi-week trends/regimes)
- Total: 45 tokens (trivially cheap for attention)

Add a learnable **scale embedding** per resolution so the encoder knows which granularity each token comes from.

**Config**: `patch_sizes: list[int] | None = None` (None = original single-scale)

**Evidence**: MTST (AISTATS 2024), Pathformer (ICLR 2024), Moirai all use multi-scale patching. Universal improvement across time series foundation models.

**Impact**: MEDIUM-HIGH | **Effort**: LOW-MEDIUM

---

## Option 3: Trailing Realized Volatility Features

**Problem**: The model sees only raw log-returns. It must *learn* volatility estimation from scratch. Providing trailing realized vol at multiple windows gives the model direct regime information.

**Fix**: Compute trailing realized vol as additional input channels:
- Channel 0: raw log-return (as now)
- Channel 1: trailing 7-day realized vol (std of last 7 returns × sqrt(365))
- Channel 2: trailing 14-day realized vol
- Channel 3: trailing 30-day realized vol

**Pretraining compatibility**: These are deterministic functions of returns — computable from synthetic SDE data identically to real data. No information leakage, no masking needed.

**Implementation**: Input becomes (B, L, C) instead of (B, L). Patch embedding: `Linear(patch_len * n_channels, d_model)`.

**Config**: `n_input_channels: int = 1` (1 = raw returns only, 4 = returns + 3 vol features)

**Evidence**: Multi-scale realized vol is the single most consistently useful feature in crypto forecasting literature. A 2025 paper using 47 features found vol features dominate the feature importance ranking.

**Impact**: MEDIUM-HIGH | **Effort**: MEDIUM

---

## Option 4: Auxiliary Quantile Loss

**Problem**: Energy distance compares full distributions but doesn't directly penalize poor quantile calibration. Neither ED nor NLL optimizes specifically for interval coverage quality.

**Fix**: Add a pinball (quantile) loss as an auxiliary objective. For each sample, compute quantiles from the predicted mixture CDF at levels [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], and apply pinball loss against the target.

**Implementation**: Compute mixture CDF analytically, invert numerically for quantiles, apply pinball loss. Weight at ~0.1-0.3.

**Config**: Training arg `quantile_weight: float = 0.0` (0.0 = disabled)

**Evidence**: Moirai 2.0 (Salesforce, 2025) switched from mixture distributions to quantile loss and reported it as **the single largest improvement** in their ablation: MASE 0.85 -> 0.744. An auxiliary quantile loss gives both the continuous density from the mixture head (needed for CFG, sampling) + direct quantile calibration pressure.

**Impact**: MEDIUM-HIGH | **Effort**: LOW (~30 lines)

---

## Option 5: Gumbel-Softmax for Fully Differentiable Energy Distance

**Problem**: `torch.multinomial(pi, M)` in energy distance is non-differentiable w.r.t. mixture weights (log_pi). The auxiliary NLL on a random branch compensates, but it's a noisy single-sample proxy.

**Fix**: Replace multinomial with Gumbel-Softmax (straight-through estimator). The entire energy distance becomes differentiable w.r.t. all parameters including mixture weights.

```python
# Current (non-differentiable w.r.t. log_pi):
comp_idx = torch.multinomial(pi, M, replacement=True)

# New (differentiable):
logits = log_pi.unsqueeze(1).expand(B, M, K)
gumbel_weights = F.gumbel_softmax(logits, tau=0.5, hard=True)
mu_sel = (gumbel_weights * mu.unsqueeze(1)).sum(-1)
sigma_sel = (gumbel_weights * sigma.unsqueeze(1)).sum(-1)
```

**Config**: Training arg `use_gumbel_softmax: bool = False`

**Evidence**: Gumbel-Softmax (Jang et al., 2017) is the standard solution for differentiable discrete sampling. Eliminates the need for the NLL crutch for pi gradient flow.

**Impact**: MEDIUM | **Effort**: LOW (~15 lines in losses.py)

---

## Option 6: Richer SDE Family (Multifractal + Fractional OU)

**Problem**: All 5 current SDEs are Markovian. But BTC exhibits strong multifractal structure (confirmed by July 2025 paper: spectral widths exceeding 2.0, R^2 > 0.85 across 2017-2024). The model never sees this during pre-training.

**Important negative finding**: Rough volatility (rough Heston, rough Bergomi) does NOT work for BTC — no valid Hurst exponent can be estimated for Bitcoin volatility. Multifractal models are the correct framework.

**New SDE families**:

1. **Multifractal Random Walk (MRW)**: Log-normal cascade for the volatility process. Generates paths with scale-dependent volatility clustering — the defining property BTC exhibits.

2. **Fractional Ornstein-Uhlenbeck with Stochastic Volatility**: Captures long-memory effects (autocorrelation in vol decays as a power law, not exponentially like Bates). Uses fractional Brownian motion via the Cholesky method.

**Config**: `sde_version: str = 'v1'` ('v1' = original 5 families, 'v2' = adds MRW + FOU)

**Evidence**: The SDE generator is the single most important component in TempoPFN (NeurIPS 2025) — removing it caused 26% CRPS degradation. Richer SDEs should produce richer pre-training.

**Impact**: MEDIUM-HIGH | **Effort**: MEDIUM-HIGH (numba-JIT implementations)

---

## Option 7: Gradual Domain Adaptation (Fine-tuning)

**Problem**: Fixed 70/30 synthetic/real mixing throughout fine-tuning. Research shows this isn't optimal.

**Fix**: Two complementary improvements:

**A. Annealing synthetic ratio**: Start at 90% synthetic / 10% real (safe, minimal forgetting), gradually shift to 30% synthetic / 70% real (maximum real-data adaptation). Chronos found ~10% synthetic is optimal at the end of fine-tuning.

**B. BTC-calibrated intermediate SDEs**: Estimate BTC's empirical parameters (drift, vol, jump frequency). During fine-tuning, generate synthetic data from SDEs whose parameters interpolate between the broad prior and BTC-calibrated values. Based on the GOAT framework (JMLR 2024) — intermediate domains along the Wasserstein geodesic.

**Config**: Training args `anneal_real_fraction: bool = False`, `btc_calibrated_sdes: bool = False`

**Evidence**: GOAT (JMLR 2024) proves uniform Wasserstein geodesic placement is optimal for gradual domain adaptation.

**Impact**: MEDIUM-HIGH | **Effort**: MEDIUM

---

## Option 9: In-Model Trend-Residual Decomposition

**Problem**: The encoder processes raw returns without separating trend from noise. Signal and noise are entangled throughout the transformer layers.

**Fix**: Add a moving-average decomposition block between encoder layers (Autoformer-style). After each transformer layer, extract the moving average (trend) and pass the residual to the next layer. The final output concatenates both representations.

```python
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=5):
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):  # (B, N, d)
        trend = self.avg_pool(x.transpose(1,2)).transpose(1,2)
        residual = x - trend
        return residual, trend
```

**Config**: `use_decomposition: bool = False`

**Evidence**: Autoformer (NeurIPS 2021) showed 38% improvement on long-term benchmarks using in-model decomposition. FEDformer and DLinear confirmed the benefit.

**Impact**: MEDIUM | **Effort**: LOW (~20 lines)

---

## NOT Included (For Later)

### Ensemble (5 seeds)
- 5-15% CRPS improvement, but 5x training cost
- Can be done after all other improvements are validated
- Aggregation via Vincentization (quantile averaging)

### Conformal Prediction Wrapper (ACI)
- ECE is already 0.005 — conformal adds robustness guarantees, not calibration improvement
- Lightweight post-processing, can be added anytime

### Flow Matching / Normalizing Flow Head
- Highest potential ceiling but requires major architecture rework
- Consider after MoT proves the concept of better distributional heads

---

## Recommended Implementation Order

| Phase | Options | Rationale |
|-------|---------|-----------|
| **A** (quick wins) | 5 + 1 + 4 | Fix gradients, fix tails, add quantile pressure. Small code changes, large impact. |
| **B** (architecture) | 9 + 2 | Decomposition + multi-scale patching. Better encoder conditioning. |
| **C** (data) | 3 + 6 + 7 | Richer inputs, richer SDEs, better domain adaptation. |

Phase A alone should produce a meaningful jump. Phase B improves the encoder's multi-scale conditioning. Phase C addresses the input representation and synthetic-to-real gap.
