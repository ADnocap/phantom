# Phantom Experiments Log

## Summary

Across 15+ experiments over pretraining and fine-tuning, the model achieves **CRPS 0.0288 on real BTC test data (2023-07 to 2026)**, matching the post-hoc Gaussian baseline (0.0287). The model provides well-calibrated distributional forecasts (ECE=0.012, 90%→93% coverage) but cannot beat a simple Gaussian with the correct mean and std.

---

## The Core Problem

The model learns a good **marginal** distribution but fails to produce meaningful **conditional** (input-dependent) predictions on real BTC data:
- Predicted mean std: **0.0008** (should be ~0.04 to add value)
- Corr(predicted_mean, actual): **0.011**
- Corr(predicted_std, |error|): **0.062**
- The model outputs nearly the same distribution for every input

---

## What Worked

| Technique | Impact | Where it helped |
|-----------|--------|-----------------|
| **Contrastive loss (InfoNCE)** | SDE accuracy 38% (best) | Encoder discrimination |
| **Moment matching** | 0.39 corr with branch mean | First conditional mean signal |
| **GARCH/Momentum SDEs (v3)** | 0.80 vol conditioning corr | Within-context signal on synthetic |
| **Aggressive encoder fine-tuning** | Val CRPS 0.0375 (best) | Only config that moved val CRPS |
| **Single Student-t head** | Same performance, 3 params vs 15 | Simpler, no component collapse |
| **CRPS-avg (closed-form)** | Zero noise floor | Better than sample-based ED |
| **Oracle CRPS discovery** | Proved pretraining is optimal | Stopped wasting time on pretraining |

## What Didn't Work

| Technique | Why it failed |
|-----------|---------------|
| **Multi-scale patching [3,5,15]** | No CRPS improvement over single-scale |
| **FiLM conditioning** | Gradient instability (spikes to 3500), no CRPS benefit |
| **Series decomposition** | No improvement — returns don't have trend/seasonal structure |
| **Multi-channel vol features** | No improvement — model already estimates vol from returns |
| **Student-t mixture K=5** | nu→60+ (collapsed to Gaussian), same as MoG |
| **Quantile loss** | No CRPS improvement |
| **High NLL weight in fine-tuning** | No improvement — drove loss negative but didn't help CRPS |
| **Domain adaptation (annealing)** | No improvement — real/synthetic ratio doesn't matter |
| **Heavy aux weight (0.5)** | Starved main loss (94% gradient budget to SDE classification) |
| **50K fine-tuning steps** | Zero improvement after step 500 |
| **L2-SP regularization** | Prevented encoder adaptation — removing it helped |
| **Encoder freeze period** | Prevented encoder adaptation — removing it helped |

---

## Key Findings

### 1. Oracle CRPS Floor
The theoretical minimum CRPS on synthetic test data is **0.065** (per-sample Gaussian fitted to 128 branches). Our models achieve **0.064**. **Pre-training is already at the theoretical optimum.** The CRPS plateau at ~0.064 is the Bayes-optimal floor, not a training failure.

### 2. Markovian SDEs Have No Context Signal
For GBM/Merton/Kou/Bates with FIXED parameters, different 75-day context realizations give branch_mean std of only **0.008**. The context is irrelevant for prediction — only the terminal state matters. The model correctly learns "context doesn't matter" because for Markovian SDEs, it mathematically doesn't. The 0.40 correlation between context features and branch stats comes entirely from **cross-SDE parameter variation**, not within-sample conditioning.

### 3. Energy Distance Gradient Vanishing
ED gradient scales as O(ED). At ED=0.005, the signal (~0.005) is below the sample noise floor (~0.009 with M=256, N=128). The model can't detect which direction to improve. This is why all ED-based experiments plateau at the same level within 3K steps.

### 4. Posterior Collapse Analog
The cross-attention decoder learns to ignore encoder output via the residual connection (`query = query + attn_out`). The horizon embedding alone carries the prediction. Verified: zeroing encoder output changes mu by only 0.018.

### 5. BTC Returns Are Near-Gaussian at 3-7 Day Horizons
BTC 3-7 day returns have excess kurtosis ~3 (dropping from ~14 at daily). A single Gaussian with the correct mean and std is near-optimal. There's very little conditional signal exploitable at this horizon.

### 6. Non-Markovian SDEs Help Pretraining But Don't Transfer
GARCH(1,1) SDEs create genuine within-context vol conditioning (corr 0.80). But this doesn't transfer to BTC fine-tuning — val CRPS stays at 0.038 regardless.

---

## Experiment Timeline

### Phase 1-2: v2 Full Config
**Config**: Student-t + Gumbel + quantile + multi-scale + decomposition + vol features + v2 SDEs
**Result**: ED plateau at 0.005 by step 3K. Aux weight 0.5 ate 94% of gradient.
**Fix**: Reduced aux_weight to 0.15. Same plateau.

### Phase 3: FracOU Scale Bug
**Bug**: fBM increment scaling `sqrt(dt)^(2H)` produced returns 10-50x too small.
**Fix**: Normalize fBM to unit variance, scale by `sigma_t * sqrt(dt)`.
**Impact**: Fixed grad spikes (2000→10), didn't fix plateau.

### Phase 4: Ablation (Exp1-3)
| Exp | Config | ED | Conclusion |
|-----|--------|-----|------------|
| Exp1 | Student-t + Gumbel + v2 SDEs | 0.004 | Best ED |
| Exp2 | Multi-scale + decomp + vol feats | 0.005 | Higher eff_k (4.2) |
| Exp3 | Full v2, LR=1e-4 | 0.005 | Lower nu (10) |

All plateau at same ED. Problem is common to all configs.

### Phase 5: Root Cause Analysis
- v1 and v2 `energy_distance_loss` produce identical values (verified numerically)
- Model has corr(pred_mean, actual) = 0.019 → no conditional signal
- Encoder output norm varies by only 0.24% across samples

### Phase 6: Targeted Fixes (ExpA-E)
| Exp | Fixes | CRPS | Key result |
|-----|-------|------|------------|
| ExpA | CRPS-avg + contrastive | 0.064 | SDE acc 38%, best encoder |
| ExpB | FiLM + CRPS-avg | 0.067 | Grad instability, killed |
| ExpC | All fixes | 0.067 | Best calibration (coverage ±2%) |
| ExpD | Moment matching | 0.067 | First conditional signal (corr 0.093) |
| ExpE | Single Student-t | 0.064 | Same CRPS, simpler head |

### Phase 7: Fine-tuning
| Exp | Strategy | Pretrained | Val CRPS | Note |
|-----|----------|-----------|----------|------|
| FT-D | Baseline | ExpD | 0.0377 | Flat from step 500 |
| **FT-F** | **Aggressive encoder** | **ExpE** | **0.0375** | **Best — only one that moved** |
| FT-H | Heavy NLL + anneal | ExpD | 0.0377 | NLL didn't help |
| FT-EF | GARCH pretrained | ExpF | 0.0380 | Vol conditioning didn't transfer |

### Phase 8: Non-Markovian SDEs (ExpF)
- GARCH(1,1) + Momentum SDEs with real within-context signal
- Vol conditioning corr 0.80, but CRPS plateau at 0.062 (same oracle floor)
- Fine-tuning on BTC: val CRPS 0.0380 (flat across 50K steps, 100 val checks)

### Final Test Set Results (Best Model: FT-F)
| Metric | Model | Naive (global) | Naive (per-horizon) |
|--------|-------|---------------|-------------------|
| **CRPS** | 0.0288 | 0.0287 | 0.0285 |
| Coverage 50% | 57.0% | — | — |
| Coverage 80% | 84.5% | — | — |
| Coverage 90% | 93.3% | — | — |
| Coverage 95% | 96.9% | — | — |
| ECE | 0.012 | — | — |

---

## Phase 9: v3 — Real Multi-Asset Pretraining

**Motivation**: Synthetic SDEs can't teach temporal patterns. Pretrain on real financial data instead.

**Setup**: 268 assets (36 crypto, 131 US equity, 31 ETFs, 45 EU equity, 15 forex, 10 commodity). 6-channel OHLCV features (returns, intraday range, body ratio, volume ratio, trailing vol, momentum). 1.56M train samples, 75-day context. NLL + CRPS loss on single targets. Asset-type classifier + vol regressor auxiliaries. Student-t head.

**Data sources**: Binance + CryptoCompare (crypto), yfinance (equities, ETFs, forex, commodities). All free.

| Metric | v3 Result |
|--------|-----------|
| NLL | -1.74 |
| CRPS | 0.031 |
| ECE | 0.009 |
| Coverage 90% | 93.1% |
| Corr(pred_std, \|error\|) | 0.50 |
| Pred mean std | 0.0008 (collapsed) |
| Corr(pred_mean, actual) | -0.012 |
| Asset-type accuracy | 94% |

**Key Finding**: Excellent **uncertainty calibration** across asset types (corr=0.50 — crypto gets wide sigma, forex gets narrow). But predicted mean is always ~0 — no directional signal from OHLCV features at 3-7 day horizons. The model correctly learns "I can't predict direction, so I'll get the spread right."

---

## Phase 10: v4 — Multi-Horizon Curve Prediction (1-30 days)

**Motivation**: Longer horizons have better drift SNR (SNR ~ sqrt(h)). Predict full term structure of distributions.

**Setup**: Same 268 assets, 120-day context (up from 75), 30 horizons (1-30 days). Model decodes all 30 horizons simultaneously via batched cross-attention queries. Added explicit MSE loss on predicted mean, weighted by sqrt(h/30). Initialized from v3 weights.

### v4a (mean_mse_weight=1.0, no regularization)
- Train pred_mean_std: 0.006 (growing — mu alive on training data)
- **Val pred_mean_std: 0.0014 and declining** — mu predictions don't generalize
- Val mean_mse: 0.018 (flat — no improvement on unseen data)
- **Diagnosis**: Model memorizes training return directions but can't transfer to validation

### v4b (mean_mse_weight=0.3, dropout=0.3, weight_decay=0.05, 15% synthetic GARCH/Momentum, min_mse_horizon=10)
- Train pred_mean_std: 0.008 (higher with more regularization pressure)
- **Val pred_mean_std: 0.001 — still collapsing**
- Val mean_mse: 0.018 (identical to v4a — regularization didn't help)

**Conclusion**: OHLCV price features do not contain generalizable directional signal at any horizon 1-30 days. The model can learn to memorize training return directions but this doesn't transfer. This is consistent with weak-form market efficiency. The mean MSE loss only causes overfitting.

---

## Phase 11: v5 — Relative Return Prediction (complete)

**Motivation**: Absolute returns are unpredictable from price features, but relative returns (asset vs peer group) might be. Shared market factors cancel, leaving idiosyncratic signal (volume surges, momentum divergence).

**Setup**: 413 assets (61 crypto, 224 equity, 57 EU equity, 15 forex, 26 commodity), 120-day context, 30 horizons. Target changed from absolute returns to relative: `Y_relative = Y_asset - mean(Y[same date, same class])`. Computed offline over 25,395 (date, class) groups. 99.7% of samples adjusted. Initialized from v3. NLL + CRPS loss. mean_mse_weight=0.3, min_mse_horizon=5. 20 epochs, batch_size=512, lr=3e-4. Trained on LaRuche A100.

**Key Evaluation Metrics**:
- **Rank IC**: Spearman correlation between predicted and actual relative returns, averaged across dates
- **Long-short portfolio**: buy top quintile predicted, short bottom quintile — PnL
- **IC by horizon**: which horizons have cross-sectional signal?

### v5 Results (step 39,000, OOS: 2025-01 to 2026-03, 81,106 samples, 412 assets)

| Metric | Value |
|--------|-------|
| **Rank IC (10d)** | **0.092** (t-stat=11.8) |
| Rank IC (1d) | 0.045 (t-stat=5.4) |
| Rank IC (30d) | 0.119 (t-stat=14.6) |
| IC positive days | 70% |
| Monthly IC positive | 13/15 months |
| **L/S Sharpe** | **4.55** (no costs) |
| L/S cumulative | 431% |
| L/S win rate | 61% |
| L/S max drawdown | -65.6% |
| NLL | -1.375 |
| Pred mean std | 0.0034 |
| Corr(pred_mean, actual) | 0.039 |
| Corr(pred_std, \|error\|) | 0.299 |
| Coverage 50% | 56.9% |
| Coverage 80% | 87.2% |
| Coverage 90% | 93.7% |
| Coverage 95% | 96.8% |
| ECE | ~0.01 |
| Nu (mean) | 2.11 (heavy tails, near floor) |

### IC by Asset Class (h=10d)

| Asset Class | Rank IC | Interpretation |
|-------------|---------|----------------|
| **Crypto** | **0.144** | Strong signal — drives overall result |
| Commodity | 0.084 | Moderate signal |
| Forex | 0.007 | No signal |
| Equity | 0.003 | No signal |

### Key Findings

1. **Cross-sectional signal exists in OHLCV** — IC=0.09 at 10d, well above threshold (0.02)
2. **Signal is almost entirely crypto** (IC=0.14) with some commodity contribution (IC=0.08). Equity and forex show no signal.
3. **IC increases with horizon** — 0.045 at 1d → 0.12 at 30d, consistent with drift SNR scaling as sqrt(h)
4. **Distributional calibration remains excellent** — PIT near-uniform, coverage well-calibrated, ECE ~0.01
5. **Nu ≈ 2 everywhere** — model pushes to minimum allowed df, indicating very heavy-tailed relative returns
6. **Mu predictions are small but cross-sectionally informative** — absolute corr=0.039 is low, but ranking signal (IC=0.09) is strong
7. **L/S Sharpe of 4.55 is pre-cost** — realistic Sharpe after transaction costs, slippage, and market impact likely 1.0-2.0

### v5 vs Previous Phases

| Aspect | v3 (absolute) | v4 (multi-horizon absolute) | v5 (relative) |
|--------|---------------|----------------------------|---------------|
| Directional signal | None (corr=-0.01) | Memorized, didn't generalize | **IC=0.09** (cross-sectional) |
| Uncertainty calibration | corr=0.50 | corr~0.30 | corr=0.30 |
| What it proves | OHLCV → good vol estimates | Absolute direction is unpredictable | **Relative ranking is predictable** |

---

## What Worked Across All Phases

| Technique | Where | Impact |
|-----------|-------|--------|
| Real multi-asset data | v3+ | Much richer representations than synthetic SDEs |
| 6-channel OHLCV features | v3+ | Universal across all asset types |
| Student-t head | v2+ | Better than MoG, captures heavy tails |
| Asset-type classifier | v3+ | 94%+ accuracy, forces encoder discrimination |
| Uncertainty calibration | v3+ | Corr(pred_std, \|error\|) = 0.50 — real cross-asset knowledge |
| Multi-horizon decoding | v4+ | Efficient batched 30-query cross-attention |
| **Relative return targets** | **v5** | **IC=0.09 cross-sectional signal from OHLCV** |

## What Didn't Work

| Technique | Why |
|-----------|-----|
| Mean MSE loss on absolute returns | Model memorizes training directions, doesn't generalize (v4a/v4b) |
| Synthetic data mixing for regularization | Synthetic features are distinguishable from real (v4b) |
| Higher dropout/weight decay | Doesn't fix the fundamental lack of directional signal (v4b) |
| Longer horizons alone | SNR improves but still insufficient for absolute prediction (v4) |
| Absolute return prediction from OHLCV | Weak-form EMH holds — no directional signal at any horizon 1-30d (v3, v4) |

## Phase 12: v6 — Crypto-Focused + Derivative Features (in progress)

**Motivation**: v5 signal is almost entirely crypto (IC=0.14). Focus on crypto and add 3 new derivative/microstructure features to boost cross-sectional signal.

**Setup**: Crypto-only (~60 assets), 9 input channels (6 OHLCV + 3 new), 120-day context, 30 horizons. Initialized from v5 weights with zero-padded patch embedding (30→45 input dim). Daily resolution.

**New Features**:
| Channel | Feature | Source | History | Signal |
|---------|---------|--------|---------|--------|
| 6 | Taker buy ratio | Binance kline (field 9/5) | Since listing | Order flow direction |
| 7 | Funding rate | Binance Futures API | 2019-2020+ | Crowded positioning reversal |
| 8 | OI change | Binance Futures API | 2019-2020+ | Positioning intensity |

**Training Recipe**: Two-phase from v5 checkpoint:
- Phase A (5K steps): Only patch embed + head unfrozen — new channel weights learn
- Phase B (up to 45K steps): Full fine-tuning with LLRD=0.8, early stopping (patience=10)
- Random feature masking (p=0.15) on channels 6-8 for regularization
- No asset classifier (single class)

**Key Evaluation**: Same metrics as v5 for comparison + feature ablation (zero each new channel, measure IC delta).

**Success Criteria**: Crypto IC > 0.16 (v5 baseline: 0.144), at least 1 new channel shows measurable IC contribution.

**Status**: Code complete, dataset pending (need to fetch funding + OI data, build dataset, run on LaRuche).

---

## Next Steps

1. **v6 execution** — Fetch data, build dataset, train on LaRuche, evaluate
2. **v7: 4h granularity** — If v6 features help, add 4h bars for 6x more data
3. **Transaction cost analysis** — Validate L/S strategy survives real-world costs
4. **Publication** — Write up v1→v6 progression
