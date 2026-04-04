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

## Next Steps

### High Priority
1. **Real multi-asset pretraining** — Pretrain on thousands of real time series (stocks, crypto, forex) instead of synthetic SDEs. This is what Chronos/TimesFM/Lag-Llama do. The synthetic approach fundamentally can't teach temporal patterns that exist in real markets.
2. **Additional features** — On-chain data (MVRV, SOPR, exchange flows), cross-asset signals (S&P correlation, DXY, VIX). Pure price context may not have enough signal at 3-7 day horizons.
3. **Longer horizons** — Try 14-30 day predictions where drift estimation has higher SNR (expected return grows linearly with time, noise grows with sqrt).

### Medium Priority
4. **Much longer context** — 200-365 days instead of 75. Monthly/quarterly patterns might emerge.
5. **Smaller model** — 31M params for 7K BTC training samples is extreme. Try 1-5M param model to reduce overfitting.
6. **Direct GARCH fitting** — Instead of learning distributions end-to-end, estimate GARCH(1,1) parameters from context and use them analytically.

### Research Directions
7. **Quantile regression head** — Replace distributional output with direct quantile predictions (what Moirai 2.0 and TempoPFN do).
8. **Flow matching / diffusion head** — Replace parametric head with non-parametric learned distribution (what Sundial does).
9. **Ensemble** — 5 models with different seeds, Vincentized quantile aggregation.
