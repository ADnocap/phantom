# Phantom Experiments Log

## The Core Problem

The model learns a good **marginal** distribution but fails to produce **conditional** (input-dependent) predictions. Diagnosed at step 12K-55K across multiple configs:

- Predicted mean std across samples: **0.003** (should be ~0.136 to match target)
- Correlation(predicted_mean, actual): **0.019** (essentially zero)
- Model CRPS 3.6x **worse** than naive Gaussian
- Zeroing encoder output barely changes predictions (mu diff: 0.018)
- Encoder output norm varies by only 0.24% across samples
- SDE classification accuracy 33% proves encoder CAN extract features — they just don't reach the head

### Root Causes (from literature review)

1. **Energy Distance gradient vanishing**: ED gradient ~ O(ED). At ED=0.005 the signal (~0.005) is below the sample noise floor (~0.009 with M=256, N=128). Model does random walk.
2. **Marginal gradient cancellation**: In a batch, different inputs push params in opposing directions for conditional features. Batch average washes out conditional signal. Marginal-improving direction is consistent across inputs.
3. **Decoder residual bypass (posterior collapse analog)**: Cross-attention output is *added* to horizon embedding via residual. Model learns to route prediction through horizon embedding alone, ignoring encoder.

---

## Experiment History

### Phase 1: v2 Full Config (all improvements at once)

**Config**: Student-t + Gumbel-Softmax + quantile loss + multi-scale [3,5,15] + 4-channel input + decomposition + v2 SDEs + aux_weight=0.5

**Result**: Plateaued at ED~0.005 by step 3-5K. Never improved. aux_weight=0.5 caused auxiliary losses to eat 94% of gradient budget (SDE CE=1.85 for 7-class problem).

**Conclusion**: Too many changes at once. Aux loss dominance starved the main loss.

### Phase 2: Reduced aux_weight (0.15)

**Config**: Same as Phase 1 but aux_weight=0.15.

**Result**: More stable gradients, total loss lower (0.24 vs 0.92). But ED still plateaued at ~0.006 by step 5K. No improvement through step 20K.

**Conclusion**: Aux weight wasn't the core issue. The ED loss itself plateaus.

### Phase 3: FracOU Scale Bug

**Discovery**: Fractional OU SDE produced returns 10-50x too small (ctx_std=0.003 vs ~0.05 for other SDEs). Bug in fBM increment scaling: `sqrt(dt_day)^(2*H)` instead of proper normalization.

**Fix**: Normalize fBM increments to unit variance, then scale by `sigma_t * sqrt(dt_day)`.

**Impact**: Fixed grad norm spikes (from 2000+ to <10). But didn't fix the plateau.

### Phase 4: Ablation Experiments (3 parallel)

| Experiment | Config | Steps | ED | NLL | Conclusion |
|---|---|---|---|---|---|
| Exp1: Minimal v2 | Student-t + Gumbel + v2 SDEs, single-scale | 55K | 0.004 | -0.75 | Best ED but still plateaued |
| Exp2: Arch only | Multi-scale + decomp + vol feats, Gaussian, v1 SDEs | 35K | 0.005 | -0.75 | Higher eff_k (4.2 vs 3.3) but same ED |
| Exp3: Low LR | Full v2, LR=1e-4 | 26K | 0.005 | -0.72 | Lower nu (10 vs 80) = heavier tails, but same plateau |

**Conclusion**: Plateau at ED~0.005 is universal across all configs. The problem is the **loss function** (ED noise floor) and **conditioning mechanism** (residual bypass), not the specific features.

### Phase 5: Root Cause Analysis

**Numerical verification**: v1 and v2 `energy_distance_loss` produce identical values and gradients. The code refactor is NOT the cause.

**Key finding**: At ED=0.005, the sample-based estimator noise (std~0.009) exceeds the signal. The model cannot detect which direction improves the conditional distribution.

---

## Phase 6: Targeted Fixes (current)

Three independent problems → three fixes:

| Fix | Problem | Mechanism |
|---|---|---|
| **CRPS-avg as primary loss** | ED noise floor | Closed-form CRPS averaged over all branches. Zero sample noise. |
| **Contrastive loss (InfoNCE)** | Marginal gradient cancellation | Explicitly maximize MI between encoder output and target branches |
| **FiLM conditioning** | Decoder residual bypass | Multiplicative conditioning: `gamma(enc) * feature + beta(enc)`. Cannot be bypassed. |

### Experiment configs (Phase 6)

All use base v1 architecture (single-scale, Gaussian, 5 SDEs) to isolate the effect of each fix.

- **ExpA**: CRPS-avg primary + NLL weight 0.5 + contrastive loss
- **ExpB**: FiLM conditioning + CRPS-avg + NLL 0.5
- **ExpC**: All three fixes combined
