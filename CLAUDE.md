# Phantom

Synthetic pre-training for Bitcoin distributional prediction (3-7 day horizon), adapting JointFM-0.1 methodology to single-asset probabilistic forecasting of BTC/USD.

## Project Status

Phase 1 (synthetic pre-training) and Phase 2 (real BTC fine-tuning) complete across 15+ experiment configs. Best model achieves CRPS 0.0288 on test data (2023-07 to 2026), matching the post-hoc Gaussian baseline. Well-calibrated (ECE=0.012, 90%→93% coverage) but no conditional signal — the model outputs the same distribution for all inputs.

See `experiments.md` for full experiment log and findings.

## Architecture

- **Encoder**: Transformer with patched input, no normalization
- **Decoder**: Cross-attention decoder (horizon query attends to encoder patches)
- **Head**: MixtureHead (MoG/MoT) or StudentTHead (single Student-t, 3 params)
- **Anti-collapse**: condition dropout, SDE type classifier, volatility regressor
- **Inference**: classifier-free guidance (forward_cfg) for sharper predictions

### v2 Additions (all backward-compatible, toggled via config)

- **Student-t head** (`--head_type student_t`): single Student-t, 3 params (mu, sigma, nu)
- **Student-t mixture** (`--use_student_t`): power-law tails for MoG components
- **Multi-scale patching** (`--patch_sizes 3 5 15`): parallel patch embeddings at multiple resolutions
- **Input features** (`--n_input_channels 4`): trailing realized vol (7d, 14d, 30d) as extra channels
- **Series decomposition** (`--use_decomposition`): Autoformer-style trend-residual separation
- **Gumbel-Softmax** (`--use_gumbel_softmax`): fully differentiable energy distance
- **Quantile loss** (`--quantile_weight 0.2`): auxiliary pinball loss for calibration
- **FiLM conditioning** (`--use_film`): multiplicative decoder conditioning
- **Contrastive loss** (`--contrastive_weight 0.5`): InfoNCE for encoder discrimination
- **Moment matching** (`--mean_match_weight 1.0 --var_match_weight 0.5`): direct mean/var supervision
- **Encoder variance penalty** (`--enc_var_weight 0.1`): prevents constant encoder output
- **CRPS-avg** (`--use_crps_avg`): closed-form CRPS over all branches (zero noise)
- **New SDEs** (`--sde_version v3`): GARCH(1,1) + Momentum (non-Markovian, context-dependent)
- **Domain adaptation** (`--anneal_real_fraction`): gradual synthetic→real ratio shift

## Key Findings

1. **Pre-training reaches oracle CRPS** (0.064 vs theoretical minimum 0.065) — the model learns synthetic conditional distributions optimally
2. **Markovian SDEs have no within-context signal** — GBM/Merton/Kou/Bates context is irrelevant for prediction; model correctly learns "context doesn't matter"
3. **GARCH SDEs create real vol conditioning** (corr 0.80) but this doesn't transfer to BTC fine-tuning
4. **BTC 3-7 day returns are near-Gaussian** — a post-hoc Gaussian is near-optimal, very little exploitable conditional structure at this horizon
5. **Fine-tuning converges in ~500 steps** then plateaus — 50K steps shows zero improvement beyond step 500

## Key Lessons

- RevIN and MeanAbsScaling both destroy conditional signal — use NO normalization
- Single-scalar targets + NLL = marginal prediction trap — use branched futures + energy distance
- Auxiliary tasks (SDE classification + vol regression) force encoder to extract input-dependent features
- Cross-attention decoder can bypass encoder via residual (posterior collapse analog)
- Energy distance gradient vanishes at small ED (O(ED) scaling, noise floor at ~0.009)
- Contrastive loss (InfoNCE) is the best technique for encoder discrimination
- Aggressive encoder unfreezing (no freeze, no L2-SP, full LR) is the best fine-tuning strategy
- Single Student-t head (3 params) equals MoG K=5 (15 params) — simpler is better

## Training

- **Phase 1**: Energy distance / CRPS-avg loss on 128 branched MC futures per sample + auxiliary NLL + SDE classification + vol regression + contrastive + moment matching. Online generation from 7 SDE families (v3: GBM, Merton, Kou, Bates, Regime-Switching, GARCH, Momentum).
- **Phase 2**: Mixed batches (synthetic + real BTC CRPS). Aggressive unfreezing (no freeze period, full LR, no L2-SP). Real data: Bitstamp 2015 + Binance 2017-2026 via ccxt.

## Project Structure

```
src/              Core library (model, losses, SDE, data)
scripts/train/    Training scripts (pretrain, finetune)
scripts/eval/     Evaluation and visualization
scripts/slurm/    HPC cluster job scripts
plots/            All result plots
logs/             Training metrics CSV
experiments.md    Full experiment log with findings
improvements.md   Literature-based improvement proposals
```

## Running

```bash
pip install -r requirements.txt

# Pre-train (v1 original)
python scripts/train/train_pretrain.py --data_mode online \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_decoder_layers 2 --cond_drop_prob 0.15 --aux_weight 0.5

# Pre-train (v3 with GARCH/Momentum + all improvements)
python scripts/train/train_pretrain.py --data_mode online \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_decoder_layers 2 --cond_drop_prob 0.15 \
    --head_type student_t --use_crps_avg --nll_weight 1.0 --aux_weight 0.3 \
    --contrastive_weight 0.3 --enc_var_weight 0.1 \
    --mean_match_weight 1.0 --var_match_weight 0.5 --sde_version v3

# Fine-tune on real BTC (aggressive encoder)
python scripts/train/train_finetune.py --pretrained checkpoints/best.pt \
    --lr_head 1e-4 --lr_encoder 1e-4 --llrd 1.0 \
    --freeze_encoder_steps 0 --l2sp_lambda 0.0 --steps 50000

# Evaluate
python scripts/eval/eval_model.py --checkpoint checkpoints/best.pt
python scripts/eval/visualize_btc.py --checkpoint checkpoints_ft/best.pt
python scripts/eval/plot_pretrain_v2.py --log logs/pretrain/train_log.csv
python scripts/eval/plot_experiments.py
python scripts/eval/plot_finetune_comparison.py
```
