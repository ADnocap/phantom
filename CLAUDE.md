# Phantom

Synthetic pre-training for Bitcoin distributional prediction (3-7 day horizon), adapting JointFM-0.1 methodology to single-asset probabilistic forecasting of BTC/USD.

## Project Status

Phase 1 (synthetic pre-training) and Phase 2 (real BTC fine-tuning) complete. Model produces well-calibrated distributional forecasts (ECE=0.005, coverage within 1% of targets at all levels).

## Architecture

- **Encoder**: Transformer with patched input, no normalization
- **Decoder**: Cross-attention decoder (horizon query attends to encoder patches)
- **Head**: MixtureHead — Gaussian (MoG) or Student-t (MoT) components, K=5
- **Anti-collapse**: condition dropout, SDE type classifier, volatility regressor
- **Inference**: classifier-free guidance (forward_cfg) for sharper predictions

### v2 Additions (all backward-compatible, toggled via config)

- **Student-t head** (`--use_student_t`): power-law tails instead of Gaussian exp(-x^2) tails
- **Multi-scale patching** (`--patch_sizes 3 5 15`): parallel patch embeddings at multiple resolutions
- **Input features** (`--n_input_channels 4`): trailing realized vol (7d, 14d, 30d) as extra channels
- **Series decomposition** (`--use_decomposition`): Autoformer-style trend-residual separation between encoder layers
- **Gumbel-Softmax** (`--use_gumbel_softmax`): fully differentiable energy distance (no NLL crutch for π)
- **Quantile loss** (`--quantile_weight 0.2`): auxiliary pinball loss for direct calibration pressure
- **New SDEs** (`--sde_version v2`): Multifractal Random Walk + Fractional OU with stochastic vol
- **Domain adaptation** (`--anneal_real_fraction`): gradual synthetic→real ratio shift during fine-tuning

## Training

- **Phase 1**: Energy distance loss on 128 branched MC futures per sample + auxiliary NLL + SDE classification + vol regression. Online generation, no pre-generated shards.
- **Phase 2**: Mixed batches (synthetic ED + real BTC CRPS). Gradual unfreezing with LLRD. L2-SP regularization. Real data: Bitstamp 2015 + Binance 2017-2026 via ccxt.

## Key Lessons

- RevIN and MeanAbsScaling both destroy conditional signal — use NO normalization
- Single-scalar targets + NLL = marginal prediction trap — use branched futures + energy distance
- Auxiliary tasks (SDE classification + vol regression) force encoder to extract input-dependent features
- Cross-attention decoder makes input-independent predictions architecturally impossible
- Fine-tuning converges in ~500 steps; the pre-trained representations transfer well

## Project Structure

```
src/              Core library (model, losses, SDE, data)
scripts/train/    Training scripts (pretrain, finetune)
scripts/eval/     Evaluation and visualization
scripts/slurm/    HPC cluster job scripts
plots/            All result plots (pretrain_*, finetune_*)
logs/             Training metrics CSV
```

## Running

```bash
pip install -r requirements.txt

# Pre-train (v1 original)
python scripts/train/train_pretrain.py --data_mode online \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_decoder_layers 2 --cond_drop_prob 0.15 --aux_weight 0.5

# Pre-train (v2 with all improvements)
python scripts/train/train_pretrain.py --data_mode online \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_decoder_layers 2 --cond_drop_prob 0.15 --aux_weight 0.5 \
    --use_student_t --use_gumbel_softmax --quantile_weight 0.2 \
    --use_decomposition --patch_sizes 3 5 15 --n_input_channels 4 --sde_version v2

# Fine-tune on real BTC
python scripts/train/train_finetune.py --pretrained checkpoints/best.pt

# Fine-tune with domain adaptation
python scripts/train/train_finetune.py --pretrained checkpoints/best.pt \
    --anneal_real_fraction --start_real_fraction 0.1 --end_real_fraction 0.7

# Evaluate
python scripts/eval/eval_model.py
python scripts/eval/visualize_btc.py --checkpoint checkpoints_ft/best.pt
```
