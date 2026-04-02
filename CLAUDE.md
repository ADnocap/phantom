# Phantom

Synthetic pre-training for Bitcoin distributional prediction (3-7 day horizon), adapting JointFM-0.1 methodology to single-asset probabilistic forecasting of BTC/USD.

## Project Status

Phase 1 (synthetic pre-training) and Phase 2 (real BTC fine-tuning) complete. Model produces well-calibrated distributional forecasts (ECE=0.005, coverage within 1% of targets at all levels).

## Architecture

- **Encoder**: 8-layer transformer (d=512, 8 heads) with patched input (5-day patches), no normalization
- **Decoder**: 2-layer cross-attention decoder (horizon query attends to encoder patches)
- **Head**: MoG (K=5 Gaussians) — outputs (π, μ, σ) per component
- **Anti-collapse**: condition dropout (15%), SDE type classifier, volatility regressor
- **Inference**: supports classifier-free guidance (forward_cfg) for sharper predictions

## Training

- **Phase 1**: Energy distance loss on 128 branched MC futures per sample + auxiliary NLL + SDE classification + vol regression. Online generation, no pre-generated shards.
- **Phase 2**: Mixed batches (70% synthetic ED + 30% real BTC CRPS). Gradual unfreezing with LLRD. L2-SP regularization. Real data: Bitstamp 2015 + Binance 2017-2026 via ccxt.

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

# Pre-train
python scripts/train/train_pretrain.py --data_mode online \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_decoder_layers 2 --cond_drop_prob 0.15 --aux_weight 0.5

# Fine-tune on real BTC
python scripts/train/train_finetune.py --pretrained checkpoints/best.pt

# Evaluate
python scripts/eval/eval_model.py
python scripts/eval/visualize_btc.py --checkpoint checkpoints_ft/best.pt
```
