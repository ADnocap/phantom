# Phantom

Synthetic pre-training for Bitcoin distributional prediction (3-7 day horizon), adapting JointFM-0.1 methodology to single-asset probabilistic forecasting of BTC/USD.

## Project Status

Phase 1 (synthetic pre-training) implemented and iterating. Key finding: RevIN normalization causes the model to predict the marginal distribution — replaced with no-normalization (JointFM-style) and energy distance loss with branched futures.

## Tech Stack

- **Language**: Python
- **Deep Learning**: PyTorch (`nn.TransformerEncoder`)
- **SDE Simulation**: Custom Euler-Maruyama with Numba JIT
- **Data**: Synthetic SDE trajectories (online generation with branched futures)
- **Evaluation**: Energy distance, CRPS, PIT histogram, coverage

## Architecture Overview

1. **Synthetic data generator** - samples SDE parameters from broad priors, simulates context trajectory, branches N=128 future paths from terminal state (JointFM-style)
2. **Transformer encoder** (8 layers, d=512) with patched log-return input (context: 75 days), NO normalization
3. **Mixture of Gaussians forecast head** (K=5 components) outputting distributional parameters for 3/5/7-day horizons
4. **Energy distance loss** comparing MoG samples vs MC ground truth + auxiliary NLL

## Key Design Decisions

- Work in **log-return space**, not raw prices
- **No normalization** — raw log-returns preserve conditional signal (RevIN destroys it)
- **Branched futures** — 128 MC paths per sample from shared terminal state, NOT 1 path per param set
- **Energy distance** as primary loss (directly compares distributions), NLL as auxiliary (weight 0.1)
- Horizon conditioning via learned embedding, not separate models
- Asymmetric jump priors: downward jumps are larger/more frequent than upward
- Online data generation (infinite stream, no pre-generated shards needed)

## Conventions

- All SDE parameter priors should stay within the ranges specified in README.md Section 3
- Evaluation must include: PIT histogram, CRPS vs baselines, coverage at 90%/95%, energy distance
- Baselines to beat: historical simulation, GARCH(1,1) with Student-t, calibrated Kou jump-diffusion

## Running

```bash
pip install -r requirements.txt

# Train (online generation, recommended):
python train_pretrain.py --data_mode online --samples_per_epoch 1000000 \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_model_samples 256 --nll_weight 0.1 \
    --batch_size 256 --epochs 30
```
