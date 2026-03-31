# Phantom

Synthetic pre-training for Bitcoin distributional prediction (3-7 day horizon), adapting JointFM-0.1 methodology to single-asset probabilistic forecasting of BTC/USD.

## Project Status

Early stage - technical specification complete (README.md), implementation not yet started.

## Tech Stack

- **Language**: Python
- **Deep Learning**: PyTorch (`nn.TransformerEncoder`)
- **SDE Simulation**: `diffrax` (JAX) or custom Euler-Maruyama
- **Data**: `yfinance` / `ccxt` for real BTC prices
- **Evaluation**: `properscoring` (CRPS), `scipy.stats` (PIT)

## Architecture Overview

1. **Synthetic data generator** - samples SDE parameters from broad priors and simulates trajectories using 5 SDE families: GBM, Merton Jump-Diffusion, Kou, Bates, Regime-Switching
2. **Transformer encoder** (6-8 layers) with patched log-return input (context: 30-90 days)
3. **Mixture of Gaussians forecast head** (K=8 components) outputting distributional parameters for 3/5/7-day horizons
4. **Two-phase training**: synthetic pre-training (80%) then mixed fine-tuning with real BTC data (20%)

## Key Design Decisions

- Work in **log-return space**, not raw prices
- Parameter diversity (not path diversity) drives generalization - 1 path per parameter set
- Combined loss: NLL + alpha * CRPS (alpha ~ 0.1-0.5)
- Horizon conditioning via learned embedding, not separate models
- Asymmetric jump priors: downward jumps are larger/more frequent than upward

## Conventions

- All SDE parameter priors should stay within the ranges specified in README.md Section 3
- Evaluation must include: PIT histogram, CRPS vs baselines, coverage at 90%/95%, tail accuracy (1st/99th percentile)
- Baselines to beat: historical simulation, GARCH(1,1) with Student-t, calibrated Kou jump-diffusion

## Running

No runnable code yet. When implemented:

```bash
pip install -r requirements.txt
```
