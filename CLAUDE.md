# Phantom

Distributional forecasting for financial assets (3-7 day horizons). Two training paradigms:
- **v1/v2 (synthetic)**: Pre-train on SDE-generated branched futures, fine-tune on BTC
- **v3 (real data)**: Pre-train on real multi-asset OHLCV data from 268 assets, fine-tune on BTC

## Project Status

**v1/v2 (complete)**: 15+ experiments across synthetic pretraining and BTC fine-tuning. Best: CRPS 0.0288 on test (2023-07 to 2026), matching post-hoc Gaussian. Well-calibrated (ECE=0.012) but no conditional signal.

**v3 (in progress)**: Real multi-asset pretraining on 1.56M samples from 268 assets (crypto, equities, ETFs, forex, commodities) with 6-channel OHLCV features. Training on LaRuche HPC.

See `experiments.md` for v1/v2 experiment log. See `plan_v3.md` for v3 design.

## Architecture

- **Encoder**: Transformer with patched input, no normalization
- **Decoder**: Cross-attention decoder (horizon query attends to encoder patches)
- **Head**: StudentTHead (single Student-t, 3 params) or MixtureHead (MoG/MoT)
- **Anti-collapse**: condition dropout, auxiliary classifiers, volatility regressor
- **Inference**: classifier-free guidance (forward_cfg)

## Training Modes

### Mode 1: Synthetic Pre-training (`--data_mode online`)
- **Data**: Online SDE-generated trajectories with 128 branched futures per sample
- **Features**: 1 channel (log returns) or 4 channels (+ trailing vol)
- **Loss**: Energy distance or CRPS-avg on branches + NLL + contrastive + moment matching
- **Auxiliary**: SDE-type classifier (5-7 classes) + volatility regressor
- **SDEs**: GBM, Merton, Kou, Bates, Regime-Switching, GARCH, Momentum

### Mode 2: Real Multi-Asset Pre-training (`--data_mode real_assets`)
- **Data**: Pre-processed .npz from 268 real assets (crypto, equities, ETFs, forex, commodities)
- **Features**: 6 OHLCV channels (log return, intraday range, body ratio, volume ratio, trailing vol, momentum)
- **Loss**: NLL (primary) + CRPS (secondary) on single realized targets
- **Auxiliary**: Asset-type classifier (4 classes) + return-sign classifier + vol regressor

### Mode 3: BTC Fine-tuning (`train_finetune.py`)
- **Data**: Mixed synthetic + real BTC data
- **Strategy**: Aggressive encoder unfreezing (no freeze, full LR, no L2-SP)

## Project Structure

```
src/
  model.py          PhantomConfig, PhantomModel, heads (StudentT, Mixture)
  losses.py         ED, NLL, CRPS, combined_loss (v1/v2), combined_loss_v3 (real data)
  features.py       6-channel OHLCV feature computation (v3)
  real_data.py      RealAssetDataset for pre-processed .npz files (v3)
  data.py           OnlineDataset (synthetic SDE generation)
  btc_data.py       BTC OHLCV fetching + rolling windows (supports 1/4/6 channels)
  sde.py            SDE simulators (GBM, Merton, Kou, Bates, GARCH, etc.)

scripts/
  train/
    train_pretrain.py   Pre-training (synthetic or real data, v1-v5)
    train_v6.py         v6 crypto-focused fine-tuning from v5 checkpoint
    train_finetune.py   BTC fine-tuning
  eval/
    eval_model.py           Checkpoint evaluation (CRPS, coverage, calibration)
    eval_v5.py              v5 relative return evaluation
    eval_v6.py              v6 crypto evaluation with feature ablation
    visualize_btc.py        BTC prediction visualization
    plot_pretrain_v2.py     Synthetic pretraining metrics (16-panel)
    plot_pretrain_v3.py     Real-data pretraining metrics (16-panel)
    monitor_v3.py           Pull logs from LaRuche + plot live v3 metrics
    plot_finetune.py        Fine-tuning metrics
    plot_finetune_comparison.py  Compare fine-tuning runs
    plot_experiments.py     Compare multiple experiment configs
  data/
    fetch_crypto.py         Fetch crypto OHLCV (Binance + CryptoCompare)
    fetch_crypto_v6.py      Fetch crypto OHLCV + taker buy + funding rates + OI
    fetch_yfinance.py       Fetch equity/ETF/forex/commodity OHLCV (yfinance)
    build_dataset.py        Raw OHLCV -> 6-channel features -> train/val/test .npz
    build_dataset_v6.py     Crypto-only 9-channel dataset builder
    validate_data.py        Dataset quality checks
  slurm/
    train_v3.slurm          v3 real-data pretraining job
    train_v6.slurm          v6 crypto-focused training job
    train_exp*.slurm        v1/v2 synthetic experiments
    ft_exp*.slurm           Fine-tuning experiments

data/                 Gitignored — raw OHLCV + processed datasets
plots/                Generated visualizations
logs/                 Training CSV logs
experiments.md        Full experiment log (v1/v2 findings)
plan_v3.md            v3 design: real multi-asset pretraining plan
```

## Running

```bash
pip install -r requirements.txt

# ─── Option A: Synthetic Pre-training (v1/v2) ───

# v1 baseline
python scripts/train/train_pretrain.py --data_mode online \
    --context_len 75 --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --n_branches 128 --n_decoder_layers 2 --cond_drop_prob 0.15 --aux_weight 0.5

# v2 with all improvements (Student-t, CRPS-avg, contrastive, GARCH SDEs)
python scripts/train/train_pretrain.py --data_mode online \
    --head_type student_t --use_crps_avg --nll_weight 1.0 --aux_weight 0.3 \
    --contrastive_weight 0.3 --enc_var_weight 0.1 \
    --mean_match_weight 1.0 --var_match_weight 0.5 --sde_version v3

# ─── Option B: Real Multi-Asset Pre-training (v3) ───

# Step 1: Fetch data (run once)
python scripts/data/fetch_crypto.py
python scripts/data/fetch_yfinance.py

# Step 2: Build dataset (run once)
python scripts/data/build_dataset.py
python scripts/data/validate_data.py

# Step 3: Pre-train on real data
python scripts/train/train_pretrain.py --data_mode real_assets \
    --real_data_dir data/processed/ --head_type student_t \
    --nll_weight_v3 1.0 --crps_weight_v3 0.5 \
    --asset_cls_weight 0.3 --sign_cls_weight 0.1 --enc_var_weight 0.1 \
    --epochs 10 --batch_size 256 --lr 3e-4

# ─── Fine-tuning on BTC ───

python scripts/train/train_finetune.py --pretrained checkpoints/best.pt \
    --lr_head 1e-4 --lr_encoder 1e-4 --llrd 1.0 \
    --freeze_encoder_steps 0 --l2sp_lambda 0.0 --steps 50000

# ─── Evaluation & Monitoring ───

python scripts/eval/eval_model.py --checkpoint checkpoints/best.pt
python scripts/eval/visualize_btc.py --checkpoint checkpoints_ft/best.pt
python scripts/eval/plot_pretrain_v2.py --log logs/pretrain/train_log.csv
python scripts/eval/plot_pretrain_v3.py --log logs/v3/train_log.csv
python scripts/eval/monitor_v3.py   # Pull live metrics from LaRuche
```

## Key Findings (v1/v2)

1. **Pre-training reaches oracle CRPS** (0.064 vs 0.065 theoretical min)
2. **Markovian SDEs have no within-context signal** — context is irrelevant for prediction
3. **GARCH SDEs create vol conditioning** (corr 0.80) but doesn't transfer to BTC
4. **BTC 3-7d returns are near-Gaussian** — post-hoc Gaussian is near-optimal
5. **Fine-tuning converges in ~500 steps** — 50K steps shows zero improvement beyond

## Key Lessons

- NO normalization (RevIN/MeanAbsScaling destroy conditional signal)
- Single Student-t head (3 params) equals MoG K=5 (15 params)
- Contrastive loss (InfoNCE) is the best anti-collapse technique
- Aggressive encoder unfreezing is the best fine-tuning strategy
- Energy distance gradient vanishes at small ED — use CRPS-avg instead
