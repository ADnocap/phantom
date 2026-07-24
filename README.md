# Phantom: Distributional Forecasting for Cross-Sectional Cryptocurrency Return Prediction

A Transformer-based model that predicts full Student-*t* distributions of cross-sectional **relative** returns across 362 cryptocurrencies at horizons of 1-30 days, using only standard OHLCV price and volume features.

**Paper**: `paper/main.tex` | **Release**: [v1.0 (checkpoints + data)](https://github.com/ADnocap/phantom/releases/tag/v1.0)

---

## Key Results (out-of-sample Jan 2025 - Mar 2026; survivor-biased universe — see Limitations)

| Metric | Value |
|--------|-------|
| **Rank IC (10d)** | 0.124 (Newey-West *t* = 8.9; conservative overlap-adjusted lags still give *t* > 6) |
| **Long-Short Sharpe (gross, horizon-corrected)** | ~4.1 upper bound (see Limitations; earlier releases reported 11.8-13.0 due to a 10-day-return annualization error) |
| **Win Rate (10-day windows)** | 74.7% of overlapping 10-day holding windows |
| **Quintile / Decile Monotonicity** | 1.000 / 0.988 |
| **Breakeven Cost (horizon-corrected)** | ~34 bps one-way |
| **IC Positive Months** | 15/15 |

### Main Results

![Main Results](plots/fig1_main_results.png)

**(a)** Rank IC increases with horizon (0.08 at 1d to 0.17 at 30d; longer horizons overlap more heavily). **(b)** Cumulative overlapping 10-day L/S spreads — indicative of signal strength, not an implementable equity curve. **(c)** Near-perfect decile monotonicity (Spearman 0.99). **(d)** Phantom has the highest IC among directional baselines; a naive low-volatility sort achieves a higher raw IC (0.165 vs 0.124) — Phantom's incremental value is concentrated in mid/high-volatility names (see Limitations).

### Robustness

![Robustness](plots/fig2_robustness.png)

IC positive in all 15 months. Rolling 60-day horizon-corrected Sharpe always > 0 (min ~0.9). Quintile-membership turnover stable at ~38%/day.

---

## Limitations (read before citing the numbers)

1. **Survivorship bias**: the universe is all Binance USDT pairs with status TRADING at data-fetch time (Mar 2026). Pairs delisted at any point 2017-2026 — including during the test window — are excluded retroactively from training, testing, and the cross-sectional demeaning that defines the target. Reported IC and spreads are upper bounds on live performance.
2. **Overlapping-return annualization**: the L/S series is built from 10-day forward returns sampled daily. Earlier claims (Sharpe 11.8-13.0, 925% annualized, 74.7% *daily* win rate, 336 bps breakeven, 2.53%/day spread) annualized these as daily returns, a ~sqrt(10) Sharpe inflation. Corrected: gross Sharpe ~4.1, ~93% annualized gross, breakeven ~34 bps one-way.
3. **Implementability**: legs are equal-weighted with no liquidity screen; many bottom-quintile pairs have no margin/perp market (the short leg is partly unimplementable); no spread, impact, or borrow costs are modeled. The signal's incremental IC is concentrated in high-volatility, low-liquidity names. Realistic net-of-cost performance is plausibly in the 1-2 Sharpe range.
4. **Low-vol overlap**: the signal is 0.63 rank-correlated with inverse trailing volatility, and a naive low-vol sort has higher raw IC (0.165). Phantom's independent contribution is significant only within mid/high-vol terciles (t = 2.6 / 4.1).
5. **Inference**: Newey-West lags (Andrews rule, 5) under-correct for the MA(9) overlap in the daily IC series; conservative lags reduce the IC t-stat from 8.9 to roughly 6-7. There are ~43 independent 10-day periods in the sample.

---

## Architecture

![Architecture](plots/fig_architecture.png)

- **Encoder**: 8-layer pre-norm Transformer on 24 patches of 120-day, 6-channel OHLCV input (512d, 8 heads)
- **Decoder**: 2-layer cross-attention decoding all 30 horizons simultaneously
- **Head**: Student-*t* output (mu, sigma, nu) per horizon - 3 parameters, no mixture
- **Anti-collapse**: Condition dropout (p=0.15) + encoder variance penalty
- **Parameters**: 31.7M

## Approach

The model predicts **relative returns** (asset return minus cross-sectional mean), isolating idiosyncratic signal from shared market factors. This is the key insight that unlocks cross-sectional predictability:

- **Absolute returns** from OHLCV are unpredictable at any horizon (consistent with weak-form EMH)
- **Relative returns** contain ranking signal concentrated in crypto (IC = 0.124 on a survivor universe; partially overlapping the low-volatility effect — see Limitations)

### Two-Stage Training

1. **Stage 1**: Pretrain on 413 assets (crypto + equities + forex + commodities) with 6-channel OHLCV features
2. **Stage 2**: Fine-tune on 362 crypto assets only - the signal concentrates in crypto

![Training Dynamics](plots/fig5_training.png)

---

## Ablation: What Works and What Doesn't

Eight model versions systematically test the design space:

| Version | Change | IC (10d) | Sharpe | Finding |
|---------|--------|----------|--------|---------|
| v1-v2 | Synthetic SDE pretrain | - | - | Oracle CRPS on synthetic, zero transfer to real |
| v3 | Real multi-asset pretrain | 0.00 | - | Good calibration, no directional signal |
| v4 | Multi-horizon absolute returns | 0.00 | - | Memorizes training, doesn't generalize |
| **v5** | **Relative returns** | **0.09** | **4.6** | **Signal unlocked** |
| v6 | + Funding rate, taker buy | 0.14 | 5.5 | Features don't help; crypto-only helps |
| v7 | 4h bars | 0.10 | 2.6 | Worse: 99.7% sample overlap, signal is daily |
| **v8** | **362 crypto assets** | **0.12** | **~4.1*** | **Breadth improves L/S diversification; IC slightly lower than v6** |

\* Sharpe values in this table were originally computed by annualizing overlapping 10-day returns with sqrt(365), inflating them by ~sqrt(10); the corrected v8 gross figure is shown. All are gross, equal-weighted, on a survivor universe.

---

## Quick Start

```bash
pip install -r requirements.txt

# Download checkpoint and data from release
# https://github.com/ADnocap/phantom/releases/tag/v1.0

# Evaluate
python scripts/eval/eval_v5.py \
    --checkpoint checkpoints_v8/best.pt \
    --test_data data/processed_v8/test.npz

# Full publication analysis (transaction costs, baselines, robustness)
python scripts/eval/publication_analysis.py \
    --checkpoint checkpoints_v8/best.pt \
    --test_data data/processed_v8/test.npz

# Train from scratch (requires data fetching first)
python scripts/data/fetch_crypto_v8.py --workers 6
python scripts/data/build_dataset_v8.py
python scripts/train/train_pretrain.py \
    --data_mode v5_real_assets \
    --real_data_dir data/processed_v8/ \
    --init_from checkpoints_v5/best.pt \
    --head_type student_t \
    --n_input_channels 6 \
    --context_len 120 --max_horizon 30 \
    --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 \
    --batch_size 512 --epochs 50 --lr 3e-4
```

---

## Project Structure

```
phantom/
├── src/
│   ├── model.py              # PhantomModel (encoder + cross-attn decoder + Student-t head)
│   ├── losses.py             # NLL, CRPS, energy distance, combined losses
│   ├── features.py           # 6-channel OHLCV feature computation
│   ├── real_data.py          # Dataset classes for processed .npz files
│   ├── data.py               # Online synthetic dataset (v1/v2 only)
│   ├── sde.py                # SDE simulators (v1/v2 only)
│   └── btc_data.py           # BTC OHLCV fetching
├── scripts/
│   ├── train/
│   │   ├── train_pretrain.py # Main training script (v3-v8)
│   │   ├── train_v6.py       # v6 crypto fine-tuning with new features
│   │   ├── train_v7.py       # v7 4h-bar training
│   │   └── train_finetune.py # BTC fine-tuning (v1/v2)
│   ├── eval/
│   │   ├── eval_v5.py        # Cross-sectional evaluation (IC, Sharpe, coverage)
│   │   ├── publication_analysis.py  # Full publication analysis
│   │   └── ...
│   ├── data/
│   │   ├── fetch_crypto_v8.py    # Fetch all Binance USDT pairs
│   │   ├── build_dataset_v8.py   # Build crypto-only dataset
│   │   └── ...
│   └── slurm/                # HPC job scripts (LaRuche A100)
├── paper/
│   └── main.tex              # Paper (ACM sigconf format)
├── notebooks/
│   └── phantom_v5_analysis.ipynb  # Interactive analysis
├── plots/                    # All figures
├── logs/                     # Training CSV logs
├── experiments.md            # Full experiment log (v1-v8)
└── CLAUDE.md                 # Project instructions
```

---

## References

- **Liu, Tsyvinski, Wu (2022)**. Common Risk Factors in Cryptocurrency. *Journal of Finance*, 77(2):1133-1177.
- **Fieberg et al. (2024)**. A Trend Factor for the Cross Section of Cryptocurrency Returns. *JFQA*.
- **Hackmann (2026)**. JointFM: A Joint Foundation Model for Distributional Prediction. [arXiv:2603.20266](https://arxiv.org/abs/2603.20266).
- **Nie et al. (2023)**. A Time Series is Worth 64 Words. *ICLR 2023*.
- **Gneiting & Raftery (2007)**. Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477):359-378.
