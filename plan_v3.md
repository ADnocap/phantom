# Phantom v3: Real Multi-Asset Pretraining with Universal Features

## Motivation

Synthetic SDE pretraining reaches oracle CRPS but doesn't transfer to BTC because Markovian SDEs have no within-context temporal signal. The model learns marginal distributions, not temporal patterns. The fix: pretrain on **real financial data** with **rich features** derived from OHLCV data that's universally available across all asset types.

---

## Universal Features from OHLCV

Every data source (ccxt for crypto, yfinance for equities/forex/commodities) provides **Open, High, Low, Close, Volume** per day. From these we can derive features that are meaningful for ALL asset types:

| Channel | Feature | Formula | What it captures |
|---------|---------|---------|-----------------|
| 0 | **Close-to-close log return** | `log(close_t / close_{t-1})` | Price direction and magnitude |
| 1 | **Intraday range** | `(high - low) / close` | Intraday volatility (Parkinson-like) |
| 2 | **Body ratio** | `(close - open) / (high - low + 1e-10)` | Bullish/bearish candle strength [-1, 1] |
| 3 | **Log volume ratio** | `log(volume / rolling_median_30d_volume)` | Volume relative to recent norm |
| 4 | **Trailing realized vol (30d)** | `std(returns[-30:]) * sqrt(252)` | Current volatility regime |
| 5 | **Trailing momentum (10d)** | `sum(returns[-10:])` | Recent trend direction |

**All 6 features are computable from OHLCV data alone** — no special APIs, no asset-specific data sources. They work for crypto, equities, forex, and commodities identically.

### Volume handling for forex
Forex volume from yfinance is 0 (OTC market). Options:
- **Option A**: Set volume ratio channel to 0 for forex assets (model learns to ignore it)
- **Option B**: Drop volume channel entirely (5 channels instead of 6)
- **Recommended**: Option A — the model should handle missing features gracefully. During pretraining it sees assets with and without volume, and learns accordingly.

---

## Data Sources (all free, no API keys needed)

### Crypto (daily, via ccxt + CryptoCompare)

**Active pairs**: ccxt/Binance (already in codebase). Extend to top 20 coins.
**Historical + delisted coins**: CryptoCompare free tier — 100K calls/month, 5000+ coins including delisted, true daily OHLCV.

```python
# ccxt (already have)
import ccxt
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1d', since=...)

# CryptoCompare (for broader coverage + delisted)
import requests
url = 'https://min-api.cryptocompare.com/data/v2/histoday'
r = requests.get(url, params={'fsym': 'ETH', 'tsym': 'USD', 'limit': 2000})
# Returns: time, open, high, low, close, volumefrom, volumeto
```

| Source | Assets | History | Samples (75d windows × 3 horizons) |
|--------|--------|---------|-------------------------------------|
| Binance (ccxt) | 20 active coins | 2017-2025 | ~144K |
| CryptoCompare | 50+ coins incl. dead | 2014-2025 | ~200K |
| **Total crypto** | | | **~300K** |

### Equities + ETFs + Forex + Commodities (daily, via Stooq)

**Stooq** (stooq.com) provides free bulk database downloads — no API, no rate limits, just download CSVs. Covers US/European stocks, ETFs, forex, commodities, indices.

```python
# Option 1: Bulk download from https://stooq.com/db/h/
# Download "World" or "US" daily database — comes as a zip of CSVs
# One CSV per asset, columns: Date,Open,High,Low,Close,Volume

# Option 2: Per-asset via pandas_datareader
from pandas_datareader import data as pdr
df = pdr.DataReader('AAPL.US', 'stooq', start='2010-01-01')
# Returns: Date, Open, High, Low, Close, Volume
```

| Universe | Assets | History | Samples |
|----------|--------|---------|---------|
| US stocks (Stooq .us) | ~3000 | 2000-2025 | ~3000 × 3600 × 3 = **32M** (subsample!) |
| Major ETFs | ~30 | 2005-2025 | ~30 × 5000 × 3 = **450K** |
| Forex pairs | ~15 | 2000-2025 | ~15 × 6000 × 3 = **270K** |
| Commodities | ~10 | 2005-2025 | ~10 × 5000 × 3 = **150K** |
| European stocks | ~500 | 2005-2025 | ~500 × 5000 × 3 = **7.5M** (subsample!) |

### Sampling Strategy

Raw total is ~40M+ samples (dominated by equities). This is too much and too imbalanced. Subsample to create a balanced dataset:

| Asset class | Target samples | Strategy |
|-------------|---------------|----------|
| Crypto | 300K (all) | Use everything |
| US equities | 500K | Random 150 stocks, all windows |
| European equities | 200K | Random 50 stocks, all windows |
| ETFs | 400K (all) | Use everything |
| Forex | 270K (all) | Use everything |
| Commodities | 150K (all) | Use everything |
| **Total** | **~1.8M** | Balanced across asset classes |

Note: Forex volume from Stooq may be 0 or unreliable — set log_volume_ratio channel to 0 for forex, same as we discussed.

---

## Data Pipeline Implementation

### New file: `src/real_data.py`

```python
class RealAssetDataset(Dataset):
    """Multi-asset dataset from real OHLCV data.

    Each sample is a (features, horizon, target) tuple where:
    - features: (context_len, n_channels) — 6-channel OHLCV-derived features
    - horizon: int in {3, 5, 7}
    - target: float — realized forward log-return
    """

    def __init__(self, data_dir='data/assets/', context_len=75, horizons=[3,5,7]):
        # Load all pre-processed .npz files
        # Each file: {features: (N, 6), dates: (N,)}
        # Create rolling windows across all assets
        # Store: X (context_len, 6), H (horizon), Y (forward return)
        ...
```

### Data fetching scripts: `scripts/data/`

```
scripts/data/
├── fetch_crypto.py         # ccxt (Binance) + CryptoCompare API
├── fetch_stooq.py          # Download + parse Stooq bulk database
├── build_dataset.py        # Compute features, create rolling windows, save
└── validate_data.py        # Check for NaN, gaps, stats
```

**fetch_crypto.py**:
```python
"""Fetch crypto OHLCV from Binance (active) + CryptoCompare (historical/dead).
Saves data/raw/crypto/{symbol}.npz with {dates, open, high, low, close, volume}.
"""
BINANCE_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', ...]
CRYPTOCOMPARE_SYMBOLS = ['BTC', 'ETH', 'LTC', 'XRP', ...]  # broader set
```

**fetch_stooq.py**:
```python
"""Download and parse Stooq daily database.

1. Download bulk zip from https://stooq.com/db/h/ (select 'daily', 'world' or 'us')
2. Extract CSVs — one per asset
3. Parse and save as data/raw/stooq/{ticker}.npz

Alternative: use pandas_datareader per-asset (slower but no manual download):
    from pandas_datareader import data as pdr
    df = pdr.DataReader('AAPL.US', 'stooq', start='2010-01-01')
"""
```

**build_dataset.py**:
```python
"""Process raw OHLCV → 6-channel features → rolling windows → final dataset.

1. Load all raw .npz files from data/raw/
2. Compute 6-channel features via compute_ohlcv_features()
3. Create rolling windows (75-day context + 3/5/7-day forward)
4. Assign asset_type labels (crypto=0, equity=1, forex=2, commodity=3)
5. Save as data/processed/train.npz, val.npz, test.npz
"""
```

### Feature computation: `src/features.py`

```python
def compute_ohlcv_features(open, high, low, close, volume, window=30):
    """Compute 6-channel features from OHLCV data.

    Args:
        open, high, low, close: (N,) daily prices
        volume: (N,) daily volume (0 for forex)

    Returns:
        features: (N-1, 6) array
    """
    n = len(close)
    log_returns = np.diff(np.log(close))                        # Channel 0
    intraday_range = (high[1:] - low[1:]) / close[1:]          # Channel 1
    body = (close[1:] - open[1:]) / (high[1:] - low[1:] + 1e-10)  # Channel 2

    # Volume ratio (log scale, relative to 30-day median)
    vol_median = pd.Series(volume[1:]).rolling(30, min_periods=1).median().values
    log_vol_ratio = np.log(volume[1:] / (vol_median + 1e-10) + 1e-10)
    log_vol_ratio[volume[1:] == 0] = 0  # Handle forex (no volume)

    # Trailing realized vol (30-day)
    trailing_vol = pd.Series(log_returns).rolling(30, min_periods=5).std().values * np.sqrt(252)

    # Trailing momentum (10-day cumulative return)
    momentum = pd.Series(log_returns).rolling(10, min_periods=1).sum().values

    features = np.column_stack([
        log_returns,      # 0: returns
        intraday_range,   # 1: intraday vol
        body,             # 2: candle body
        log_vol_ratio,    # 3: relative volume
        trailing_vol,     # 4: realized vol
        momentum,         # 5: momentum
    ])
    return features.astype(np.float32)
```

---

## Pretraining Changes

### Architecture
- Same encoder → cross-attention decoder → Student-t head
- **`n_input_channels=6`** (set during pretraining, not expanded later)
- Patch embedding: `Linear(patch_len * 6, d_model)`

### Loss
Since real data has single outcomes (not 128 branches):
- **Primary: NLL** (closed-form Student-t, strong conditional gradient)
- **Secondary: CRPS** (closed-form Student-t)
- **Contrastive loss**: encoder representations for different assets should differ
- **Encoder variance penalty**: prevent constant encoder output
- **Drop**: energy distance, moment matching (require branches)

### Auxiliary tasks
- **Drop**: SDE classifier (no SDEs)
- **Add**: Asset-type classifier (crypto/equity/forex/commodity — 4 classes)
- **Keep**: Volatility regressor (realized vol from context)
- **Add**: Return-sign classifier (binary: positive/negative forward return)

### Training
```bash
python scripts/train/train_pretrain.py \
    --data_mode real_assets \
    --data_dir data/assets/ \
    --n_input_channels 6 \
    --head_type student_t \
    --nll_weight 1.0 \
    --contrastive_weight 0.3 \
    --enc_var_weight 0.1 \
    --epochs 10 \
    --batch_size 256 \
    --lr 3e-4
```

---

## Fine-tuning Changes

### BTC-specific features
During fine-tuning, the model receives the same 6 OHLCV-derived channels — no channel expansion needed because we pretrained with 6 channels.

However, we CAN add BTC-specific channels (7+) via zero-initialized expansion if desired:
- Channel 6: Funding rate (Binance perps API, free)
- Channel 7: Open interest change
- Channel 8: On-chain metrics (if API available)

### Fine-tuning strategy
Use what worked best: aggressive encoder (no freeze, full LR, no L2-SP).

---

## What Pretrained Model to Use

**Start fresh.** The existing checkpoints were trained with:
- `n_input_channels=1` (returns only) — can't accept 6-channel input
- Energy distance on 128-branch synthetic futures — different loss paradigm
- SDE classifier head — not applicable to real data

The architecture is the same but the weights won't transfer due to the input dimension change and different training objective. A fresh model trained on 6M real samples with 6-channel features should learn much richer representations than any synthetic-pretrained model.

---

## Dependencies

```
# Add to requirements.txt:
pandas_datareader   # Stooq data reader
requests            # CryptoCompare API
```

No `yfinance` needed — using Stooq (via `pandas_datareader`) and CryptoCompare (via `requests`) instead.

## Implementation Checklist

1. [ ] Add `pandas_datareader` to requirements.txt
2. [ ] Write `scripts/data/fetch_crypto.py` — ccxt + CryptoCompare fetcher
3. [ ] Write `scripts/data/fetch_stooq.py` — Stooq bulk data download/parse
4. [ ] Write `src/features.py` — universal OHLCV → 6-channel feature computation
5. [ ] Write `scripts/data/build_dataset.py` — features → rolling windows → train/val/test
6. [ ] Write `scripts/data/validate_data.py` — NaN/gap checks, stats summary
7. [ ] Write `src/real_data.py` — `RealAssetDataset` class
8. [ ] Modify `train_pretrain.py` — add `--data_mode real_assets` path
9. [ ] Add asset-type classifier + return-sign classifier auxiliary tasks
10. [ ] Fetch all data, build dataset, validate
11. [ ] Launch pretraining on LaRuche
12. [ ] Fine-tune on BTC (same 6 channels, no expansion needed)
13. [ ] Evaluate and compare against current best (CRPS 0.0288)
