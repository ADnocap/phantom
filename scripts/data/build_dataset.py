#!/usr/bin/env python
"""
Process raw OHLCV data into the final train/val/test datasets for Phantom v3.

Pipeline:
  1. Scan data/raw/ for all .npz files
  2. Load OHLCV, validate, compute 6-channel features
  3. Create rolling windows (75-day context + 3/5/7-day forward returns)
  4. Assign asset-type labels
  5. Balance across asset classes via subsampling
  6. Split by time (train/val/test)
  7. Save to data/processed/{train,val,test}.npz

Usage:
  python scripts/data/build_dataset.py
  python scripts/data/build_dataset.py --raw_dir data/raw --output_dir data/processed
  python scripts/data/build_dataset.py --val_cutoff 2024-01-01 --test_cutoff 2025-01-01
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.features import compute_ohlcv_features, validate_ohlcv


# Asset type labels
ASSET_TYPE_CRYPTO = 0
ASSET_TYPE_EQUITY = 1
ASSET_TYPE_FOREX = 2
ASSET_TYPE_COMMODITY = 3

# Target sample budget per asset class
SAMPLE_BUDGET = {
    'crypto': 300_000,
    'equity': 500_000,
    'etf': 400_000,
    'forex': 270_000,
    'commodity': 150_000,
    'eu_equity': 200_000,
}

# Map directory names to asset type labels
DIR_TO_ASSET_TYPE = {
    'crypto': ASSET_TYPE_CRYPTO,
    'equity': ASSET_TYPE_EQUITY,
    'etf': ASSET_TYPE_EQUITY,       # Group ETFs with equities
    'forex': ASSET_TYPE_FOREX,
    'commodity': ASSET_TYPE_COMMODITY,
    'eu_equity': ASSET_TYPE_EQUITY,  # Group EU equities with equities
}


def load_raw_asset(npz_path: Path) -> dict | None:
    """Load and validate a single raw OHLCV .npz file."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        dates = d['dates']
        open_ = d['open'].astype(np.float64)
        high = d['high'].astype(np.float64)
        low = d['low'].astype(np.float64)
        close = d['close'].astype(np.float64)
        volume = d['volume'].astype(np.float64)

        is_valid, issues = validate_ohlcv(open_, high, low, close, volume)
        if not is_valid:
            return None

        return {
            'dates': dates, 'open': open_, 'high': high,
            'low': low, 'close': close, 'volume': volume,
        }
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return None


def make_windows(features: np.ndarray, dates: np.ndarray,
                 context_len: int, horizons: list) -> dict:
    """Create rolling window samples from 6-channel features.

    Args:
        features: (T, 6) float32 array from compute_ohlcv_features
        dates: (T+1,) string array (original dates; features are 1 shorter)
        context_len: context window length
        horizons: list of forecast horizons

    Returns:
        Dict with X, H, Y, dates_end arrays
    """
    T = len(features)
    max_h = max(horizons)

    if T < context_len + max_h:
        return None

    X_list, H_list, Y_list, date_list = [], [], [], []
    log_returns = features[:, 0]  # Channel 0 is log returns

    for start in range(T - context_len - max_h + 1):
        ctx = features[start:start + context_len]  # (context_len, 6)
        for h in horizons:
            fwd = log_returns[start + context_len:start + context_len + h].sum()
            X_list.append(ctx)
            H_list.append(h)
            Y_list.append(fwd)
            # Date at end of context (for temporal splitting)
            # dates[i+1] corresponds to features[i] (offset by 1 from diff)
            date_list.append(dates[start + context_len])

    if not X_list:
        return None

    return {
        'X': np.array(X_list, dtype=np.float32),
        'H': np.array(H_list, dtype=np.int8),
        'Y': np.array(Y_list, dtype=np.float32),
        'dates_end': np.array(date_list),
    }


def scan_raw_dir(raw_dir: Path) -> dict[str, list[Path]]:
    """Scan raw data directory and organize by asset class."""
    assets_by_class = defaultdict(list)

    # Check for crypto
    crypto_dir = raw_dir / 'crypto'
    if crypto_dir.exists():
        for f in sorted(crypto_dir.glob('*.npz')):
            assets_by_class['crypto'].append(f)

    # Check for stooq subdirs
    stooq_dir = raw_dir / 'stooq'
    if stooq_dir.exists():
        for subdir in sorted(stooq_dir.iterdir()):
            if subdir.is_dir():
                cls_name = subdir.name
                for f in sorted(subdir.glob('*.npz')):
                    assets_by_class[cls_name].append(f)

    return dict(assets_by_class)


def process_asset_class(asset_class: str, files: list[Path],
                        context_len: int, horizons: list) -> dict | None:
    """Process all assets in one class, return combined windows."""
    asset_type = DIR_TO_ASSET_TYPE.get(asset_class, ASSET_TYPE_EQUITY)
    all_X, all_H, all_Y, all_dates, all_rv = [], [], [], [], []

    for f in files:
        data = load_raw_asset(f)
        if data is None:
            continue

        features = compute_ohlcv_features(
            data['open'], data['high'], data['low'],
            data['close'], data['volume'],
        )

        windows = make_windows(features, data['dates'], context_len, horizons)
        if windows is None:
            continue

        all_X.append(windows['X'])
        all_H.append(windows['H'])
        all_Y.append(windows['Y'])
        all_dates.append(windows['dates_end'])
        # Realized vol = channel 4 at end of context
        rv = windows['X'][:, -1, 4]  # trailing vol at context end
        all_rv.append(rv)

    if not all_X:
        return None

    X = np.concatenate(all_X)
    H = np.concatenate(all_H)
    Y = np.concatenate(all_Y)
    dates = np.concatenate(all_dates)
    rv = np.concatenate(all_rv)
    asset_types = np.full(len(X), asset_type, dtype=np.int8)

    print(f"  {asset_class}: {len(files)} assets -> {len(X):,} windows")
    return {
        'X': X, 'H': H, 'Y': Y, 'dates_end': dates,
        'asset_type': asset_types, 'realized_vol': rv,
    }


def subsample(data: dict, budget: int, rng: np.random.Generator) -> dict:
    """Randomly subsample to budget if needed."""
    n = len(data['X'])
    if n <= budget:
        return data
    idx = rng.choice(n, budget, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in data.items()}


def temporal_split(data: dict, val_cutoff: str, test_cutoff: str) -> dict:
    """Split data by date into train/val/test."""
    dates = data['dates_end']
    train_mask = dates < val_cutoff
    val_mask = (dates >= val_cutoff) & (dates < test_cutoff)
    test_mask = dates >= test_cutoff

    splits = {}
    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        splits[name] = {k: v[mask] for k, v in data.items() if k != 'dates_end'}
        print(f"    {name}: {mask.sum():,} samples")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Build Phantom v3 dataset")
    parser.add_argument('--raw_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--context_len', type=int, default=75)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--val_cutoff', type=str, default='2024-01-01')
    parser.add_argument('--test_cutoff', type=str, default='2025-01-01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Scanning {raw_dir} for raw OHLCV data...")
    assets_by_class = scan_raw_dir(raw_dir)

    if not assets_by_class:
        print("ERROR: No data found. Run fetch_crypto.py and/or fetch_stooq.py first.")
        return

    for cls, files in assets_by_class.items():
        print(f"  {cls}: {len(files)} files")

    # Process each asset class
    all_data = {}
    for cls, files in assets_by_class.items():
        print(f"\nProcessing {cls}...")
        data = process_asset_class(cls, files, args.context_len, args.horizons)
        if data is not None:
            budget = SAMPLE_BUDGET.get(cls, 200_000)
            data = subsample(data, budget, rng)
            all_data[cls] = data

    if not all_data:
        print("ERROR: No valid windows created.")
        return

    # Combine all asset classes
    print("\nCombining all asset classes...")
    combined = {}
    for key in ['X', 'H', 'Y', 'dates_end', 'asset_type', 'realized_vol']:
        combined[key] = np.concatenate([d[key] for d in all_data.values()])

    total = len(combined['X'])
    print(f"Total: {total:,} samples")

    # Asset type distribution
    for t, name in [(0, 'crypto'), (1, 'equity'), (2, 'forex'), (3, 'commodity')]:
        count = (combined['asset_type'] == t).sum()
        if count > 0:
            print(f"  {name}: {count:,} ({100*count/total:.1f}%)")

    # Temporal split
    print(f"\nSplitting: train < {args.val_cutoff} | val < {args.test_cutoff} | test >= {args.test_cutoff}")
    splits = temporal_split(combined, args.val_cutoff, args.test_cutoff)

    # Shuffle train set
    n_train = len(splits['train']['X'])
    perm = rng.permutation(n_train)
    for key in splits['train']:
        splits['train'][key] = splits['train'][key][perm]

    # Save
    for name, data in splits.items():
        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, **data)
        print(f"Saved {out_path} ({len(data['X']):,} samples, "
              f"X shape: {data['X'].shape})")

    print("\nDone!")


if __name__ == '__main__':
    main()
