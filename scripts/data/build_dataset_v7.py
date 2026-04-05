#!/usr/bin/env python
"""
Build crypto-only dataset for Phantom v7 (4h bars).

Pipeline:
  1. Load 4h OHLCV from data/raw/crypto_v7/
  2. Compute 6-channel features (adjusted windows for 4h)
  3. Create rolling windows (720-bar context, 90-bar curve targets)
  4. Cross-sectional demeaning by timestamp
  5. Split by time (train < 2024, val = 2024, test >= 2025)
  6. Save to data/processed_v7/

Usage:
  python scripts/data/build_dataset_v7.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.features import compute_ohlcv_features_4h, validate_ohlcv


CONTEXT_LEN = 720    # 720 4h-bars = 120 days
MAX_HORIZON = 90     # 90 4h-bars = 15 days


def load_4h_data(npz_path: Path) -> dict | None:
    """Load 4h OHLCV data."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        timestamps = d['timestamps']
        open_ = d['open'].astype(np.float64)
        high = d['high'].astype(np.float64)
        low = d['low'].astype(np.float64)
        close = d['close'].astype(np.float64)
        volume = d['volume'].astype(np.float64)

        is_valid, issues = validate_ohlcv(open_, high, low, close, volume)
        if not is_valid:
            return None

        return {
            'timestamps': timestamps, 'open': open_, 'high': high,
            'low': low, 'close': close, 'volume': volume,
        }
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return None


def make_windows_4h(features, timestamps, context_len, max_horizon):
    """Create rolling windows from 4h features.

    Args:
        features: (T, 6) float32 from compute_ohlcv_features_4h
        timestamps: (T+1,) string array (features are 1 shorter due to diff)
        context_len: number of 4h bars in context window
        max_horizon: number of 4h bars to predict

    Returns:
        Dict with X:(N,context_len,6), Y:(N,max_horizon), timestamps_end:(N,)
    """
    T = len(features)
    if T < context_len + max_horizon:
        return None

    X_list, Y_list, ts_list = [], [], []
    log_returns = features[:, 0]

    for start in range(T - context_len - max_horizon + 1):
        ctx = features[start:start + context_len]
        fwd = log_returns[start + context_len:start + context_len + max_horizon]
        curve = np.cumsum(fwd).astype(np.float32)

        X_list.append(ctx)
        Y_list.append(curve)
        # timestamps[i+1] corresponds to features[i] (offset by diff)
        ts_list.append(timestamps[start + context_len])

    if not X_list:
        return None

    return {
        'X': np.array(X_list, dtype=np.float32),
        'Y': np.array(Y_list, dtype=np.float32),
        'timestamps_end': np.array(ts_list),
    }


def compute_relative_returns(Y, timestamps, min_group_size=3):
    """Cross-sectional demeaning per 4h timestamp."""
    N = len(Y)
    Y_relative = Y.copy()

    unique_ts, ts_ids = np.unique(timestamps, return_inverse=True)
    n_groups = len(unique_ts)

    print(f"  Computing relative returns: {N:,} samples, {n_groups:,} timestamp groups...")
    n_adjusted = 0
    for g in range(n_groups):
        mask = (ts_ids == g)
        n = mask.sum()
        if n >= min_group_size:
            group_mean = Y[mask].mean(axis=0)
            Y_relative[mask] -= group_mean
            n_adjusted += n

    print(f"  Adjusted {n_adjusted:,}/{N:,} samples ({100*n_adjusted/N:.1f}%)")
    print(f"  Y_relative stats: mean={Y_relative.mean():.6f}, std={Y_relative.std():.4f}")
    return Y_relative.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build v7 4h crypto dataset")
    parser.add_argument('--raw_dir', type=str, default='data/raw/crypto_v7')
    parser.add_argument('--output_dir', type=str, default='data/processed_v7')
    parser.add_argument('--context_len', type=int, default=CONTEXT_LEN)
    parser.add_argument('--max_horizon', type=int, default=MAX_HORIZON)
    parser.add_argument('--val_cutoff', type=str, default='2024-01-01')
    parser.add_argument('--test_cutoff', type=str, default='2025-01-01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(raw_dir.glob('*.npz'))
    if not files:
        print(f"ERROR: No data in {raw_dir}. Run fetch_crypto_v7.py first.")
        return

    print(f"Building v7 dataset (4h bars, context={args.context_len}, horizon={args.max_horizon})")
    print(f"Found {len(files)} files")

    all_X, all_Y, all_ts = [], [], []
    all_rv, all_asset_ids = [], []
    asset_names = {}

    for file_idx, f in enumerate(files):
        symbol = f.stem
        data = load_4h_data(f)
        if data is None:
            print(f"  {symbol}: invalid data, skipping")
            continue

        features = compute_ohlcv_features_4h(
            data['open'], data['high'], data['low'],
            data['close'], data['volume'],
        )

        windows = make_windows_4h(features, data['timestamps'],
                                  args.context_len, args.max_horizon)
        if windows is None:
            print(f"  {symbol}: too short ({len(features)} bars), skipping")
            continue

        n_windows = len(windows['X'])
        all_X.append(windows['X'])
        all_Y.append(windows['Y'])
        all_ts.append(windows['timestamps_end'])
        all_rv.append(windows['X'][:, -1, 4])  # trailing vol
        all_asset_ids.append(np.full(n_windows, file_idx, dtype=np.int32))
        asset_names[file_idx] = symbol

        n_days = len(data['timestamps']) / 6
        print(f"  {symbol}: {n_windows:,} windows ({len(data['timestamps'])} bars, ~{n_days:.0f} days)")

    if not all_X:
        print("ERROR: No valid windows.")
        return

    # Concatenate metadata (small arrays) but NOT X (too large for RAM)
    all_Y_cat = np.concatenate(all_Y)
    all_ts_cat = np.concatenate(all_ts)
    all_rv_cat = np.concatenate(all_rv)
    all_aid_cat = np.concatenate(all_asset_ids)
    total = len(all_Y_cat)

    # Track per-asset offsets for indexing into all_X list
    offsets = np.cumsum([0] + [len(x) for x in all_X])

    print(f"\nTotal: {total:,} samples from {len(asset_names)} assets")
    print(f"Y shape: {all_Y_cat.shape}")

    # Relative returns (operates on Y only — small)
    print("\nComputing relative returns...")
    Y_relative = compute_relative_returns(all_Y_cat, all_ts_cat)

    # Temporal split
    dates = np.array([ts[:10] for ts in all_ts_cat])
    print(f"\nSplitting: train < {args.val_cutoff} | val < {args.test_cutoff} | test >= {args.test_cutoff}")
    train_mask = dates < args.val_cutoff
    val_mask = (dates >= args.val_cutoff) & (dates < args.test_cutoff)
    test_mask = dates >= args.test_cutoff

    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        n = mask.sum()
        if n == 0:
            print(f"  {name}: 0 samples — skipping")
            continue

        # Build X for this split by gathering from per-asset arrays
        indices = np.where(mask)[0]
        if name == 'train':
            indices = rng.permutation(indices)

        X_split = np.empty((n, args.context_len, 6), dtype=np.float32)
        for i, idx in enumerate(indices):
            # Find which asset this index belongs to
            asset_idx = np.searchsorted(offsets[1:], idx, side='right')
            local_idx = idx - offsets[asset_idx]
            X_split[i] = all_X[asset_idx][local_idx]

        split_data = {
            'X': X_split,
            'Y': all_Y_cat[indices],
            'Y_relative': Y_relative[indices],
            'asset_type': np.zeros(n, dtype=np.int8),
            'realized_vol': all_rv_cat[indices],
            'dates_end': dates[indices],
            'timestamps_end': all_ts_cat[indices],
            'asset_id': all_aid_cat[indices],
        }

        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, **split_data)
        print(f"  {name}: {n:,} samples, X: {X_split.shape}")
        del X_split  # free memory

    # Metadata
    meta = {
        'asset_id_to_name': {str(k): v for k, v in asset_names.items()},
        'n_assets': len(asset_names),
        'version': 'v7',
        'granularity': '4h',
        'bars_per_day': 6,
        'context_len': args.context_len,
        'max_horizon': args.max_horizon,
        'n_channels': 6,
        'channel_names': [
            'log_return', 'intraday_range', 'body_ratio',
            'log_vol_ratio', 'trailing_vol', 'momentum',
        ],
    }
    meta_path = output_dir / 'asset_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata ({len(asset_names)} assets)")
    print("Done!")


if __name__ == '__main__':
    main()
