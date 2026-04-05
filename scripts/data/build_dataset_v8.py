#!/usr/bin/env python
"""
Build crypto-only daily dataset for Phantom v8.

Same as v5 pipeline but crypto-only with expanded asset list.
6 OHLCV channels, 120-day context, 30-day horizons, relative returns.

Usage:
  python scripts/data/build_dataset_v8.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.features import compute_ohlcv_features, validate_ohlcv


def load_raw(npz_path):
    try:
        d = np.load(npz_path, allow_pickle=True)
        o, h, l, c, v = (d[k].astype(np.float64) for k in ['open', 'high', 'low', 'close', 'volume'])
        is_valid, _ = validate_ohlcv(o, h, l, c, v)
        if not is_valid:
            return None
        return {'dates': d['dates'], 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
    except Exception as e:
        print(f"  Error: {npz_path.name}: {e}")
        return None


def make_windows(features, dates, context_len=120, max_horizon=30):
    T = len(features)
    if T < context_len + max_horizon:
        return None
    X, Y, D = [], [], []
    lr = features[:, 0]
    for s in range(T - context_len - max_horizon + 1):
        X.append(features[s:s + context_len])
        Y.append(np.cumsum(lr[s + context_len:s + context_len + max_horizon]).astype(np.float32))
        D.append(dates[s + context_len])
    return {
        'X': np.array(X, dtype=np.float32),
        'Y': np.array(Y, dtype=np.float32),
        'dates_end': np.array(D),
    }


def compute_relative_returns(Y, dates, min_group=3):
    Y_rel = Y.copy()
    unique_dates, ids = np.unique(dates, return_inverse=True)
    n_adj = 0
    for g in range(len(unique_dates)):
        mask = ids == g
        if mask.sum() >= min_group:
            Y_rel[mask] -= Y[mask].mean(axis=0)
            n_adj += mask.sum()
    print(f"  Adjusted {n_adj:,}/{len(Y):,} ({100*n_adj/len(Y):.1f}%)")
    return Y_rel.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='data/raw/crypto_v8')
    parser.add_argument('--output_dir', default='data/processed_v8')
    parser.add_argument('--context_len', type=int, default=120)
    parser.add_argument('--max_horizon', type=int, default=30)
    parser.add_argument('--val_cutoff', default='2024-01-01')
    parser.add_argument('--test_cutoff', default='2025-01-01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(raw_dir.glob('*.npz'))
    if not files:
        print(f"No data in {raw_dir}. Run fetch_crypto_v8.py first.")
        return

    print(f"Building v8 dataset: {len(files)} files, context={args.context_len}, horizon={args.max_horizon}")

    all_X, all_Y, all_dates, all_rv, all_aid = [], [], [], [], []
    asset_names = {}

    for i, f in enumerate(files):
        data = load_raw(f)
        if data is None:
            continue
        features = compute_ohlcv_features(data['open'], data['high'], data['low'],
                                           data['close'], data['volume'])
        windows = make_windows(features, data['dates'], args.context_len, args.max_horizon)
        if windows is None:
            continue

        n = len(windows['X'])
        all_X.append(windows['X'])
        all_Y.append(windows['Y'])
        all_dates.append(windows['dates_end'])
        all_rv.append(windows['X'][:, -1, 4])
        all_aid.append(np.full(n, i, dtype=np.int32))
        asset_names[i] = f.stem
        print(f"  {f.stem}: {n:,} windows ({len(data['dates'])} days)")

    if not all_X:
        print("No valid windows.")
        return

    Y = np.concatenate(all_Y)
    dates = np.concatenate(all_dates)
    rv = np.concatenate(all_rv)
    aid = np.concatenate(all_aid)
    # Track offsets for memory-efficient X assembly
    offsets = np.cumsum([0] + [len(x) for x in all_X])
    total = len(Y)

    print(f"\nTotal: {total:,} samples from {len(asset_names)} assets")

    print("Computing relative returns...")
    Y_rel = compute_relative_returns(Y, dates)

    print(f"\nSplitting: train < {args.val_cutoff} | val < {args.test_cutoff} | test >= {args.test_cutoff}")
    train_mask = dates < args.val_cutoff
    val_mask = (dates >= args.val_cutoff) & (dates < args.test_cutoff)
    test_mask = dates >= args.test_cutoff

    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        n = mask.sum()
        if n == 0:
            print(f"  {name}: 0 samples")
            continue

        indices = np.where(mask)[0]
        if name == 'train':
            indices = rng.permutation(indices)

        X_split = np.empty((n, args.context_len, 6), dtype=np.float32)
        for j, idx in enumerate(indices):
            a = np.searchsorted(offsets[1:], idx, side='right')
            X_split[j] = all_X[a][idx - offsets[a]]

        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, X=X_split, Y=Y[indices], Y_relative=Y_rel[indices],
                            asset_type=np.zeros(n, dtype=np.int8),
                            realized_vol=rv[indices], dates_end=dates[indices],
                            asset_id=aid[indices])
        print(f"  {name}: {n:,} samples, X: {X_split.shape}")
        del X_split

    meta = {
        'asset_id_to_name': {str(k): v for k, v in asset_names.items()},
        'n_assets': len(asset_names),
        'version': 'v8',
        'n_channels': 6,
    }
    with open(output_dir / 'asset_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone! {len(asset_names)} assets")


if __name__ == '__main__':
    main()
