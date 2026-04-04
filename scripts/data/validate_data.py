#!/usr/bin/env python
"""
Validate the processed Phantom v3 dataset.

Checks:
  1. Shape consistency
  2. No NaN/Inf values
  3. Per-channel feature statistics
  4. Asset-type distribution
  5. Target distribution per horizon
  6. Cross-channel correlation

Usage:
  python scripts/data/validate_data.py
  python scripts/data/validate_data.py --data_dir data/processed
"""

import argparse
from pathlib import Path

import numpy as np


ASSET_NAMES = {0: 'crypto', 1: 'equity', 2: 'forex', 3: 'commodity'}
CHANNEL_NAMES = ['log_return', 'intraday_range', 'body_ratio',
                 'log_vol_ratio', 'trailing_vol', 'momentum']


def validate_split(name: str, path: Path) -> bool:
    """Validate one split (train/val/test)."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — {path}")
    print(f"{'='*60}")

    d = np.load(path, allow_pickle=True)
    X = d['X']
    H = d['H']
    Y = d['Y']
    asset_type = d['asset_type']
    rv = d['realized_vol']

    issues = []

    # 1. Shape checks
    N = len(X)
    print(f"\nSamples: {N:,}")
    print(f"X shape: {X.shape}")
    print(f"H shape: {H.shape}, Y shape: {Y.shape}")

    if X.ndim != 3 or X.shape[2] != 6:
        issues.append(f"X should be (N, context_len, 6), got {X.shape}")
    if len(H) != N or len(Y) != N or len(asset_type) != N or len(rv) != N:
        issues.append(f"Length mismatch: X={N}, H={len(H)}, Y={len(Y)}, "
                      f"asset={len(asset_type)}, rv={len(rv)}")

    # 2. NaN/Inf checks
    for arr_name, arr in [('X', X), ('Y', Y), ('rv', rv)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0:
            issues.append(f"{arr_name} has {n_nan} NaN values")
        if n_inf > 0:
            issues.append(f"{arr_name} has {n_inf} Inf values")

    # 3. Per-channel statistics
    print(f"\nPer-channel statistics:")
    print(f"  {'Channel':<16} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*56}")
    for ch in range(min(6, X.shape[2])):
        vals = X[:, :, ch].flatten()
        print(f"  {CHANNEL_NAMES[ch]:<16} {vals.mean():>10.4f} {vals.std():>10.4f} "
              f"{vals.min():>10.4f} {vals.max():>10.4f}")

    # 4. Asset-type distribution
    print(f"\nAsset-type distribution:")
    for t in sorted(np.unique(asset_type)):
        count = (asset_type == t).sum()
        name_t = ASSET_NAMES.get(t, f'type_{t}')
        print(f"  {name_t}: {count:,} ({100*count/N:.1f}%)")

    # 5. Horizon distribution
    print(f"\nHorizon distribution:")
    for h in sorted(np.unique(H)):
        count = (H == h).sum()
        print(f"  h={h}: {count:,} ({100*count/N:.1f}%)")

    # 6. Target statistics per horizon
    print(f"\nTarget (Y) statistics per horizon:")
    print(f"  {'Horizon':>8} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Kurt':>10}")
    print(f"  {'-'*48}")
    for h in sorted(np.unique(H)):
        y_h = Y[H == h]
        from scipy.stats import skew, kurtosis
        print(f"  h={h:>5d} {y_h.mean():>10.5f} {y_h.std():>10.4f} "
              f"{skew(y_h):>10.3f} {kurtosis(y_h):>10.3f}")

    # 7. Realized vol stats
    print(f"\nRealized vol: mean={rv.mean():.4f}, std={rv.std():.4f}, "
          f"min={rv.min():.4f}, max={rv.max():.4f}")

    # 8. Cross-channel correlation (on a subsample)
    n_sub = min(50000, N)
    idx = np.random.choice(N, n_sub, replace=False)
    X_sub = X[idx].reshape(n_sub, -1, 6)
    # Average over time dimension
    X_mean = X_sub.mean(axis=1)  # (n_sub, 6)
    corr = np.corrcoef(X_mean.T)
    print(f"\nCross-channel correlation (context-averaged):")
    header = "  " + " " * 16 + "".join(f"{CHANNEL_NAMES[i][:8]:>10}" for i in range(6))
    print(header)
    for i in range(6):
        row = f"  {CHANNEL_NAMES[i]:<16}"
        for j in range(6):
            row += f"{corr[i, j]:>10.3f}"
        print(row)

    # Summary
    if issues:
        print(f"\n!! {len(issues)} ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n  All checks passed.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate Phantom v3 dataset")
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_ok = True

    for split in ['train', 'val', 'test']:
        path = data_dir / f"{split}.npz"
        if path.exists():
            ok = validate_split(split, path)
            all_ok = all_ok and ok
        else:
            print(f"\nWARNING: {path} not found")

    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL SPLITS PASSED VALIDATION")
    else:
        print("  VALIDATION FAILED — see issues above")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
