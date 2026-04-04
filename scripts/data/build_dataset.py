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


def make_windows_v4(features: np.ndarray, dates: np.ndarray,
                    context_len: int, max_horizon: int = 30) -> dict:
    """Create rolling windows with multi-horizon curve targets (v4).

    Each context window produces ONE sample with 30 target values
    (cumulative returns at horizons 1..max_horizon).

    Args:
        features: (T, 6) float32 array from compute_ohlcv_features
        dates: (T+1,) string array
        context_len: context window length
        max_horizon: maximum prediction horizon (default 30)

    Returns:
        Dict with X: (N, context_len, 6), Y: (N, max_horizon), dates_end: (N,)
    """
    T = len(features)
    if T < context_len + max_horizon:
        return None

    X_list, Y_list, date_list = [], [], []
    log_returns = features[:, 0]  # Channel 0 is log returns

    for start in range(T - context_len - max_horizon + 1):
        ctx = features[start:start + context_len]

        # Cumulative returns at horizons 1..max_horizon
        fwd_returns = log_returns[start + context_len:start + context_len + max_horizon]
        curve = np.cumsum(fwd_returns).astype(np.float32)  # (max_horizon,)

        X_list.append(ctx)
        Y_list.append(curve)
        date_list.append(dates[start + context_len])

    if not X_list:
        return None

    return {
        'X': np.array(X_list, dtype=np.float32),
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

    # Check for yfinance (or legacy stooq) subdirs
    for parent_name in ['yfinance', 'stooq']:
        yf_dir = raw_dir / parent_name
        if yf_dir.exists():
            for subdir in sorted(yf_dir.iterdir()):
                if subdir.is_dir():
                    cls_name = subdir.name
                    for f in sorted(subdir.glob('*.npz')):
                        assets_by_class[cls_name].append(f)

    return dict(assets_by_class)


def process_asset_class(asset_class: str, files: list[Path],
                        context_len: int, horizons: list,
                        version: str = 'v3', max_horizon: int = 30,
                        asset_id_offset: int = 0) -> dict | None:
    """Process all assets in one class, return combined windows."""
    asset_type = DIR_TO_ASSET_TYPE.get(asset_class, ASSET_TYPE_EQUITY)
    all_X, all_Y, all_dates, all_rv = [], [], [], []
    all_H = []       # only used for v3
    all_asset_ids = []  # v5: per-sample asset identifier
    asset_names = {}    # v5: id → name mapping

    for file_idx, f in enumerate(files):
        data = load_raw_asset(f)
        if data is None:
            continue

        features = compute_ohlcv_features(
            data['open'], data['high'], data['low'],
            data['close'], data['volume'],
        )

        if version in ('v4', 'v5'):
            windows = make_windows_v4(features, data['dates'], context_len, max_horizon)
        else:
            windows = make_windows(features, data['dates'], context_len, horizons)

        if windows is None:
            continue

        n_windows = len(windows['X'])
        all_X.append(windows['X'])
        all_Y.append(windows['Y'])
        all_dates.append(windows['dates_end'])
        rv = windows['X'][:, -1, 4]
        all_rv.append(rv)
        if 'H' in windows:
            all_H.append(windows['H'])

        # Track asset ID (v5)
        aid = asset_id_offset + file_idx
        all_asset_ids.append(np.full(n_windows, aid, dtype=np.int32))
        asset_names[aid] = f.stem

    if not all_X:
        return None

    X_cat = np.concatenate(all_X)
    result = {
        'X': X_cat,
        'Y': np.concatenate(all_Y),
        'dates_end': np.concatenate(all_dates),
        'asset_type': np.full(len(X_cat), asset_type, dtype=np.int8),
        'realized_vol': np.concatenate(all_rv),
        'asset_id': np.concatenate(all_asset_ids),
    }
    if all_H:
        result['H'] = np.concatenate(all_H)

    print(f"  {asset_class}: {len(files)} assets -> {len(X_cat):,} windows")
    return result, asset_names


def compute_relative_returns(combined: dict, min_group_size: int = 3) -> np.ndarray:
    """Compute cross-sectional relative returns per date per asset class.

    Y_relative[i] = Y[i] - mean(Y[same date, same class])

    Args:
        combined: dict with 'Y', 'dates_end', 'asset_type' arrays.
        min_group_size: minimum assets in a group to compute relative returns.

    Returns:
        Y_relative: (N, H) float32 array of relative cumulative returns.
    """
    Y = combined['Y']
    dates = combined['dates_end']
    asset_types = combined['asset_type']
    N = len(Y)

    Y_relative = Y.copy()

    # Build composite group key: "date_assettype"
    # Use integer encoding for speed
    unique_dates, date_ids = np.unique(dates, return_inverse=True)
    n_classes = int(asset_types.max()) + 1
    group_ids = date_ids * n_classes + asset_types.astype(np.int32)

    unique_groups, group_inv = np.unique(group_ids, return_inverse=True)
    n_groups = len(unique_groups)

    print(f"  Computing relative returns: {N:,} samples, {n_groups:,} (date, class) groups...")

    n_adjusted = 0
    for g in range(n_groups):
        mask = (group_inv == g)
        n = mask.sum()
        if n >= min_group_size:
            class_mean = Y[mask].mean(axis=0)  # (H,)
            Y_relative[mask] -= class_mean
            n_adjusted += n

    print(f"  Adjusted {n_adjusted:,}/{N:,} samples ({100*n_adjusted/N:.1f}%)")
    print(f"  Y_relative stats: mean={Y_relative.mean():.6f}, std={Y_relative.std():.4f}")
    return Y_relative.astype(np.float32)


def subsample(data: dict, budget: int, rng: np.random.Generator) -> dict:
    """Randomly subsample to budget if needed."""
    n = len(data['X'])
    if n <= budget:
        return data
    idx = rng.choice(n, budget, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in data.items()}


def temporal_split(data: dict, val_cutoff: str, test_cutoff: str,
                   keep_dates: bool = False) -> dict:
    """Split data by date into train/val/test."""
    dates = data['dates_end']
    train_mask = dates < val_cutoff
    val_mask = (dates >= val_cutoff) & (dates < test_cutoff)
    test_mask = dates >= test_cutoff

    splits = {}
    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        if keep_dates:
            splits[name] = {k: v[mask] for k, v in data.items()}
        else:
            splits[name] = {k: v[mask] for k, v in data.items() if k != 'dates_end'}
        print(f"    {name}: {mask.sum():,} samples")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Build Phantom dataset")
    parser.add_argument('--raw_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--version', type=str, default='v3', choices=['v3', 'v4', 'v5'],
                        help='v3: discrete horizons, v4: curve targets, v5: relative returns')
    parser.add_argument('--context_len', type=int, default=None,
                        help='Context window length (default: 75 for v3, 120 for v4)')
    parser.add_argument('--max_horizon', type=int, default=30,
                        help='Max prediction horizon for v4 (default 30)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 5, 7],
                        help='Discrete horizons for v3 (default [3, 5, 7])')
    parser.add_argument('--val_cutoff', type=str, default='2024-01-01')
    parser.add_argument('--test_cutoff', type=str, default='2025-01-01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Defaults based on version
    if args.context_len is None:
        args.context_len = 120 if args.version == 'v4' else 75

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Building {args.version} dataset (context={args.context_len})")
    print(f"Scanning {raw_dir} for raw OHLCV data...")
    assets_by_class = scan_raw_dir(raw_dir)

    if not assets_by_class:
        print("ERROR: No data found. Run fetch_crypto.py and/or fetch_yfinance.py first.")
        return

    for cls, files in assets_by_class.items():
        print(f"  {cls}: {len(files)} files")

    # Process each asset class
    all_data = {}
    all_asset_names = {}
    asset_id_offset = 0
    for cls, files in assets_by_class.items():
        print(f"\nProcessing {cls}...")
        result = process_asset_class(cls, files, args.context_len, args.horizons,
                                     version=args.version, max_horizon=args.max_horizon,
                                     asset_id_offset=asset_id_offset)
        if result is not None:
            data, asset_names = result
            budget = SAMPLE_BUDGET.get(cls, 200_000)
            data = subsample(data, budget, rng)
            all_data[cls] = data
            all_asset_names.update(asset_names)
            asset_id_offset = max(asset_names.keys()) + 1 if asset_names else asset_id_offset

    if not all_data:
        print("ERROR: No valid windows created.")
        return

    # Combine all asset classes
    print("\nCombining all asset classes...")
    # Determine which keys are present
    all_keys = set()
    for d in all_data.values():
        all_keys.update(d.keys())
    combined = {}
    for key in all_keys:
        arrays = [d[key] for d in all_data.values() if key in d]
        if arrays:
            combined[key] = np.concatenate(arrays)

    total = len(combined['X'])
    print(f"Total: {total:,} samples")
    print(f"X shape: {combined['X'].shape}, Y shape: {combined['Y'].shape}")

    # Asset type distribution
    for t, name in [(0, 'crypto'), (1, 'equity'), (2, 'forex'), (3, 'commodity')]:
        count = (combined['asset_type'] == t).sum()
        if count > 0:
            print(f"  {name}: {count:,} ({100*count/total:.1f}%)")

    # v5: compute relative returns before splitting
    if args.version == 'v5':
        print("\nComputing relative returns...")
        combined['Y_relative'] = compute_relative_returns(combined)

    # Temporal split (keep dates for v5 eval)
    keep_dates = args.version == 'v5'
    print(f"\nSplitting: train < {args.val_cutoff} | val < {args.test_cutoff} | test >= {args.test_cutoff}")
    splits = temporal_split(combined, args.val_cutoff, args.test_cutoff,
                            keep_dates=keep_dates)

    # Shuffle train set
    n_train = len(splits['train']['X'])
    perm = rng.permutation(n_train)
    for key in splits['train']:
        splits['train'][key] = splits['train'][key][perm]

    # Save
    for name, data in splits.items():
        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, **data)
        y_key = 'Y_relative' if 'Y_relative' in data else 'Y'
        print(f"Saved {out_path} ({len(data['X']):,} samples, "
              f"X shape: {data['X'].shape}, {y_key} shape: {data[y_key].shape})")

    # v5: save asset metadata
    if args.version == 'v5' and all_asset_names:
        import json
        meta = {
            'asset_id_to_name': {str(k): v for k, v in all_asset_names.items()},
            'n_assets': len(all_asset_names),
            'version': args.version,
        }
        meta_path = output_dir / 'asset_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {meta_path} ({len(all_asset_names)} assets)")

    print("\nDone!")


if __name__ == '__main__':
    main()
