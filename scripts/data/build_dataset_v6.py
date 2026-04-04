#!/usr/bin/env python
"""
Build crypto-only dataset for Phantom v6.

Pipeline:
  1. Load spot OHLCV + taker buy from data/raw/crypto_v6/
  2. Align funding rates by date (from data/raw/funding/)
  3. Compute 8-channel features (6 OHLCV + taker buy ratio + funding rate)
  4. Create rolling windows (120-day context, 1-30 day curve targets)
  5. Compute cross-sectional relative returns
  6. Split by time (train < 2024, val = 2024, test >= 2025)
  7. Save to data/processed_v6/

Usage:
  python scripts/data/build_dataset_v6.py
  python scripts/data/build_dataset_v6.py --output_dir data/processed_v6
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.features import compute_ohlcv_features_v6, validate_ohlcv


ASSET_TYPE_CRYPTO = 0


def load_spot_data(npz_path: Path) -> dict | None:
    """Load spot OHLCV + taker buy volume."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        dates = d['dates']
        open_ = d['open'].astype(np.float64)
        high = d['high'].astype(np.float64)
        low = d['low'].astype(np.float64)
        close = d['close'].astype(np.float64)
        volume = d['volume'].astype(np.float64)
        taker_buy = d['taker_buy_volume'].astype(np.float64) if 'taker_buy_volume' in d else None

        is_valid, issues = validate_ohlcv(open_, high, low, close, volume)
        if not is_valid:
            return None

        return {
            'dates': dates, 'open': open_, 'high': high,
            'low': low, 'close': close, 'volume': volume,
            'taker_buy_volume': taker_buy,
        }
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return None


def load_auxiliary(npz_path: Path, key: str) -> dict | None:
    """Load auxiliary data (funding or OI) as {date: value} dict."""
    if not npz_path.exists():
        return None
    try:
        d = np.load(npz_path, allow_pickle=True)
        dates = d['dates']
        values = d[key].astype(np.float64)
        return dict(zip(dates, values))
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return None


def align_auxiliary(spot_dates: np.ndarray, aux_dict: dict | None) -> np.ndarray | None:
    """Align auxiliary data to spot dates. Returns None if no data."""
    if aux_dict is None:
        return None
    aligned = np.zeros(len(spot_dates), dtype=np.float64)
    n_matched = 0
    for i, d in enumerate(spot_dates):
        if d in aux_dict:
            aligned[i] = aux_dict[d]
            n_matched += 1
    if n_matched < 30:
        return None
    return aligned


def make_windows_v6(features: np.ndarray, dates: np.ndarray,
                    context_len: int, max_horizon: int = 30) -> dict | None:
    """Create rolling windows with multi-horizon curve targets.

    Args:
        features: (T, 9) float32 array from compute_ohlcv_features_v6
        dates: (T+1,) string array (features are 1 shorter due to diff)
        context_len: context window length
        max_horizon: maximum prediction horizon

    Returns:
        Dict with X: (N, context_len, 9), Y: (N, max_horizon), dates_end: (N,)
    """
    T = len(features)
    if T < context_len + max_horizon:
        return None

    X_list, Y_list, date_list = [], [], []
    log_returns = features[:, 0]  # Channel 0 is log returns

    for start in range(T - context_len - max_horizon + 1):
        ctx = features[start:start + context_len]
        fwd_returns = log_returns[start + context_len:start + context_len + max_horizon]
        curve = np.cumsum(fwd_returns).astype(np.float32)

        X_list.append(ctx)
        Y_list.append(curve)
        # dates[i+1] corresponds to features[i] (offset by 1 from diff)
        date_list.append(dates[start + context_len])

    if not X_list:
        return None

    return {
        'X': np.array(X_list, dtype=np.float32),
        'Y': np.array(Y_list, dtype=np.float32),
        'dates_end': np.array(date_list),
    }


def compute_relative_returns(Y, dates, min_group_size=3):
    """Cross-sectional demeaning per date (single asset class = crypto)."""
    N = len(Y)
    Y_relative = Y.copy()

    unique_dates, date_ids = np.unique(dates, return_inverse=True)
    n_groups = len(unique_dates)

    print(f"  Computing relative returns: {N:,} samples, {n_groups:,} date groups...")
    n_adjusted = 0
    for g in range(n_groups):
        mask = (date_ids == g)
        n = mask.sum()
        if n >= min_group_size:
            class_mean = Y[mask].mean(axis=0)
            Y_relative[mask] -= class_mean
            n_adjusted += n

    print(f"  Adjusted {n_adjusted:,}/{N:,} samples ({100*n_adjusted/N:.1f}%)")
    print(f"  Y_relative stats: mean={Y_relative.mean():.6f}, std={Y_relative.std():.4f}")
    return Y_relative.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build v6 crypto-only dataset")
    parser.add_argument('--spot_dir', type=str, default='data/raw/crypto_v6')
    parser.add_argument('--funding_dir', type=str, default='data/raw/funding')
    parser.add_argument('--output_dir', type=str, default='data/processed_v6')
    parser.add_argument('--context_len', type=int, default=120)
    parser.add_argument('--max_horizon', type=int, default=30)
    parser.add_argument('--val_cutoff', type=str, default='2024-01-01')
    parser.add_argument('--test_cutoff', type=str, default='2025-01-01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    spot_dir = Path(args.spot_dir)
    funding_dir = Path(args.funding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    spot_files = sorted(spot_dir.glob('*.npz'))
    if not spot_files:
        print(f"ERROR: No spot data found in {spot_dir}")
        print("Run: python scripts/data/fetch_crypto_v6.py --source spot")
        return

    print(f"Building v6 dataset (context={args.context_len}, max_horizon={args.max_horizon})")
    print(f"Found {len(spot_files)} spot files")

    all_X, all_Y, all_dates = [], [], []
    all_rv, all_asset_ids = [], []
    asset_names = {}
    stats = {'n_with_funding': 0, 'n_with_oi': 0, 'n_with_taker': 0}

    for file_idx, spot_file in enumerate(spot_files):
        symbol = spot_file.stem  # e.g., "BTC_USDT"
        data = load_spot_data(spot_file)
        if data is None:
            print(f"  {symbol}: invalid data, skipping")
            continue

        # Map symbol to perp name for funding/OI lookup
        perp_name = symbol.replace('_', '')  # BTC_USDT -> BTCUSDT

        # Load and align auxiliary data
        funding_dict = load_auxiliary(funding_dir / f"{perp_name}.npz", 'funding_rate')
        funding_aligned = align_auxiliary(data['dates'], funding_dict)

        has_taker = data['taker_buy_volume'] is not None
        has_funding = funding_aligned is not None

        if has_taker:
            stats['n_with_taker'] += 1
        if has_funding:
            stats['n_with_funding'] += 1

        # Compute 8-channel features
        features = compute_ohlcv_features_v6(
            data['open'], data['high'], data['low'],
            data['close'], data['volume'],
            taker_buy_volume=data['taker_buy_volume'],
            funding_rate=funding_aligned,
        )

        # Create windows
        windows = make_windows_v6(features, data['dates'],
                                  args.context_len, args.max_horizon)
        if windows is None:
            print(f"  {symbol}: too short for windows, skipping")
            continue

        n_windows = len(windows['X'])
        all_X.append(windows['X'])
        all_Y.append(windows['Y'])
        all_dates.append(windows['dates_end'])
        all_rv.append(windows['X'][:, -1, 4])  # Channel 4 = realized vol
        all_asset_ids.append(np.full(n_windows, file_idx, dtype=np.int32))
        asset_names[file_idx] = symbol

        print(f"  {symbol}: {n_windows:,} windows "
              f"(taker={'Y' if has_taker else 'N'}, "
              f"fund={'Y' if has_funding else 'N'})")

    if not all_X:
        print("ERROR: No valid windows created.")
        return

    # Combine
    X = np.concatenate(all_X)
    Y = np.concatenate(all_Y)
    dates = np.concatenate(all_dates)
    realized_vol = np.concatenate(all_rv)
    asset_type = np.zeros(len(X), dtype=np.int8)  # all crypto
    asset_id = np.concatenate(all_asset_ids)

    print(f"\nTotal: {len(X):,} samples from {len(asset_names)} assets")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Feature coverage: taker_buy={stats['n_with_taker']}, "
          f"funding={stats['n_with_funding']}, oi={stats['n_with_oi']}")

    # Check new channel activity
    for ch, name in [(6, 'taker_buy_ratio'), (7, 'funding_rate')]:
        nonzero = (X[:, :, ch] != 0).any(axis=1).mean()
        ch_std = X[:, :, ch].std()
        print(f"  Channel {ch} ({name}): {nonzero*100:.1f}% samples have data, std={ch_std:.4f}")

    # Compute relative returns
    print("\nComputing relative returns...")
    Y_relative = compute_relative_returns(Y, dates)

    # Temporal split
    print(f"\nSplitting: train < {args.val_cutoff} | val < {args.test_cutoff} | test >= {args.test_cutoff}")
    train_mask = dates < args.val_cutoff
    val_mask = (dates >= args.val_cutoff) & (dates < args.test_cutoff)
    test_mask = dates >= args.test_cutoff

    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        n = mask.sum()
        if n == 0:
            print(f"  {name}: 0 samples — skipping")
            continue

        split_data = {
            'X': X[mask],
            'Y': Y[mask],
            'Y_relative': Y_relative[mask],
            'asset_type': asset_type[mask],
            'realized_vol': realized_vol[mask],
            'dates_end': dates[mask],
            'asset_id': asset_id[mask],
        }

        # Shuffle train
        if name == 'train':
            perm = rng.permutation(n)
            split_data = {k: v[perm] for k, v in split_data.items()}

        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, **split_data)
        print(f"  {name}: {n:,} samples, X shape: {split_data['X'].shape}")

    # Save asset metadata
    meta = {
        'asset_id_to_name': {str(k): v for k, v in asset_names.items()},
        'n_assets': len(asset_names),
        'version': 'v6',
        'n_channels': 8,
        'channel_names': [
            'log_return', 'intraday_range', 'body_ratio', 'log_vol_ratio',
            'trailing_vol', 'momentum', 'taker_buy_ratio', 'funding_rate',
        ],
        'feature_coverage': stats,
    }
    meta_path = output_dir / 'asset_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_path} ({len(asset_names)} assets)")
    print("Done!")


if __name__ == '__main__':
    main()
