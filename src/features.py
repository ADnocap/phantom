"""
Universal OHLCV feature computation for Phantom v3.

Computes 6-channel features from Open/High/Low/Close/Volume data that are
meaningful for ALL asset types (crypto, equities, forex, commodities).

Channels:
  0: Close-to-close log return
  1: Intraday range (Parkinson-like volatility)
  2: Body ratio (bullish/bearish candle strength, in [-1, 1])
  3: Log volume ratio (relative to 30-day median; 0 for forex)
  4: Trailing realized vol (30-day annualized)
  5: Trailing momentum (10-day cumulative return)
"""

import warnings
import numpy as np
import pandas as pd


def compute_ohlcv_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vol_window: int = 30,
    mom_window: int = 10,
) -> np.ndarray:
    """Compute 6-channel features from OHLCV data.

    Args:
        open_:  (N,) daily open prices.
        high:   (N,) daily high prices.
        low:    (N,) daily low prices.
        close:  (N,) daily close prices.
        volume: (N,) daily volume (0 for forex/missing).
        vol_window: Window for trailing realized vol (default 30).
        mom_window: Window for trailing momentum (default 10).

    Returns:
        features: (N-1, 6) float32 array. Index i corresponds to day i+1
                  in the original series (since ch0 needs close_{t-1}).
    """
    n = len(close)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")

    # Channel 0: Close-to-close log return
    log_returns = np.diff(np.log(close))  # (N-1,)

    # Align all other channels to [1:] (same length as log_returns)
    h = high[1:]
    l = low[1:]
    c = close[1:]
    o = open_[1:]
    v = volume[1:]

    # Channel 1: Intraday range (Parkinson-like)
    intraday_range = (h - l) / (c + 1e-10)

    # Channel 2: Body ratio — bullish/bearish candle strength
    candle_range = h - l + 1e-10
    body_ratio = (c - o) / candle_range
    body_ratio = np.clip(body_ratio, -1.0, 1.0)

    # Channel 3: Log volume ratio (relative to rolling 30-day median)
    vol_series = pd.Series(v)
    vol_median = vol_series.rolling(30, min_periods=1).median().values
    # Avoid log(0): set to 0 where volume is 0 (forex, missing)
    log_vol_ratio = np.zeros_like(v, dtype=np.float64)
    valid_vol = (v > 0) & (vol_median > 0)
    log_vol_ratio[valid_vol] = np.log(v[valid_vol] / vol_median[valid_vol])

    # Channel 4: Trailing realized vol (annualized)
    ret_series = pd.Series(log_returns)
    trailing_vol = ret_series.rolling(vol_window, min_periods=5).std().values * np.sqrt(252)
    # Fill early NaN with expanding std
    expanding_vol = ret_series.expanding(min_periods=2).std().values * np.sqrt(252)
    nan_mask = np.isnan(trailing_vol)
    trailing_vol[nan_mask] = expanding_vol[nan_mask]
    # If still NaN (e.g., first value), fill with 0
    trailing_vol = np.nan_to_num(trailing_vol, nan=0.0)

    # Channel 5: Trailing momentum (cumulative return over window)
    momentum = ret_series.rolling(mom_window, min_periods=1).sum().values

    features = np.column_stack([
        log_returns,      # 0: returns
        intraday_range,   # 1: intraday vol
        body_ratio,       # 2: candle body
        log_vol_ratio,    # 3: relative volume
        trailing_vol,     # 4: realized vol
        momentum,         # 5: momentum
    ])

    # Final safety: replace any remaining NaN/Inf
    bad = ~np.isfinite(features)
    if bad.any():
        n_bad = bad.sum()
        warnings.warn(f"Replaced {n_bad} non-finite values in features")
        features[bad] = 0.0

    return features.astype(np.float32)


def compute_ohlcv_features_4h(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vol_window: int = 180,
    mom_window: int = 60,
) -> np.ndarray:
    """Compute 6-channel features from 4-hour OHLCV bars.

    Same channels as compute_ohlcv_features() but with windows adjusted
    for 4h bars (6 bars per day):
      - Volume median: 180 bars (30 days)
      - Trailing vol: 180 bars (30 days), annualized by sqrt(252*6)=sqrt(1512)
      - Momentum: 60 bars (10 days)

    Returns:
        features: (N-1, 6) float32 array.
    """
    n = len(close)
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")

    bars_per_year = 252 * 6  # 1512

    log_returns = np.diff(np.log(close))
    h = high[1:]
    l = low[1:]
    c = close[1:]
    o = open_[1:]
    v = volume[1:]

    intraday_range = (h - l) / (c + 1e-10)

    candle_range = h - l + 1e-10
    body_ratio = (c - o) / candle_range
    body_ratio = np.clip(body_ratio, -1.0, 1.0)

    vol_series = pd.Series(v)
    vol_median = vol_series.rolling(vol_window, min_periods=1).median().values
    log_vol_ratio = np.zeros_like(v, dtype=np.float64)
    valid_vol = (v > 0) & (vol_median > 0)
    log_vol_ratio[valid_vol] = np.log(v[valid_vol] / vol_median[valid_vol])

    ret_series = pd.Series(log_returns)
    trailing_vol = ret_series.rolling(vol_window, min_periods=30).std().values * np.sqrt(bars_per_year)
    expanding_vol = ret_series.expanding(min_periods=10).std().values * np.sqrt(bars_per_year)
    nan_mask = np.isnan(trailing_vol)
    trailing_vol[nan_mask] = expanding_vol[nan_mask]
    trailing_vol = np.nan_to_num(trailing_vol, nan=0.0)

    momentum = ret_series.rolling(mom_window, min_periods=1).sum().values

    features = np.column_stack([
        log_returns, intraday_range, body_ratio,
        log_vol_ratio, trailing_vol, momentum,
    ])

    bad = ~np.isfinite(features)
    if bad.any():
        warnings.warn(f"Replaced {bad.sum()} non-finite values in 4h features")
        features[bad] = 0.0

    return features.astype(np.float32)


def compute_ohlcv_features_v6(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    taker_buy_volume: np.ndarray | None = None,
    funding_rate: np.ndarray | None = None,
    vol_window: int = 30,
    mom_window: int = 10,
) -> np.ndarray:
    """Compute 8-channel features for v6 (6 OHLCV + 2 crypto-specific).

    Channels 0-5: identical to compute_ohlcv_features()
    Channel 6: Taker buy ratio (centered at 0) — order flow direction
    Channel 7: Funding rate (scaled) — leveraged positioning sentiment

    Note: OI was dropped — Binance only provides 30 days of OI history.

    Args:
        open_, high, low, close, volume: (N,) daily OHLCV arrays.
        taker_buy_volume: (N,) taker buy base volume, or None (zero-fill).
        funding_rate: (N,) daily average funding rate, or None (zero-fill).
        vol_window: Window for trailing realized vol (default 30).
        mom_window: Window for trailing momentum (default 10).

    Returns:
        features: (N-1, 8) float32 array.
    """
    # Base 6-channel features
    base = compute_ohlcv_features(open_, high, low, close, volume,
                                  vol_window, mom_window)
    n = len(base)  # N-1

    # Channel 6: Taker buy ratio, centered at 0 (range ~ [-0.5, 0.5])
    if taker_buy_volume is not None:
        tbv = taker_buy_volume[1:]  # align with base (skip first)
        vol = volume[1:]
        taker_ratio = tbv / (vol + 1e-10) - 0.5
    else:
        taker_ratio = np.zeros(n, dtype=np.float64)

    # Channel 7: Funding rate (scaled by 100 so magnitude ~ 0.01)
    if funding_rate is not None:
        fr = funding_rate[1:] * 100.0  # align with base
    else:
        fr = np.zeros(n, dtype=np.float64)

    features = np.column_stack([base, taker_ratio, fr])

    # Safety: replace NaN/Inf
    bad = ~np.isfinite(features)
    if bad.any():
        n_bad = bad.sum()
        warnings.warn(f"Replaced {n_bad} non-finite values in v6 features")
        features[bad] = 0.0

    return features.astype(np.float32)  # (N-1, 8)


def validate_ohlcv(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[bool, list[str]]:
    """Validate OHLCV data integrity.

    Returns:
        (is_valid, issues): True if data is usable, list of issues found.
    """
    issues = []
    n = len(close)

    # Basic checks
    if n < 50:
        issues.append(f"Too few data points: {n} (need >= 50)")

    for name, arr in [('open', open_), ('high', high), ('low', low),
                      ('close', close), ('volume', volume)]:
        if len(arr) != n:
            issues.append(f"{name} length {len(arr)} != close length {n}")
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            issues.append(f"{name} has {nan_count} NaN values")
        if name != 'volume':
            neg_count = (arr <= 0).sum()
            if neg_count > 0:
                issues.append(f"{name} has {neg_count} non-positive values")

    # High >= Low
    violations = (high < low - 1e-10).sum()
    if violations > 0:
        issues.append(f"high < low in {violations} rows")

    # Extreme returns (>100% in a day for non-crypto is suspicious)
    if n > 1:
        log_ret = np.abs(np.diff(np.log(close + 1e-10)))
        extreme = (log_ret > 2.0).sum()  # > 600% move
        if extreme > 0:
            issues.append(f"{extreme} extreme daily moves (>600%)")

    is_valid = len(issues) == 0 or all('Too few' not in i for i in issues)
    return is_valid, issues
