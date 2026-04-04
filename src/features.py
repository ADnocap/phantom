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
