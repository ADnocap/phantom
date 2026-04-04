"""
Real BTC price data fetching and preprocessing.

Stitches Bitstamp (2015-2017) + Binance (2017-present) for maximum
history with high-quality data from the most liquid exchanges.
"""

import time
from pathlib import Path
from datetime import datetime

import numpy as np


def _load_ccxt():
    try:
        import ccxt
        return ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")


def _fetch_all_ohlcv(exchange, symbol, timeframe, since_ms, until_ms=None):
    """Fetch all OHLCV data by paginating through the API."""
    all_ohlcv = []
    limit = 1000
    current = since_ms

    if until_ms is None:
        until_ms = int(datetime.now().timestamp() * 1000)

    while current < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=limit)
        except Exception as e:
            print(f"  Retry after error: {e}")
            time.sleep(2)
            continue

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]

        if last_ts <= current:
            break
        current = last_ts + 1

        # Rate limiting
        time.sleep(exchange.rateLimit / 1000)

    return all_ohlcv


def fetch_btc_daily(
    cache_path: str = "data/btc_daily.npz",
    start_date: str = "2015-01-01",
    force_refresh: bool = False,
) -> dict:
    """Fetch daily BTC data, stitching Bitstamp + Binance.

    Returns dict with:
        dates:       (N,) array of date strings 'YYYY-MM-DD'
        opens:       (N,) float64 daily open prices
        highs:       (N,) float64 daily high prices
        lows:        (N,) float64 daily low prices
        closes:      (N,) float64 daily close prices
        volumes:     (N,) float64 daily volume
        log_returns: (N-1,) float32 daily log-returns
    """
    cache = Path(cache_path)
    if cache.exists() and not force_refresh:
        d = np.load(cache, allow_pickle=True)
        print(f"Loaded cached BTC data: {len(d['dates'])} days from {d['dates'][0]} to {d['dates'][-1]}")
        return dict(d)

    cache.parent.mkdir(parents=True, exist_ok=True)

    # ── Bitstamp: 2015 to 2017-08-16 ──
    ccxt = _load_ccxt()
    print("Fetching Bitstamp BTC/USD (2015 to 2017-08-16)...")
    bitstamp = ccxt.bitstamp()
    since_bs = bitstamp.parse8601(f"{start_date}T00:00:00Z")
    until_bs = bitstamp.parse8601("2017-08-17T00:00:00Z")
    ohlcv_bs = _fetch_all_ohlcv(bitstamp, "BTC/USD", "1d", since_bs, until_bs)
    print(f"  Got {len(ohlcv_bs)} candles from Bitstamp")

    # ── Binance: 2017-08-17 to present ──
    print("Fetching Binance BTC/USDT (2017-08-17 to present)...")
    binance = ccxt.binance()
    since_bn = binance.parse8601("2017-08-17T00:00:00Z")
    ohlcv_bn = _fetch_all_ohlcv(binance, "BTC/USDT", "1d", since_bn)
    print(f"  Got {len(ohlcv_bn)} candles from Binance")

    # ── Stitch together ──
    # Remove any Bitstamp candles on or after Binance start date
    ohlcv_bs = [c for c in ohlcv_bs if c[0] < since_bn]
    combined = ohlcv_bs + ohlcv_bn

    # Sort by timestamp and deduplicate
    combined.sort(key=lambda c: c[0])
    seen = set()
    deduped = []
    for c in combined:
        date_str = datetime.utcfromtimestamp(c[0] / 1000).strftime("%Y-%m-%d")
        if date_str not in seen:
            seen.add(date_str)
            # (date, open, high, low, close, volume)
            deduped.append((date_str, c[1], c[2], c[3], c[4], c[5]))

    dates = np.array([d[0] for d in deduped])
    opens = np.array([d[1] for d in deduped], dtype=np.float64)
    highs = np.array([d[2] for d in deduped], dtype=np.float64)
    lows = np.array([d[3] for d in deduped], dtype=np.float64)
    closes = np.array([d[4] for d in deduped], dtype=np.float64)
    volumes = np.array([d[5] for d in deduped], dtype=np.float64)

    # Compute log-returns
    log_returns = np.diff(np.log(closes)).astype(np.float32)

    print(f"Stitched: {len(dates)} days from {dates[0]} to {dates[-1]}")
    print(f"Log-returns: {len(log_returns)} values, mean={log_returns.mean():.6f}, std={log_returns.std():.4f}")

    # Cache
    np.savez(cache, dates=dates, opens=opens, highs=highs, lows=lows,
             closes=closes, volumes=volumes, log_returns=log_returns)
    print(f"Cached to {cache}")

    return {'dates': dates, 'opens': opens, 'highs': highs, 'lows': lows,
            'closes': closes, 'volumes': volumes, 'log_returns': log_returns}


def make_rolling_windows(
    log_returns: np.ndarray,
    context_len: int = 75,
    horizons: list = [3, 5, 7],
    n_input_channels: int = 1,
    ohlcv: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create rolling window samples from real log-return series.

    Args:
        log_returns: Full log-return series.
        context_len: Number of days per context window.
        horizons: List of forecast horizons.
        n_input_channels: 1 = returns only, 4 = returns + 3 vol features,
                          6 = OHLCV features (requires ohlcv dict).
        ohlcv: Dict with opens, highs, lows, closes, volumes arrays.
               Required when n_input_channels == 6.

    Returns:
        X: (N, context_len) or (N, context_len, C) float32 — context windows
        H: (N,) int8 — horizons
        Y: (N,) float32 — cumulative forward log-returns
    """
    X_list, H_list, Y_list = [], [], []
    max_h = max(horizons)
    n = len(log_returns)

    if n_input_channels == 6 and ohlcv is not None:
        # Compute 6-channel OHLCV features for the entire series
        from .features import compute_ohlcv_features
        features = compute_ohlcv_features(
            ohlcv['opens'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['closes'], ohlcv['volumes'],
        )  # (N-1, 6) — same length as log_returns

        for start in range(n - context_len - max_h + 1):
            ctx = features[start:start + context_len]  # (context_len, 6)
            for h in horizons:
                fwd = log_returns[start + context_len:start + context_len + h].sum()
                X_list.append(ctx)
                H_list.append(h)
                Y_list.append(fwd)
    elif n_input_channels > 1 and n_input_channels != 6:
        from .data import compute_vol_features
        for start in range(n - context_len - max_h + 1):
            ctx = log_returns[start:start + context_len]
            for h in horizons:
                fwd = log_returns[start + context_len:start + context_len + h].sum()
                vol_feats = compute_vol_features(ctx)  # (L, 3)
                x = np.concatenate([ctx.astype(np.float32).reshape(-1, 1), vol_feats], axis=1)
                X_list.append(x)
                H_list.append(h)
                Y_list.append(fwd)
    else:
        for start in range(n - context_len - max_h + 1):
            ctx = log_returns[start:start + context_len]
            for h in horizons:
                fwd = log_returns[start + context_len:start + context_len + h].sum()
                X_list.append(ctx)
                H_list.append(h)
                Y_list.append(fwd)

    X = np.array(X_list, dtype=np.float32)
    H = np.array(H_list, dtype=np.int8)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"Rolling windows: {len(X)} samples (context={context_len}, horizons={horizons}, channels={n_input_channels})")
    return X, H, Y


def temporal_split(
    dates: np.ndarray,
    log_returns: np.ndarray,
    context_len: int = 75,
    horizons: list = [3, 5, 7],
    val_start: str = "2022-01-01",
    test_start: str = "2023-07-01",
    n_input_channels: int = 1,
    ohlcv: dict | None = None,
) -> dict:
    """Split real data into train/val/test by date.

    Returns dict with 'train', 'val', 'test' keys, each containing (X, H, Y).
    """
    val_idx = np.searchsorted(dates, val_start)
    test_idx = np.searchsorted(dates, test_start)

    splits = {}
    for name, start, end in [
        ('train', 0, val_idx - 1),
        ('val', val_idx - context_len, test_idx - 1),
        ('test', test_idx - context_len, len(log_returns)),
    ]:
        s = max(start, 0)
        lr_slice = log_returns[s:end]
        # Slice OHLCV data to match (OHLCV is 1 longer than log_returns)
        ohlcv_slice = None
        if ohlcv is not None and n_input_channels == 6:
            ohlcv_slice = {
                'opens': ohlcv['opens'][s:end + 1],
                'highs': ohlcv['highs'][s:end + 1],
                'lows': ohlcv['lows'][s:end + 1],
                'closes': ohlcv['closes'][s:end + 1],
                'volumes': ohlcv['volumes'][s:end + 1],
            }
        X, H, Y = make_rolling_windows(lr_slice, context_len, horizons,
                                        n_input_channels=n_input_channels,
                                        ohlcv=ohlcv_slice)
        splits[name] = (X, H, Y)
        print(f"  {name}: {len(X)} samples")

    return splits
