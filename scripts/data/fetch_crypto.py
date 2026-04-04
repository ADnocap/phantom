#!/usr/bin/env python
"""
Fetch crypto OHLCV data from Binance (active coins) and CryptoCompare (broader coverage).

Saves one .npz per symbol to data/raw/crypto/ with keys:
  dates, open, high, low, close, volume

Usage:
  python scripts/data/fetch_crypto.py
  python scripts/data/fetch_crypto.py --source binance
  python scripts/data/fetch_crypto.py --source cryptocompare
  python scripts/data/fetch_crypto.py --output_dir data/raw/crypto
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Symbol lists ─────────────────────────────────────────────────────

BINANCE_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
    'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'FIL/USDT',
    'NEAR/USDT', 'APT/USDT', 'OP/USDT', 'ARB/USDT', 'AAVE/USDT',
]

# Broader set for CryptoCompare (includes dead/delisted coins)
CRYPTOCOMPARE_SYMBOLS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LINK',
    'MATIC', 'UNI', 'ATOM', 'LTC', 'FIL', 'NEAR', 'APT', 'OP', 'ARB', 'AAVE',
    # Historical / less active
    'EOS', 'XLM', 'TRX', 'ETC', 'XMR', 'DASH', 'ZEC', 'NEO', 'IOTA', 'XTZ',
    'ALGO', 'VET', 'THETA', 'FTM', 'SAND', 'MANA', 'AXS', 'ENJ', 'CRV', 'COMP',
    'SUSHI', 'YFI', 'SNX', 'MKR', 'BAT', 'ZRX', 'ICX', 'ONT', 'WAVES', 'QTUM',
    # Dead / collapsed (good for diversity)
    'LUNA', 'FTT', 'CEL', 'UST', 'HEX',
    # Newer / mid-cap
    'SUI', 'SEI', 'TIA', 'INJ', 'STX', 'RNDR', 'FET', 'PEPE', 'WLD', 'JUP',
]

MIN_DAYS = 200  # Skip symbols with less than this many days of data


def _fetch_all_ohlcv(exchange, symbol, timeframe, since_ms, until_ms=None):
    """Paginated OHLCV fetch (reuses pattern from btc_data.py)."""
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
        time.sleep(exchange.rateLimit / 1000)

    return all_ohlcv


def fetch_binance(output_dir: Path, start_date: str = '2017-01-01'):
    """Fetch daily OHLCV for active coins from Binance via ccxt."""
    import ccxt

    output_dir.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.binance()
    since_ms = exchange.parse8601(f"{start_date}T00:00:00Z")

    for symbol in BINANCE_SYMBOLS:
        safe_name = symbol.replace('/', '_')
        out_path = output_dir / f"{safe_name}.npz"
        if out_path.exists():
            print(f"  {symbol}: already exists, skipping")
            continue

        print(f"  Fetching {symbol} from Binance...")
        try:
            ohlcv = _fetch_all_ohlcv(exchange, symbol, '1d', since_ms)
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")
            continue

        if len(ohlcv) < MIN_DAYS:
            print(f"  {symbol}: only {len(ohlcv)} days, skipping (need {MIN_DAYS})")
            continue

        _save_ohlcv(ohlcv, out_path, symbol)


def fetch_cryptocompare(output_dir: Path):
    """Fetch daily OHLCV from CryptoCompare free API."""
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'

    for fsym in CRYPTOCOMPARE_SYMBOLS:
        out_path = output_dir / f"{fsym}_USD.npz"
        if out_path.exists():
            print(f"  {fsym}: already exists, skipping")
            continue

        print(f"  Fetching {fsym}/USD from CryptoCompare...")
        all_data = []

        # CryptoCompare returns max 2000 per call, paginate backward
        to_ts = None
        while True:
            params = {'fsym': fsym, 'tsym': 'USD', 'limit': 2000}
            if to_ts is not None:
                params['toTs'] = to_ts

            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  ERROR fetching {fsym}: {e}")
                break

            if data.get('Response') != 'Success' or 'Data' not in data:
                print(f"  {fsym}: API error: {data.get('Message', 'unknown')}")
                break

            rows = data['Data']['Data']
            if not rows:
                break

            # Filter out rows with zero prices (before listing)
            valid_rows = [r for r in rows if r.get('close', 0) > 0]
            all_data = valid_rows + all_data  # prepend older data

            # Check if we've reached the beginning
            if len(rows) < 2000 or data['Data'].get('TimeFrom', 0) <= 0:
                break

            # Next page: go further back
            to_ts = rows[0]['time'] - 1
            time.sleep(1.0)  # Rate limiting

        if len(all_data) < MIN_DAYS:
            print(f"  {fsym}: only {len(all_data)} valid days, skipping")
            continue

        # Deduplicate by date
        seen = set()
        deduped = []
        for row in all_data:
            dt = datetime.utcfromtimestamp(row['time']).strftime('%Y-%m-%d')
            if dt not in seen:
                seen.add(dt)
                deduped.append((dt, row))

        deduped.sort(key=lambda x: x[0])

        dates = np.array([d[0] for d in deduped])
        opens = np.array([d[1]['open'] for d in deduped], dtype=np.float64)
        highs = np.array([d[1]['high'] for d in deduped], dtype=np.float64)
        lows = np.array([d[1]['low'] for d in deduped], dtype=np.float64)
        closes = np.array([d[1]['close'] for d in deduped], dtype=np.float64)
        volumes = np.array([d[1].get('volumeto', 0) for d in deduped], dtype=np.float64)

        np.savez(out_path, dates=dates, open=opens, high=highs,
                 low=lows, close=closes, volume=volumes)
        print(f"  {fsym}: {len(dates)} days ({dates[0]} to {dates[-1]})")
        time.sleep(1.0)


def _save_ohlcv(ohlcv_list, out_path, symbol):
    """Convert ccxt OHLCV list to .npz and save."""
    # Deduplicate by date
    seen = set()
    deduped = []
    for c in ohlcv_list:
        dt = datetime.utcfromtimestamp(c[0] / 1000).strftime('%Y-%m-%d')
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, c))

    deduped.sort(key=lambda x: x[0])

    dates = np.array([d[0] for d in deduped])
    opens = np.array([d[1][1] for d in deduped], dtype=np.float64)
    highs = np.array([d[1][2] for d in deduped], dtype=np.float64)
    lows = np.array([d[1][3] for d in deduped], dtype=np.float64)
    closes = np.array([d[1][4] for d in deduped], dtype=np.float64)
    volumes = np.array([d[1][5] for d in deduped], dtype=np.float64)

    np.savez(out_path, dates=dates, open=opens, high=highs,
             low=lows, close=closes, volume=volumes)
    print(f"  {symbol}: {len(dates)} days ({dates[0]} to {dates[-1]})")


def main():
    parser = argparse.ArgumentParser(description="Fetch crypto OHLCV data")
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'binance', 'cryptocompare'])
    parser.add_argument('--output_dir', type=str, default='data/raw/crypto')
    args = parser.parse_args()

    out = Path(args.output_dir)

    if args.source in ('all', 'binance'):
        print(f"\n=== Fetching from Binance ({len(BINANCE_SYMBOLS)} symbols) ===")
        fetch_binance(out)

    if args.source in ('all', 'cryptocompare'):
        print(f"\n=== Fetching from CryptoCompare ({len(CRYPTOCOMPARE_SYMBOLS)} symbols) ===")
        fetch_cryptocompare(out)

    # Count results
    npz_files = list(out.glob('*.npz'))
    print(f"\nDone. {len(npz_files)} .npz files in {out}")


if __name__ == '__main__':
    main()
