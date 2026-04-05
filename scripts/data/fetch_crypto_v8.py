#!/usr/bin/env python
"""
Fetch daily OHLCV for ALL active USDT pairs on Binance for v8.

Dynamically queries Binance exchangeInfo for all active USDT trading pairs,
then fetches daily klines. Filters out stablecoins and leveraged tokens.

Usage:
  python scripts/data/fetch_crypto_v8.py
  python scripts/data/fetch_crypto_v8.py --workers 6 --min_days 300
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests


BASE_URL = 'https://api.binance.com'

# Skip stablecoins, leveraged tokens, and wrapped assets
SKIP_PATTERNS = [
    'USDCUSDT', 'TUSDUSDT', 'BUSDUSDT', 'DAIUSDT', 'USDPUSDT',
    'FDUSDUSDT', 'EURUSDT', 'GBPUSDT', 'TRYUSDT', 'BRLUSDT',
    'AEURUSDT', 'USDTUSDT',
]
SKIP_SUFFIXES = ['UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT']


def get_all_usdt_symbols():
    """Get all active USDT trading pairs from Binance, filtering junk."""
    r = requests.get(f'{BASE_URL}/api/v3/exchangeInfo', timeout=30)
    r.raise_for_status()
    symbols = []
    for s in r.json()['symbols']:
        sym = s['symbol']
        if (s['status'] == 'TRADING' and
            s['quoteAsset'] == 'USDT' and
            sym not in SKIP_PATTERNS and
            not any(sym.endswith(suf) for suf in SKIP_SUFFIXES)):
            symbols.append(sym)
    return sorted(symbols)


def fetch_daily_klines(symbol, min_days=200, start_date='2017-01-01', output_dir=None):
    """Fetch daily klines from Binance spot API."""
    output_dir = output_dir or Path('data/raw/crypto_v8')
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.replace('USDT', '_USDT')
    out_path = output_dir / f"{safe_name}.npz"
    if out_path.exists():
        return 'skip'

    start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)
    all_klines = []
    current = start_ms

    while current < end_ms:
        retries = 0
        while retries < 5:
            try:
                r = requests.get(f"{BASE_URL}/api/v3/klines",
                                 params={'symbol': symbol, 'interval': '1d',
                                         'startTime': current, 'limit': 1000},
                                 timeout=30)
                r.raise_for_status()
                klines = r.json()
                break
            except Exception as e:
                retries += 1
                if retries >= 5:
                    klines = []
                    break
                time.sleep(2)

        if not klines:
            break
        all_klines.extend(klines)
        last_ts = klines[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(0.1)

    if len(all_klines) < min_days:
        return 'short'

    # Deduplicate by date
    seen = set()
    deduped = []
    for k in all_klines:
        dt = datetime.utcfromtimestamp(k[0] / 1000).strftime('%Y-%m-%d')
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, k))
    deduped.sort(key=lambda x: x[0])

    dates = np.array([d[0] for d in deduped])
    opens = np.array([float(d[1][1]) for d in deduped], dtype=np.float64)
    highs = np.array([float(d[1][2]) for d in deduped], dtype=np.float64)
    lows = np.array([float(d[1][3]) for d in deduped], dtype=np.float64)
    closes = np.array([float(d[1][4]) for d in deduped], dtype=np.float64)
    volumes = np.array([float(d[1][5]) for d in deduped], dtype=np.float64)

    np.savez(out_path, dates=dates, open=opens, high=highs,
             low=lows, close=closes, volume=volumes)
    print(f"  {symbol}: {len(dates)} days ({dates[0]} to {dates[-1]})")
    return 'ok'


def main():
    parser = argparse.ArgumentParser(description="Fetch ALL Binance crypto for v8")
    parser.add_argument('--output_dir', type=str, default='data/raw/crypto_v8')
    parser.add_argument('--start_date', type=str, default='2017-01-01')
    parser.add_argument('--min_days', type=int, default=300,
                        help='Minimum days of data (default 300, need 120 context + 30 horizon + buffer)')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    print("Querying Binance for all active USDT pairs...")
    symbols = get_all_usdt_symbols()
    print(f"Found {len(symbols)} active USDT pairs (after filtering)")

    out_dir = Path(args.output_dir)
    existing = len(list(out_dir.glob('*.npz'))) if out_dir.exists() else 0
    print(f"Already have {existing} files in {out_dir}")

    ok, short, skip, fail = 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_daily_klines, sym, args.min_days,
                            args.start_date, out_dir): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result == 'ok':
                    ok += 1
                elif result == 'skip':
                    skip += 1
                elif result == 'short':
                    short += 1
            except Exception as e:
                print(f"  {sym}: FAILED — {e}")
                fail += 1

    n = len(list(out_dir.glob('*.npz')))
    print(f"\nDone: {n} files in {out_dir}")
    print(f"  Fetched: {ok}, Skipped (exists): {skip}, Too short: {short}, Failed: {fail}")


if __name__ == '__main__':
    main()
