#!/usr/bin/env python
"""
Fetch 4-hour OHLCV data from Binance for Phantom v7.

Saves .npz per symbol to data/raw/crypto_v7/ with keys:
  timestamps, open, high, low, close, volume

Usage:
  python scripts/data/fetch_crypto_v7.py
  python scripts/data/fetch_crypto_v7.py --workers 4
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests


SPOT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
    'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
    'MATICUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'FILUSDT',
    'NEARUSDT', 'APTUSDT', 'OPUSDT', 'ARBUSDT', 'AAVEUSDT',
    'EOSUSDT', 'XLMUSDT', 'TRXUSDT', 'ETCUSDT', 'XMRUSDT',
    'DASHUSDT', 'ZECUSDT', 'NEOUSDT', 'IOTAUSDT', 'XTZUSDT',
    'ALGOUSDT', 'VETUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT',
    'MANAUSDT', 'AXSUSDT', 'ENJUSDT', 'CRVUSDT', 'COMPUSDT',
    'SUSHIUSDT', 'YFIUSDT', 'SNXUSDT', 'MKRUSDT', 'BATUSDT',
    'ZRXUSDT', 'ICXUSDT', 'ONTUSDT', 'WAVESUSDT', 'QTUMUSDT',
    'SUIUSDT', 'SEIUSDT', 'TIAUSDT', 'INJUSDT', 'STXUSDT',
    'RNDRUSDT', 'FETUSDT', 'PEPEUSDT', 'WLDUSDT', 'JUPUSDT',
]

BASE_URL = 'https://api.binance.com'
# 720 context + 90 horizon = 810 bars minimum. At 6 bars/day = 135 days
MIN_BARS = 1000


def fetch_4h_klines(symbol, start_date='2017-01-01', output_dir=None):
    """Fetch 4h klines from Binance spot API."""
    output_dir = output_dir or Path('data/raw/crypto_v7')
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.replace('USDT', '_USDT')
    out_path = output_dir / f"{safe_name}.npz"
    if out_path.exists():
        print(f"  {symbol}: already exists, skipping")
        return

    print(f"  Fetching {symbol} 4h klines...")
    start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)

    all_klines = []
    current = start_ms

    while current < end_ms:
        params = {
            'symbol': symbol,
            'interval': '4h',
            'startTime': current,
            'limit': 1000,
        }
        retries = 0
        while retries < 5:
            try:
                r = requests.get(f"{BASE_URL}/api/v3/klines", params=params, timeout=30)
                r.raise_for_status()
                klines = r.json()
                break
            except Exception as e:
                retries += 1
                if retries >= 5:
                    print(f"    {symbol}: giving up after 5 retries")
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
        time.sleep(0.1)  # faster than daily (more requests needed)

    if len(all_klines) < MIN_BARS:
        print(f"  {symbol}: only {len(all_klines)} bars, skipping (need {MIN_BARS})")
        return

    # Deduplicate by open timestamp
    seen = set()
    deduped = []
    for k in all_klines:
        ts = k[0]
        if ts not in seen:
            seen.add(ts)
            deduped.append(k)
    deduped.sort(key=lambda x: x[0])

    # Build arrays
    timestamps = np.array([
        datetime.utcfromtimestamp(k[0] / 1000).strftime('%Y-%m-%dT%H:%M')
        for k in deduped
    ])
    opens = np.array([float(k[1]) for k in deduped], dtype=np.float64)
    highs = np.array([float(k[2]) for k in deduped], dtype=np.float64)
    lows = np.array([float(k[3]) for k in deduped], dtype=np.float64)
    closes = np.array([float(k[4]) for k in deduped], dtype=np.float64)
    volumes = np.array([float(k[5]) for k in deduped], dtype=np.float64)

    np.savez(out_path, timestamps=timestamps, open=opens, high=highs,
             low=lows, close=closes, volume=volumes)
    print(f"  {symbol}: {len(timestamps)} bars ({timestamps[0]} to {timestamps[-1]})")


def main():
    parser = argparse.ArgumentParser(description="Fetch 4h crypto data for v7")
    parser.add_argument('--output_dir', type=str, default='data/raw/crypto_v7')
    parser.add_argument('--start_date', type=str, default='2017-01-01')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    print(f"=== Fetching 4h OHLCV ({len(SPOT_SYMBOLS)} symbols, {args.workers} workers) ===")
    out_dir = Path(args.output_dir)

    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_4h_klines, sym, args.start_date, out_dir): sym
            for sym in SPOT_SYMBOLS
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                print(f"  {sym}: FAILED — {e}")
                failed += 1

    n = len(list(out_dir.glob('*.npz')))
    print(f"\nDone: {n} files in {out_dir} ({completed} fetched, {failed} failed)")


if __name__ == '__main__':
    main()
