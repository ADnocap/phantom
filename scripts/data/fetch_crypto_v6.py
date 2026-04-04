#!/usr/bin/env python
"""
Fetch crypto data for Phantom v6: OHLCV + taker buy volume, funding rates, open interest.

Three fetch modes:
  --source spot:    Binance spot klines with taker buy volume
  --source funding: Binance Futures funding rates (daily avg)
  --source oi:      Binance Futures open interest (daily)
  --source all:     All three

Saves .npz files to:
  data/raw/crypto_v6/{SYMBOL}.npz   (spot + taker buy)
  data/raw/funding/{SYMBOL}.npz     (funding rates)
  data/raw/oi/{SYMBOL}.npz          (open interest)

Usage:
  python scripts/data/fetch_crypto_v6.py
  python scripts/data/fetch_crypto_v6.py --source spot
  python scripts/data/fetch_crypto_v6.py --source funding
  python scripts/data/fetch_crypto_v6.py --source oi
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests


# ── Symbol lists ─────────────────────────────────────────────────────

# Spot pairs (Binance) — same as v5 fetch_crypto.py
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

# Perpetual futures symbols (for funding + OI)
# Subset of spot that have perpetual contracts on Binance
PERP_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
    'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
    'MATICUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'FILUSDT',
    'NEARUSDT', 'APTUSDT', 'OPUSDT', 'ARBUSDT', 'AAVEUSDT',
    'EOSUSDT', 'XLMUSDT', 'TRXUSDT', 'ETCUSDT',
    'XTZUSDT', 'ALGOUSDT', 'VETUSDT', 'THETAUSDT', 'FTMUSDT',
    'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'ENJUSDT', 'CRVUSDT',
    'COMPUSDT', 'SUSHIUSDT', 'YFIUSDT', 'SNXUSDT', 'MKRUSDT',
    'BATUSDT', 'ZRXUSDT',
    'SUIUSDT', 'SEIUSDT', 'TIAUSDT', 'INJUSDT', 'STXUSDT',
    'RNDRUSDT', 'FETUSDT', 'PEPEUSDT', 'WLDUSDT', 'JUPUSDT',
]

MIN_DAYS = 200
SPOT_BASE_URL = 'https://api.binance.com'
FUTURES_BASE_URL = 'https://fapi.binance.com'


# ── Spot OHLCV + Taker Buy Volume ────────────────────────────────────

def fetch_spot_klines(symbol, start_date='2017-01-01', output_dir=None):
    """Fetch daily klines from Binance spot API (includes taker buy volume)."""
    output_dir = output_dir or Path('data/raw/crypto_v6')
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.replace('USDT', '_USDT')
    out_path = output_dir / f"{safe_name}.npz"
    if out_path.exists():
        print(f"  {symbol}: already exists, skipping")
        return

    print(f"  Fetching {symbol} spot klines...")
    start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)

    all_klines = []
    current = start_ms

    while current < end_ms:
        url = f"{SPOT_BASE_URL}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': current,
            'limit': 1000,
        }
        retries = 0
        while retries < 5:
            try:
                r = requests.get(url, params=params, timeout=30)
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
        time.sleep(0.2)

    if len(all_klines) < MIN_DAYS:
        print(f"  {symbol}: only {len(all_klines)} days, skipping (need {MIN_DAYS})")
        return

    # Deduplicate by date
    seen = set()
    deduped = []
    for k in all_klines:
        dt = datetime.utcfromtimestamp(k[0] / 1000).strftime('%Y-%m-%d')
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, k))
    deduped.sort(key=lambda x: x[0])

    # Kline fields: [openTime, O, H, L, C, vol, closeTime,
    #                quoteVol, trades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
    dates = np.array([d[0] for d in deduped])
    opens = np.array([float(d[1][1]) for d in deduped], dtype=np.float64)
    highs = np.array([float(d[1][2]) for d in deduped], dtype=np.float64)
    lows = np.array([float(d[1][3]) for d in deduped], dtype=np.float64)
    closes = np.array([float(d[1][4]) for d in deduped], dtype=np.float64)
    volumes = np.array([float(d[1][5]) for d in deduped], dtype=np.float64)
    taker_buy_vol = np.array([float(d[1][9]) for d in deduped], dtype=np.float64)

    np.savez(out_path, dates=dates, open=opens, high=highs, low=lows,
             close=closes, volume=volumes, taker_buy_volume=taker_buy_vol)
    print(f"  {symbol}: {len(dates)} days ({dates[0]} to {dates[-1]})")


# ── Funding Rates ────────────────────────────────────────────────────

def fetch_funding_rates(symbol, output_dir=None):
    """Fetch historical funding rates from Binance Futures API."""
    output_dir = output_dir or Path('data/raw/funding')
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{symbol}.npz"
    if out_path.exists():
        print(f"  {symbol}: already exists, skipping")
        return

    print(f"  Fetching {symbol} funding rates...")
    all_rates = []
    # Start from 2019-09-01 (earliest Binance perps)
    start_ms = int(datetime(2019, 9, 1).timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)
    current = start_ms

    while current < end_ms:
        url = f"{FUTURES_BASE_URL}/fapi/v1/fundingRate"
        params = {
            'symbol': symbol,
            'startTime': current,
            'limit': 1000,
        }
        retries = 0
        while retries < 5:
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                retries += 1
                if retries >= 5:
                    print(f"    {symbol}: giving up after 5 retries")
                    data = []
                    break
                time.sleep(2)

        if not data:
            break

        all_rates.extend(data)
        last_ts = data[-1]['fundingTime']
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(0.2)

    if not all_rates:
        print(f"  {symbol}: no funding data found")
        return

    # Aggregate to daily: mean of the 3 daily readings (every 8h)
    daily = defaultdict(list)
    for r in all_rates:
        dt = datetime.utcfromtimestamp(r['fundingTime'] / 1000).strftime('%Y-%m-%d')
        daily[dt].append(float(r['fundingRate']))

    sorted_dates = sorted(daily.keys())
    dates = np.array(sorted_dates)
    funding = np.array([np.mean(daily[d]) for d in sorted_dates], dtype=np.float64)

    if len(dates) < 30:
        print(f"  {symbol}: only {len(dates)} days of funding, skipping")
        return

    np.savez(out_path, dates=dates, funding_rate=funding)
    print(f"  {symbol}: {len(dates)} days ({dates[0]} to {dates[-1]})")


# ── Open Interest ────────────────────────────────────────────────────

def fetch_open_interest(symbol, output_dir=None):
    """Fetch historical open interest from Binance Futures API."""
    output_dir = output_dir or Path('data/raw/oi')
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{symbol}.npz"
    if out_path.exists():
        print(f"  {symbol}: already exists, skipping")
        return

    print(f"  Fetching {symbol} open interest...")
    all_oi = []
    # OI history typically starts from mid-2020, not 2019
    start_ms = int(datetime(2020, 7, 1).timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)
    current = start_ms
    max_retries = 3

    while current < end_ms:
        url = f"{FUTURES_BASE_URL}/futures/data/openInterestHist"
        params = {
            'symbol': symbol,
            'period': '1d',
            'startTime': current,
            'limit': 500,
        }
        retries = 0
        while retries < max_retries:
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"    {symbol}: giving up after {max_retries} retries at {current}")
                    data = []
                    break
                time.sleep(2)

        if not data:
            break

        all_oi.extend(data)
        last_ts = data[-1]['timestamp']
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(0.5)

    if not all_oi:
        print(f"  {symbol}: no OI data found")
        return

    # Deduplicate by date
    seen = set()
    deduped = []
    for entry in all_oi:
        dt = datetime.utcfromtimestamp(entry['timestamp'] / 1000).strftime('%Y-%m-%d')
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, entry))
    deduped.sort(key=lambda x: x[0])

    if len(deduped) < 30:
        print(f"  {symbol}: only {len(deduped)} days of OI, skipping")
        return

    dates = np.array([d[0] for d in deduped])
    oi = np.array([float(d[1]['sumOpenInterestValue']) for d in deduped], dtype=np.float64)

    np.savez(out_path, dates=dates, open_interest=oi)
    print(f"  {symbol}: {len(dates)} days ({dates[0]} to {dates[-1]})")


# ── Main ─────────────────────────────────────────────────────────────

def fetch_parallel(func, symbols, n_workers=4, **kwargs):
    """Run fetch function across symbols in parallel using thread pool."""
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(func, sym, **kwargs): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                print(f"  {sym}: FAILED — {e}")
                failed += 1
    return completed, failed


def main():
    parser = argparse.ArgumentParser(description="Fetch crypto data for v6")
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'spot', 'funding', 'oi'])
    parser.add_argument('--spot_dir', type=str, default='data/raw/crypto_v6')
    parser.add_argument('--funding_dir', type=str, default='data/raw/funding')
    parser.add_argument('--oi_dir', type=str, default='data/raw/oi')
    parser.add_argument('--start_date', type=str, default='2017-01-01')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel download workers (default 4)')
    args = parser.parse_args()

    if args.source in ('all', 'spot'):
        print(f"\n=== Fetching Spot OHLCV + Taker Buy ({len(SPOT_SYMBOLS)} symbols, {args.workers} workers) ===")
        spot_dir = Path(args.spot_dir)
        ok, fail = fetch_parallel(
            fetch_spot_klines, SPOT_SYMBOLS, n_workers=args.workers,
            start_date=args.start_date, output_dir=spot_dir)
        n = len(list(spot_dir.glob('*.npz')))
        print(f"Done: {n} files in {spot_dir} ({ok} fetched, {fail} failed)")

    if args.source in ('all', 'funding'):
        print(f"\n=== Fetching Funding Rates ({len(PERP_SYMBOLS)} symbols, {args.workers} workers) ===")
        funding_dir = Path(args.funding_dir)
        ok, fail = fetch_parallel(
            fetch_funding_rates, PERP_SYMBOLS, n_workers=args.workers,
            output_dir=funding_dir)
        n = len(list(funding_dir.glob('*.npz')))
        print(f"Done: {n} files in {funding_dir} ({ok} fetched, {fail} failed)")

    if args.source in ('all', 'oi'):
        print(f"\n=== Fetching Open Interest ({len(PERP_SYMBOLS)} symbols, {args.workers} workers) ===")
        oi_dir = Path(args.oi_dir)
        ok, fail = fetch_parallel(
            fetch_open_interest, PERP_SYMBOLS, n_workers=args.workers,
            output_dir=oi_dir)
        n = len(list(oi_dir.glob('*.npz')))
        print(f"Done: {n} files in {oi_dir} ({ok} fetched, {fail} failed)")


if __name__ == '__main__':
    main()
