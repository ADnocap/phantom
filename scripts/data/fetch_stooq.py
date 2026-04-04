#!/usr/bin/env python
"""
Fetch equity/ETF/forex/commodity OHLCV data via yfinance.

Saves one .npz per asset to data/raw/stooq/{asset_class}/{ticker}.npz

Usage:
  python scripts/data/fetch_stooq.py
  python scripts/data/fetch_stooq.py --asset_class etf
  python scripts/data/fetch_stooq.py --asset_class equity
"""

import argparse
import time
from pathlib import Path

import numpy as np


# ── Asset lists (Yahoo Finance tickers) ──────────────────────────

US_EQUITIES = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
    'TSLA', 'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL',
    'CSCO', 'QCOM', 'TXN', 'AVGO', 'MU', 'NOW',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK',
    'C', 'AXP', 'SCHW', 'USB', 'PNC', 'TFC',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY',
    'TMO', 'ABT', 'BMY', 'AMGN', 'GILD', 'MDT',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD',
    'NKE', 'SBUX', 'TGT', 'HD', 'LOW', 'DIS',
    # Industrial
    'CAT', 'BA', 'GE', 'HON', 'MMM', 'UPS',
    'DE', 'LMT', 'RTX', 'GD', 'NOC', 'WM',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
    'PSX', 'VLO', 'OXY', 'HAL',
    # Telecom / Utilities
    'T', 'VZ', 'TMUS', 'NEE', 'DUK', 'SO',
    'D', 'AEP', 'EXC', 'SRE',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA',
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM',
    # Additional
    'V', 'MA', 'PYPL', 'SHOP', 'UBER', 'ABNB',
    'F', 'GM', 'AAL', 'DAL', 'UAL', 'LUV',
    'MAR', 'HLT', 'CMG', 'YUM',
    'NFLX', 'ROKU', 'SPOT',
    'TSM', 'ASML', 'LRCX', 'KLAC', 'AMAT',
    'PANW', 'CRWD', 'ZS', 'FTNT', 'NET',
    'DDOG', 'MDB', 'PLTR', 'TEAM',
]

ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA',         # US indices
    'EEM', 'EFA', 'VWO', 'IEMG',         # International
    'TLT', 'IEF', 'SHY', 'HYG',          # Bonds
    'LQD', 'AGG', 'BND',                  # More bonds
    'GLD', 'SLV', 'GDX',                  # Precious metals
    'USO', 'UNG',                          # Energy
    'VNQ', 'IYR',                          # Real estate
    'XLF', 'XLE', 'XLK', 'XLV',          # Sectors
    'XLI', 'XLP', 'XLU', 'XLY',          # More sectors
    'ARKK',                                # Thematic
]

# Yahoo Finance forex tickers use format like EURUSD=X
FOREX = [
    'EURUSD=X', 'GBPUSD=X', 'JPY=X', 'AUDUSD=X', 'CAD=X',
    'CHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
    'AUDJPY=X', 'EURAUD=X', 'EURCHF=X', 'MXN=X', 'ZAR=X',
]

# Yahoo Finance commodity tickers
COMMODITIES = [
    'GC=F',   # Gold futures
    'SI=F',   # Silver futures
    'CL=F',   # Crude oil (WTI)
    'NG=F',   # Natural gas
    'HG=F',   # Copper
    'ZC=F',   # Corn
    'ZW=F',   # Wheat
    'ZS=F',   # Soybeans
    'KC=F',   # Coffee
    'SB=F',   # Sugar
]

EU_EQUITIES = [
    # UK
    'SHEL.L', 'HSBA.L', 'AZN.L', 'ULVR.L', 'BP.L',
    'RIO.L', 'GSK.L', 'BATS.L', 'DGE.L', 'LSEG.L',
    # Germany
    'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAS.DE',
    'BMW.DE', 'MBG.DE', 'MUV2.DE', 'ADS.DE', 'IFX.DE',
    # France
    'MC.PA', 'OR.PA', 'SAN.PA', 'AI.PA', 'SU.PA',
    'BNP.PA', 'ACA.PA', 'DG.PA', 'CS.PA', 'RI.PA',
    # Netherlands
    'ASML.AS', 'PHIA.AS', 'UNA.AS', 'INGA.AS', 'HEIA.AS',
    # Switzerland
    'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'ABBN.SW',
    # Other
    'NOVO-B.CO', 'ERIC-B.ST', 'VOLV-B.ST', 'SAN.MC', 'ISP.MI',
]

MIN_DAYS = 200
ASSET_CLASS_MAP = {
    'equity': US_EQUITIES,
    'etf': ETFS,
    'forex': FOREX,
    'commodity': COMMODITIES,
    'eu_equity': EU_EQUITIES,
}


def fetch_per_asset(output_dir: Path, asset_class: str, tickers: list,
                    start_date: str = '2000-01-01'):
    """Fetch assets one by one via yfinance."""
    import yfinance as yf

    out = output_dir / asset_class
    out.mkdir(parents=True, exist_ok=True)

    success, skip, fail = 0, 0, 0
    for ticker in tickers:
        safe_name = ticker.replace('/', '_').replace('.', '_').replace('=', '_')
        out_path = out / f"{safe_name}.npz"
        if out_path.exists():
            skip += 1
            continue

        print(f"  Fetching {ticker}...", end=' ', flush=True)
        try:
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if df is None or len(df) < MIN_DAYS:
                print(f"only {len(df) if df is not None else 0} rows, skipping")
                fail += 1
                continue

            # Flatten multi-level columns if present
            if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                df.columns = df.columns.get_level_values(0)

            dates = np.array([d.strftime('%Y-%m-%d') for d in df.index])
            opens = df['Open'].values.astype(np.float64)
            highs = df['High'].values.astype(np.float64)
            lows = df['Low'].values.astype(np.float64)
            closes = df['Close'].values.astype(np.float64)
            volumes = df['Volume'].values.astype(np.float64) if 'Volume' in df.columns else np.zeros(len(df))

            # Skip if too many NaN
            if np.isnan(closes).sum() > len(closes) * 0.05:
                print(f"too many NaN ({np.isnan(closes).sum()}), skipping")
                fail += 1
                continue

            # Forward-fill small gaps
            for arr in [opens, highs, lows, closes]:
                mask = np.isnan(arr)
                if mask.any():
                    idx = np.where(~mask, np.arange(len(arr)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    arr[mask] = arr[idx[mask]]

            np.savez(out_path, dates=dates, open=opens, high=highs,
                     low=lows, close=closes, volume=volumes)
            print(f"{len(dates)} days ({dates[0]} to {dates[-1]})")
            success += 1

        except Exception as e:
            print(f"ERROR: {e}")
            fail += 1

        time.sleep(0.3)  # Be polite to Yahoo

    print(f"  {asset_class}: {success} fetched, {skip} skipped, {fail} failed")


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data via yfinance")
    parser.add_argument('--asset_class', type=str, default='all',
                        choices=['all', 'equity', 'etf', 'forex', 'commodity', 'eu_equity'])
    parser.add_argument('--output_dir', type=str, default='data/raw/stooq')
    parser.add_argument('--start_date', type=str, default='2000-01-01')
    args = parser.parse_args()

    out = Path(args.output_dir)

    if args.asset_class == 'all':
        classes = list(ASSET_CLASS_MAP.keys())
    else:
        classes = [args.asset_class]

    for cls in classes:
        tickers = ASSET_CLASS_MAP[cls]
        print(f"\n=== Fetching {cls} ({len(tickers)} tickers) ===")
        fetch_per_asset(out, cls, tickers, args.start_date)

    # Count results
    total = sum(1 for _ in out.rglob('*.npz'))
    print(f"\nDone. {total} .npz files in {out}")


if __name__ == '__main__':
    main()
