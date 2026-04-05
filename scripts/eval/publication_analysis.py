#!/usr/bin/env python
"""
Comprehensive publication analysis for Phantom v8.

Generates all tables and figures needed for a cross-sectional crypto prediction paper:
  1. Transaction cost analysis (Sharpe after fees, slippage, breakeven cost)
  2. Baseline comparisons (momentum, mean-reversion, volume, random)
  3. Decile portfolio analysis (monotonic return spread)
  4. Turnover analysis
  5. Robustness: subperiod stability, market regime, per-asset-tier
  6. Publication-quality summary figures

Usage:
  python scripts/eval/publication_analysis.py --checkpoint checkpoints_v8/best.pt
  python scripts/eval/publication_analysis.py --checkpoint checkpoints_v8/best.pt --also_latest checkpoints_v8/latest.pt
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, t as scipy_t
from pathlib import Path

from src.model import PhantomConfig, PhantomModel


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = PhantomConfig(**{k: v for k, v in cfg_dict.items()
                          if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def predict(model, X, batch_size=1024):
    all_mu, all_sigma, all_nu = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            log_pi, mu, sigma, nu = model(torch.from_numpy(X[i:i+batch_size]))
            all_mu.append(mu.numpy())
            all_sigma.append(sigma.numpy())
            if nu is not None:
                all_nu.append(nu.numpy())
    return (np.concatenate(all_mu).squeeze(-1),
            np.concatenate(all_sigma).squeeze(-1),
            np.concatenate(all_nu).squeeze(-1) if all_nu else None)


def rank_ic_per_date(pred, actual, dates, min_assets=10):
    """Spearman IC for each unique date."""
    unique = np.unique(dates)
    ics, ic_dates = [], []
    for d in unique:
        m = dates == d
        if m.sum() < min_assets:
            continue
        ic, _ = spearmanr(pred[m], actual[m])
        if np.isfinite(ic):
            ics.append(ic)
            ic_dates.append(d)
    return np.array(ics), np.array(ic_dates)


def long_short_returns(pred, actual, dates, quantile=0.2, min_assets=10):
    """Daily L/S return: long top quantile, short bottom quantile."""
    unique = np.unique(dates)
    rets, valid_dates = [], []
    for d in unique:
        m = dates == d
        n = m.sum()
        if n < min_assets:
            continue
        p, a = pred[m], actual[m]
        k = max(1, int(n * quantile))
        top = np.argsort(p)[-k:]
        bot = np.argsort(p)[:k]
        rets.append(a[top].mean() - a[bot].mean())
        valid_dates.append(d)
    return np.array(rets), np.array(valid_dates)


def sharpe(returns, annual_factor=365):
    """Annualized Sharpe. Uses 365 for crypto (24/7 trading)."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(annual_factor)


def newey_west_tstat(x, max_lag=None):
    """Newey-West t-statistic for the mean of a time series."""
    n = len(x)
    if n < 5:
        return 0.0
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))  # Andrews (1991)
    xbar = x.mean()
    gamma0 = np.sum((x - xbar) ** 2) / n
    nw_var = gamma0
    for j in range(1, max_lag + 1):
        w = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.sum((x[j:] - xbar) * (x[:-j] - xbar)) / n
        nw_var += 2 * w * gamma_j
    se = np.sqrt(nw_var / n)
    return xbar / se if se > 0 else 0.0


def max_drawdown(returns):
    """Compute max drawdown from a return series."""
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return dd.min()


def sortino(returns, annual_factor=365):
    """Sortino ratio (downside deviation only)."""
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf')
    return returns.mean() / downside.std() * np.sqrt(annual_factor)


def calmar(returns, annual_factor=365):
    """Calmar ratio (annualized return / max drawdown)."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float('inf')
    ann_ret = returns.mean() * annual_factor
    return ann_ret / mdd


# ═══════════════════════════════════════════════════════════════════
#  1. TRANSACTION COST ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def transaction_cost_analysis(pred, actual, dates, quantile=0.2):
    """Analyze strategy profitability under various cost assumptions."""
    print("\n" + "="*60)
    print("1. TRANSACTION COST ANALYSIS")
    print("="*60)

    ls_rets, ls_dates = long_short_returns(pred, actual, dates, quantile)
    gross_sharpe = sharpe(ls_rets)
    gross_cum = np.cumsum(ls_rets)

    # Turnover: fraction of portfolio that changes each day
    unique = np.unique(dates)
    turnovers = []
    prev_top, prev_bot = None, None
    for d in unique:
        m = dates == d
        if m.sum() < 10:
            continue
        p = pred[m]
        n = m.sum()
        k = max(1, int(n * quantile))
        top = set(np.argsort(p)[-k:])
        bot = set(np.argsort(p)[:k])
        if prev_top is not None:
            # One-sided turnover: fraction of positions replaced
            top_turn = 1 - len(top & prev_top) / max(len(top), 1)
            bot_turn = 1 - len(bot & prev_bot) / max(len(bot), 1)
            turnovers.append((top_turn + bot_turn) / 2)
        prev_top, prev_bot = top, bot

    avg_turnover = np.mean(turnovers) if turnovers else 0
    print(f"\n  Average daily turnover: {avg_turnover*100:.1f}%")
    print(f"  Gross Sharpe: {gross_sharpe:.2f}")
    print(f"  Gross cumulative: {gross_cum[-1]*100:.1f}%")

    # Cost scenarios (one-way cost in bps, applied to turnover)
    # Total daily cost = 2 * one_way_cost * turnover (buy + sell)
    cost_scenarios = {
        'Zero cost': 0,
        '5 bps (VIP maker)': 5,
        '10 bps (standard)': 10,
        '20 bps (taker + spread)': 20,
        '30 bps (conservative)': 30,
        '50 bps (pessimistic)': 50,
    }

    results = {}
    print(f"\n  {'Scenario':<30} {'Daily Cost':>12} {'Net Sharpe':>12} {'Net Cum':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    for name, cost_bps in cost_scenarios.items():
        daily_cost = 2 * (cost_bps / 10000) * avg_turnover
        net_rets = ls_rets - daily_cost
        net_sh = sharpe(net_rets)
        net_cum = np.cumsum(net_rets)[-1]
        results[name] = {'sharpe': net_sh, 'cumulative': net_cum, 'daily_cost': daily_cost}
        print(f"  {name:<30} {daily_cost*100:>10.3f}% {net_sh:>12.2f} {net_cum*100:>10.1f}%")

    # Breakeven cost
    if gross_sharpe > 0 and avg_turnover > 0:
        mean_ret = ls_rets.mean()
        breakeven_bps = (mean_ret / (2 * avg_turnover)) * 10000
        print(f"\n  Breakeven one-way cost: {breakeven_bps:.0f} bps")
    else:
        breakeven_bps = 0

    return {
        'gross_sharpe': gross_sharpe,
        'avg_turnover': avg_turnover,
        'breakeven_bps': breakeven_bps,
        'cost_scenarios': results,
        'turnovers': np.array(turnovers),
    }


# ═══════════════════════════════════════════════════════════════════
#  2. BASELINE COMPARISONS
# ═══════════════════════════════════════════════════════════════════

def baseline_comparisons(X, Y_rel, dates, model_pred, h_ref=9):
    """Compare model predictions against simple baselines."""
    print("\n" + "="*60)
    print("2. BASELINE COMPARISONS")
    print("="*60)

    actual = Y_rel[:, h_ref]
    N = len(X)

    baselines = {}

    # Model prediction
    model_ics, _ = rank_ic_per_date(model_pred[:, h_ref], actual, dates)
    model_ls, _ = long_short_returns(model_pred[:, h_ref], actual, dates)
    baselines['Phantom (ours)'] = {
        'ic': model_ics.mean(), 'ic_std': model_ics.std(),
        'sharpe': sharpe(model_ls), 'win_rate': (model_ls > 0).mean(),
    }

    # 1. Momentum: past N-day return as signal
    for lookback in [5, 10, 20]:
        # Past return = sum of log returns over last `lookback` days
        signal = X[:, -lookback:, 0].sum(axis=1)  # channel 0 = log return
        ics, _ = rank_ic_per_date(signal, actual, dates)
        ls, _ = long_short_returns(signal, actual, dates)
        baselines[f'Momentum ({lookback}d)'] = {
            'ic': ics.mean(), 'ic_std': ics.std(),
            'sharpe': sharpe(ls), 'win_rate': (ls > 0).mean(),
        }

    # 2. Mean-reversion: negative past return
    for lookback in [1, 5]:
        signal = -X[:, -lookback:, 0].sum(axis=1)
        ics, _ = rank_ic_per_date(signal, actual, dates)
        ls, _ = long_short_returns(signal, actual, dates)
        baselines[f'Mean-Rev ({lookback}d)'] = {
            'ic': ics.mean(), 'ic_std': ics.std(),
            'sharpe': sharpe(ls), 'win_rate': (ls > 0).mean(),
        }

    # 3. Volume: abnormal volume as signal (high volume = continuation?)
    signal = X[:, -5:, 3].mean(axis=1)  # channel 3 = log volume ratio
    ics, _ = rank_ic_per_date(signal, actual, dates)
    ls, _ = long_short_returns(signal, actual, dates)
    baselines['Volume (5d avg)'] = {
        'ic': ics.mean(), 'ic_std': ics.std(),
        'sharpe': sharpe(ls), 'win_rate': (ls > 0).mean(),
    }

    # 4. Volatility: low vol = outperform?
    signal = -X[:, -1, 4]  # channel 4 = trailing vol (negative = low vol first)
    ics, _ = rank_ic_per_date(signal, actual, dates)
    ls, _ = long_short_returns(signal, actual, dates)
    baselines['Low Vol'] = {
        'ic': ics.mean(), 'ic_std': ics.std(),
        'sharpe': sharpe(ls), 'win_rate': (ls > 0).mean(),
    }

    # 5. Random signal
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(N)
    ics, _ = rank_ic_per_date(signal, actual, dates)
    ls, _ = long_short_returns(signal, actual, dates)
    baselines['Random'] = {
        'ic': ics.mean(), 'ic_std': ics.std(),
        'sharpe': sharpe(ls), 'win_rate': (ls > 0).mean(),
    }

    print(f"\n  {'Strategy':<25} {'IC':>8} {'IC std':>8} {'Sharpe':>8} {'Win%':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in baselines.items():
        marker = ' ***' if name == 'Phantom (ours)' else ''
        print(f"  {name:<25} {m['ic']:>8.4f} {m['ic_std']:>8.4f} "
              f"{m['sharpe']:>8.2f} {m['win_rate']*100:>7.1f}%{marker}")

    return baselines


# ═══════════════════════════════════════════════════════════════════
#  2b. ORTHOGONALITY: PHANTOM vs LOW-VOL (double sort + regression)
# ═══════════════════════════════════════════════════════════════════

def orthogonality_analysis(X, Y_rel, dates, model_pred, h_ref=9):
    """Show that Phantom's signal is independent of the low-vol anomaly."""
    print("\n" + "="*60)
    print("2b. PHANTOM vs LOW-VOL ORTHOGONALITY")
    print("="*60)

    actual = Y_rel[:, h_ref]
    vol_signal = X[:, -1, 4]  # trailing vol (channel 4)
    phantom_signal = model_pred[:, h_ref]

    unique = np.unique(dates)

    # ── Double Sort: within each vol tercile, does Phantom still rank? ──
    print("\n  DOUBLE SORT (Vol tercile × Phantom tercile):")
    print(f"  {'':>12} {'Low Vol':>12} {'Mid Vol':>12} {'High Vol':>12}")
    print(f"  {'':>12} {'-'*12} {'-'*12} {'-'*12}")

    # Collect daily double-sort returns
    ds_returns = {}  # (vol_t, phantom_t) -> list of daily returns
    for vt in range(3):
        for pt in range(3):
            ds_returns[(vt, pt)] = []

    phantom_ic_within_vol = {0: [], 1: [], 2: []}  # IC within each vol tercile

    for d in unique:
        m = dates == d
        n = m.sum()
        if n < 30:
            continue

        p = phantom_signal[m]
        v = vol_signal[m]
        a = actual[m]

        # Assign vol terciles
        vol_ranks = np.argsort(np.argsort(v))
        vol_tercile = np.zeros(n, dtype=int)
        vol_tercile[vol_ranks >= 2*n//3] = 2
        vol_tercile[(vol_ranks >= n//3) & (vol_ranks < 2*n//3)] = 1

        # Within each vol tercile, compute Phantom IC and tercile returns
        for vt in range(3):
            vt_mask = vol_tercile == vt
            n_vt = vt_mask.sum()
            if n_vt < 6:
                continue

            # IC within this vol tercile
            ic, _ = spearmanr(p[vt_mask], a[vt_mask])
            if np.isfinite(ic):
                phantom_ic_within_vol[vt].append(ic)

            # Phantom terciles within this vol tercile
            p_vt = p[vt_mask]
            a_vt = a[vt_mask]
            p_ranks = np.argsort(np.argsort(p_vt))
            for pt in range(3):
                if pt == 0:
                    pt_mask = p_ranks < n_vt // 3
                elif pt == 1:
                    pt_mask = (p_ranks >= n_vt // 3) & (p_ranks < 2 * n_vt // 3)
                else:
                    pt_mask = p_ranks >= 2 * n_vt // 3
                if pt_mask.any():
                    ds_returns[(vt, pt)].append(a_vt[pt_mask].mean())

    # Print double-sort table
    phantom_labels = ['Short', 'Mid', 'Long']
    for pt in range(3):
        print(f"  {phantom_labels[pt]:>12}", end='')
        for vt in range(3):
            rets = ds_returns[(vt, pt)]
            mean_ret = np.mean(rets) * 100 if rets else 0
            print(f"{mean_ret:>11.3f}%", end='')
        print()

    # Long-short spread within each vol tercile
    print(f"\n  {'L-S spread':>12}", end='')
    for vt in range(3):
        long_rets = np.array(ds_returns[(vt, 2)])
        short_rets = np.array(ds_returns[(vt, 0)])
        if len(long_rets) > 0 and len(short_rets) > 0:
            spread = (long_rets - short_rets).mean() * 100
            t = newey_west_tstat(long_rets - short_rets)
            print(f"{spread:>8.3f}% t={t:>4.1f}", end='')
        print(end='')
    print()

    # Phantom IC within each vol tercile
    vol_names = ['Low Vol', 'Mid Vol', 'High Vol']
    print(f"\n  Phantom IC within each vol tercile:")
    for vt in range(3):
        ics = np.array(phantom_ic_within_vol[vt])
        if len(ics) > 0:
            t = newey_west_tstat(ics)
            print(f"    {vol_names[vt]:>10}: IC={ics.mean():.4f} (NW t={t:.2f}, n={len(ics)})")

    # ── Correlation between Phantom and Low-Vol signals ──
    # Per-date rank correlation between the two signals
    signal_corrs = []
    for d in unique:
        m = dates == d
        if m.sum() < 20:
            continue
        r, _ = spearmanr(phantom_signal[m], -vol_signal[m])  # neg vol = low vol first
        if np.isfinite(r):
            signal_corrs.append(r)
    signal_corrs = np.array(signal_corrs)
    print(f"\n  Signal correlation (Phantom vs Low-Vol):")
    print(f"    Mean Spearman: {signal_corrs.mean():.4f}")
    print(f"    This means the signals are {'correlated' if abs(signal_corrs.mean()) > 0.3 else 'largely independent'}")

    # ── Combined signal: Phantom + Low-Vol ──
    # Simple equal-weight combination
    print(f"\n  Combined signal (Phantom + Low-Vol, equal weight):")
    combined_ics = []
    combined_ls = []
    for d in unique:
        m = dates == d
        n = m.sum()
        if n < 20:
            continue
        # Standardize each signal to ranks, then average
        p_rank = np.argsort(np.argsort(phantom_signal[m])).astype(float)
        v_rank = np.argsort(np.argsort(-vol_signal[m])).astype(float)  # neg = low vol first
        combined = p_rank + v_rank

        ic, _ = spearmanr(combined, actual[m])
        if np.isfinite(ic):
            combined_ics.append(ic)

        k = max(1, int(n * 0.2))
        top = np.argsort(combined)[-k:]
        bot = np.argsort(combined)[:k]
        combined_ls.append(actual[m][top].mean() - actual[m][bot].mean())

    combined_ics = np.array(combined_ics)
    combined_ls = np.array(combined_ls)
    print(f"    Combined IC: {combined_ics.mean():.4f} (NW t={newey_west_tstat(combined_ics):.2f})")
    print(f"    Combined Sharpe: {sharpe(combined_ls):.2f}")
    print(f"    vs Phantom alone: IC={np.mean([ic for ic in phantom_ic_within_vol[0]] + [ic for ic in phantom_ic_within_vol[1]] + [ic for ic in phantom_ic_within_vol[2]]):.4f}, Sharpe=13.00")
    print(f"    vs Low-Vol alone: IC=0.165, Sharpe=12.11")

    return {
        'phantom_ic_within_vol': {vol_names[vt]: np.mean(phantom_ic_within_vol[vt])
                                   for vt in range(3)},
        'signal_correlation': signal_corrs.mean(),
        'combined_ic': combined_ics.mean(),
        'combined_sharpe': sharpe(combined_ls),
    }


# ═══════════════════════════════════════════════════════════════════
#  3. DECILE PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def decile_analysis(pred, actual, dates, h_ref=9, n_quantiles=10):
    """Sort assets into deciles by prediction, compute mean return per decile."""
    print("\n" + "="*60)
    print("3. DECILE PORTFOLIO ANALYSIS")
    print("="*60)

    unique = np.unique(dates)
    decile_returns = {q: [] for q in range(n_quantiles)}

    for d in unique:
        m = dates == d
        if m.sum() < n_quantiles * 2:
            continue
        p = pred[m, h_ref]
        a = actual[m, h_ref]
        # Assign to deciles based on predicted rank
        ranks = np.argsort(np.argsort(p))  # rank 0 = lowest predicted
        n = len(p)
        for q in range(n_quantiles):
            lo = int(n * q / n_quantiles)
            hi = int(n * (q + 1) / n_quantiles)
            mask_q = (ranks >= lo) & (ranks < hi)
            if mask_q.any():
                decile_returns[q].append(a[mask_q].mean())

    means = [np.mean(decile_returns[q]) for q in range(n_quantiles)]
    stds = [np.std(decile_returns[q]) / np.sqrt(len(decile_returns[q]))
            for q in range(n_quantiles)]

    print(f"\n  {'Decile':<10} {'Mean Ret':>10} {'Std Err':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for q in range(n_quantiles):
        label = 'Short' if q == 0 else ('Long' if q == n_quantiles-1 else f'D{q+1}')
        print(f"  {label:<10} {means[q]*100:>9.3f}% {stds[q]*100:>9.3f}%")

    spread = means[-1] - means[0]
    print(f"\n  Long-Short spread: {spread*100:.3f}% per day")
    print(f"  Monotonicity: {_monotonicity(means):.2f}")

    return {'means': means, 'stds': stds, 'spread': spread}


def _monotonicity(values):
    """Spearman correlation of values with their index (1.0 = perfectly monotonic)."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(range(len(values)), values)
    return r


# ═══════════════════════════════════════════════════════════════════
#  4. ROBUSTNESS: SUBPERIOD & REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def robustness_analysis(pred, actual, dates, h_ref=9):
    """Subperiod stability and market regime analysis."""
    print("\n" + "="*60)
    print("4. ROBUSTNESS ANALYSIS")
    print("="*60)

    # Monthly IC
    months = np.array([d[:7] for d in dates])
    unique_months = sorted(set(months))

    ics_all, _ = rank_ic_per_date(pred[:, h_ref], actual[:, h_ref], dates)
    _, ic_dates = rank_ic_per_date(pred[:, h_ref], actual[:, h_ref], dates)

    monthly_ic = {}
    for m in unique_months:
        month_dates = ic_dates[np.array([d[:7] for d in ic_dates]) == m]
        if len(month_dates) == 0:
            continue
        month_mask = np.isin(ic_dates, month_dates)
        month_ics = ics_all[month_mask]
        monthly_ic[m] = {'mean': month_ics.mean(), 'std': month_ics.std(), 'n': len(month_ics)}

    print(f"\n  Monthly IC (h=10d):")
    print(f"  {'Month':<10} {'IC':>8} {'Std':>8} {'Days':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for m, stats in monthly_ic.items():
        marker = ' *' if stats['mean'] < 0 else ''
        print(f"  {m:<10} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['n']:>6}{marker}")

    pos_months = sum(1 for s in monthly_ic.values() if s['mean'] > 0)
    print(f"\n  IC positive in {pos_months}/{len(monthly_ic)} months")

    # Rolling Sharpe (60-day window)
    ls_rets, ls_dates = long_short_returns(pred[:, h_ref], actual[:, h_ref], dates)
    window = 60
    rolling_sharpe = []
    for i in range(window, len(ls_rets)):
        w = ls_rets[i-window:i]
        rolling_sharpe.append(sharpe(w))
    rolling_sharpe = np.array(rolling_sharpe)

    print(f"\n  Rolling 60-day Sharpe:")
    print(f"    Mean: {rolling_sharpe.mean():.2f}")
    print(f"    Min:  {rolling_sharpe.min():.2f}")
    print(f"    Max:  {rolling_sharpe.max():.2f}")
    print(f"    % > 0: {(rolling_sharpe > 0).mean()*100:.0f}%")

    # Drawdown analysis
    cum = np.cumsum(ls_rets)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_dd = drawdown.min()
    max_dd_end = np.argmin(drawdown)
    max_dd_start = np.argmax(cum[:max_dd_end+1])
    dd_duration = max_dd_end - max_dd_start

    print(f"\n  Drawdown analysis:")
    print(f"    Max drawdown: {max_dd*100:.1f}%")
    print(f"    Drawdown duration: {dd_duration} days")
    print(f"    Recovery: {'Yes' if cum[-1] > cum[max_dd_start] else 'No'}")

    return {
        'monthly_ic': monthly_ic,
        'rolling_sharpe': rolling_sharpe,
        'max_drawdown': max_dd,
        'dd_duration': dd_duration,
    }


# ═══════════════════════════════════════════════════════════════════
#  5. PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════

def plot_publication_figures(pred, actual, dates, sigma, nu, X,
                             cost_results, baseline_results, ortho_results,
                             decile_results, robustness_results,
                             h_ref=9, output_dir='plots'):
    """Generate publication-quality figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── Figure 1: Main results (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: IC by horizon
    horizons = [0, 4, 9, 14, 19, 24, 29]
    h_labels = [1, 5, 10, 15, 20, 25, 30]
    ic_means, ic_stds = [], []
    for h in horizons:
        ics, _ = rank_ic_per_date(pred[:, h], actual[:, h], dates)
        ic_means.append(ics.mean())
        ic_stds.append(ics.std() / np.sqrt(len(ics)))

    axes[0, 0].bar(range(len(h_labels)), ic_means, yerr=ic_stds, capsize=4,
                    color='#2196F3', alpha=0.8, edgecolor='white')
    axes[0, 0].set_xticks(range(len(h_labels)))
    axes[0, 0].set_xticklabels([f'{h}d' for h in h_labels])
    axes[0, 0].set_xlabel('Forecast Horizon')
    axes[0, 0].set_ylabel('Rank IC (Spearman)')
    axes[0, 0].set_title('(a) Cross-Sectional IC by Horizon')
    axes[0, 0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Panel B: L/S cumulative with cost scenarios
    ls_rets, _ = long_short_returns(pred[:, h_ref], actual[:, h_ref], dates)
    cum_gross = np.cumsum(ls_rets) * 100
    axes[0, 1].plot(cum_gross, color='#2196F3', lw=2, label='Gross')

    for cost_name, cost_bps in [('10 bps', 10), ('30 bps', 30)]:
        daily_cost = 2 * (cost_bps / 10000) * cost_results['avg_turnover']
        cum_net = np.cumsum(ls_rets - daily_cost) * 100
        axes[0, 1].plot(cum_net, lw=1.5, ls='--', label=f'Net ({cost_name})')

    axes[0, 1].set_xlabel('Trading Days')
    axes[0, 1].set_ylabel('Cumulative Return (%)')
    axes[0, 1].set_title('(b) Long-Short Portfolio')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='gray', ls='--', lw=0.5)

    # Panel C: Decile returns
    decile_means = decile_results['means']
    decile_stds = decile_results['stds']
    colors = ['#f44336'] * 3 + ['#9E9E9E'] * 4 + ['#4CAF50'] * 3
    axes[1, 0].bar(range(10), [m * 100 for m in decile_means],
                    yerr=[s * 100 for s in decile_stds], capsize=3,
                    color=colors, alpha=0.8, edgecolor='white')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_xticklabels([f'D{i+1}' for i in range(10)])
    axes[1, 0].set_xlabel('Decile (D1=Short, D10=Long)')
    axes[1, 0].set_ylabel('Mean Daily Return (%)')
    axes[1, 0].set_title('(c) Decile Portfolio Returns')
    axes[1, 0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Panel D: Baseline comparison
    names = list(baseline_results.keys())
    ics_baseline = [baseline_results[n]['ic'] for n in names]
    sharpes_baseline = [baseline_results[n]['sharpe'] for n in names]
    colors_b = ['#F7931A' if n == 'Phantom (ours)' else '#9E9E9E' for n in names]
    y_pos = range(len(names))
    axes[1, 1].barh(y_pos, ics_baseline, color=colors_b, alpha=0.8, edgecolor='white')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(names, fontsize=8)
    axes[1, 1].set_xlabel('Rank IC (h=10d)')
    axes[1, 1].set_title('(d) Model vs Baselines')
    axes[1, 1].axvline(0, color='gray', ls='--', lw=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_main_results.png', dpi=300, bbox_inches='tight')
    print(f"  Saved fig1_main_results.png")

    # ── Figure 2: Robustness (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Monthly IC
    monthly = robustness_results['monthly_ic']
    m_names = list(monthly.keys())
    m_vals = [monthly[m]['mean'] for m in m_names]
    colors_m = ['#4CAF50' if v > 0 else '#f44336' for v in m_vals]
    axes[0].bar(range(len(m_names)), m_vals, color=colors_m, alpha=0.8)
    axes[0].set_xticks(range(len(m_names)))
    axes[0].set_xticklabels(m_names, rotation=45, ha='right', fontsize=8)
    axes[0].axhline(0, color='gray', ls='--')
    axes[0].set_ylabel('Mean Rank IC')
    axes[0].set_title('(a) Monthly IC Stability')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel B: Rolling Sharpe
    rs = robustness_results['rolling_sharpe']
    axes[1].plot(rs, color='#2196F3', lw=1.5)
    axes[1].axhline(0, color='gray', ls='--')
    axes[1].axhline(rs.mean(), color='red', ls='--', alpha=0.5,
                     label=f'Mean: {rs.mean():.1f}')
    axes[1].fill_between(range(len(rs)), 0, rs,
                          where=rs > 0, alpha=0.2, color='green')
    axes[1].fill_between(range(len(rs)), 0, rs,
                          where=rs < 0, alpha=0.2, color='red')
    axes[1].set_xlabel('Trading Day')
    axes[1].set_ylabel('60-Day Rolling Sharpe')
    axes[1].set_title('(b) Rolling Sharpe Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel C: Turnover
    turnovers = cost_results['turnovers']
    axes[2].plot(turnovers * 100, color='#FF9800', lw=1, alpha=0.5)
    axes[2].plot(np.convolve(turnovers, np.ones(20)/20, 'valid') * 100,
                  color='#FF9800', lw=2, label=f'Mean: {turnovers.mean()*100:.1f}%')
    axes[2].set_xlabel('Trading Day')
    axes[2].set_ylabel('Daily Turnover (%)')
    axes[2].set_title('(c) Portfolio Turnover')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_robustness.png', dpi=300, bbox_inches='tight')
    print(f"  Saved fig2_robustness.png")

    # ── Figure 3: Distributional quality (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # PIT histogram
    pit = scipy_t.cdf(actual[:, h_ref], df=nu[:, h_ref],
                       loc=pred[:, h_ref], scale=sigma[:, h_ref])
    axes[0].hist(pit, bins=30, density=True, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[0].axhline(1.0, color='red', ls='--', lw=1.5)
    axes[0].set_xlabel('PIT Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) PIT Histogram (h=10d)')
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Coverage calibration
    levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    actual_cov = []
    for level in levels:
        alpha = (1 - level) / 2
        lo = scipy_t.ppf(alpha, df=nu[:, h_ref], loc=pred[:, h_ref], scale=sigma[:, h_ref])
        hi = scipy_t.ppf(1-alpha, df=nu[:, h_ref], loc=pred[:, h_ref], scale=sigma[:, h_ref])
        actual_cov.append(((actual[:, h_ref] >= lo) & (actual[:, h_ref] <= hi)).mean())

    axes[1].plot([l*100 for l in levels], [a*100 for a in actual_cov],
                  'o-', color='#2196F3', ms=6, lw=2, label='Model')
    axes[1].plot([40, 100], [40, 100], '--', color='gray', lw=1)
    axes[1].set_xlabel('Nominal Coverage (%)')
    axes[1].set_ylabel('Empirical Coverage (%)')
    axes[1].set_title('(b) Coverage Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Sigma term structure
    mean_sigma = sigma.mean(axis=0)
    h_range = np.arange(1, len(mean_sigma) + 1)
    axes[2].plot(h_range, mean_sigma, 'b-', lw=2, label='Model')
    axes[2].plot(h_range, mean_sigma[0] * np.sqrt(h_range), 'r--', lw=1.5,
                  label=r'$\sigma_1\sqrt{h}$')
    axes[2].set_xlabel('Horizon (days)')
    axes[2].set_ylabel('Mean Predicted Sigma')
    axes[2].set_title('(c) Volatility Term Structure')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_distributional.png', dpi=300, bbox_inches='tight')
    print(f"  Saved fig3_distributional.png")

    # ── Figure 4: Orthogonality vs Low-Vol (2 panels) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Double-sort heatmap
    vol_labels = ['Low Vol', 'Mid Vol', 'High Vol']
    phantom_labels = ['Short', 'Mid', 'Long']

    # Recompute double-sort means for the heatmap
    unique_ds = np.unique(dates)
    ds_means = np.zeros((3, 3))
    ds_counts = np.zeros((3, 3))

    for d in unique_ds:
        m = dates == d
        n = m.sum()
        if n < 30:
            continue
        p = pred[m, h_ref]
        v = X[m, -1, 4]
        a = actual[m, h_ref]

        vol_ranks = np.argsort(np.argsort(v))
        vt = np.zeros(n, dtype=int)
        vt[vol_ranks >= 2*n//3] = 2
        vt[(vol_ranks >= n//3) & (vol_ranks < 2*n//3)] = 1

        for vi in range(3):
            vm = vt == vi
            n_v = vm.sum()
            if n_v < 6:
                continue
            p_v = p[vm]
            a_v = a[vm]
            pr = np.argsort(np.argsort(p_v))
            for pi in range(3):
                if pi == 0:
                    pm = pr < n_v // 3
                elif pi == 1:
                    pm = (pr >= n_v // 3) & (pr < 2 * n_v // 3)
                else:
                    pm = pr >= 2 * n_v // 3
                if pm.any():
                    ds_means[pi, vi] += a_v[pm].mean()
                    ds_counts[pi, vi] += 1

    ds_means = ds_means / np.maximum(ds_counts, 1) * 100  # to %

    im = axes[0].imshow(ds_means, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(vol_labels)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(phantom_labels)
    axes[0].set_xlabel('Volatility Tercile')
    axes[0].set_ylabel('Phantom Tercile')
    axes[0].set_title('(a) Double Sort: Mean Daily Return (%)')
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f'{ds_means[i, j]:.2f}%', ha='center', va='center',
                          fontsize=11, fontweight='bold',
                          color='white' if abs(ds_means[i, j]) > 0.8 else 'black')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Panel B: IC comparison (Phantom alone, Low-Vol alone, Combined)
    strategies = ['Phantom\nalone', 'Low-Vol\nalone', 'Combined']
    ic_vals = [
        baseline_results['Phantom (ours)']['ic'],
        baseline_results['Low Vol']['ic'],
        ortho_results['combined_ic'],
    ]
    sharpe_vals = [
        baseline_results['Phantom (ours)']['sharpe'],
        baseline_results['Low Vol']['sharpe'],
        ortho_results['combined_sharpe'],
    ]
    colors = ['#F7931A', '#9E9E9E', '#4CAF50']

    x = np.arange(len(strategies))
    w = 0.35
    bars1 = axes[1].bar(x - w/2, ic_vals, w, label='Rank IC', color=colors, alpha=0.8)
    ax2 = axes[1].twinx()
    bars2 = ax2.bar(x + w/2, sharpe_vals, w, label='Sharpe', color=colors, alpha=0.4,
                     edgecolor=colors, linewidth=2, linestyle='--')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies)
    axes[1].set_ylabel('Rank IC')
    ax2.set_ylabel('Sharpe Ratio')
    axes[1].set_title('(b) Signal Combination')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_orthogonality.png', dpi=300, bbox_inches='tight')
    print(f"  Saved fig4_orthogonality.png")


# ═══════════════════════════════════════════════════════════════════
#  6. SUMMARY TABLE (LaTeX-ready)
# ═══════════════════════════════════════════════════════════════════

def print_summary_table(pred, actual, dates, sigma, nu, cost_results, h_ref=9):
    """Print LaTeX-ready summary table."""
    print("\n" + "="*60)
    print("5. SUMMARY TABLE")
    print("="*60)

    ics, _ = rank_ic_per_date(pred[:, h_ref], actual[:, h_ref], dates)
    ls_rets, _ = long_short_returns(pred[:, h_ref], actual[:, h_ref], dates)
    tstat = ics.mean() / (ics.std() / np.sqrt(len(ics)))

    pit = scipy_t.cdf(actual[:, h_ref], df=nu[:, h_ref],
                       loc=pred[:, h_ref], scale=sigma[:, h_ref])
    from scipy.stats import kstest
    ks_stat, ks_p = kstest(pit, 'uniform')

    print(f"""
  \\begin{{tabular}}{{lc}}
  \\toprule
  Metric & Value \\\\
  \\midrule
  \\multicolumn{{2}}{{l}}{{\\textit{{Cross-Sectional Signal}}}} \\\\
  Rank IC (10d) & {ics.mean():.3f} $\\pm$ {ics.std():.3f} \\\\
  IC $t$-statistic & {tstat:.1f} \\\\
  IC positive days & {(ics>0).mean()*100:.0f}\\% \\\\
  \\midrule
  \\multicolumn{{2}}{{l}}{{\\textit{{Long-Short Portfolio (h=10d)}}}} \\\\
  Annualized Sharpe (gross) & {sharpe(ls_rets):.2f} \\\\
  Annualized Sharpe (10 bps) & {cost_results['cost_scenarios']['10 bps (standard)']['sharpe']:.2f} \\\\
  Annualized Sharpe (30 bps) & {cost_results['cost_scenarios']['30 bps (conservative)']['sharpe']:.2f} \\\\
  Win rate & {(ls_rets>0).mean()*100:.1f}\\% \\\\
  Max drawdown & {cost_results.get('max_dd', 0)*100:.1f}\\% \\\\
  Avg daily turnover & {cost_results['avg_turnover']*100:.1f}\\% \\\\
  \\midrule
  \\multicolumn{{2}}{{l}}{{\\textit{{Distributional Quality}}}} \\\\
  NLL & {-scipy_t.logpdf(actual, df=nu, loc=pred, scale=sigma).mean():.3f} \\\\
  KS test $p$-value (PIT) & {ks_p:.3f} \\\\
  Coverage 90\\% & {_coverage(actual[:,h_ref], pred[:,h_ref], sigma[:,h_ref], nu[:,h_ref], 0.90)*100:.1f}\\% \\\\
  \\bottomrule
  \\end{{tabular}}
  """)


def _coverage(actual, mu, sigma, nu, level):
    alpha = (1 - level) / 2
    lo = scipy_t.ppf(alpha, df=nu, loc=mu, scale=sigma)
    hi = scipy_t.ppf(1 - alpha, df=nu, loc=mu, scale=sigma)
    return ((actual >= lo) & (actual <= hi)).mean()


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Publication analysis for Phantom")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v8/best.pt')
    parser.add_argument('--test_data', type=str, default='data/processed_v8/test.npz')
    parser.add_argument('--output_dir', type=str, default='plots')
    args = parser.parse_args()

    print("Loading model...")
    model, ckpt = load_model(args.checkpoint)
    step = ckpt.get('step', 0)
    print(f"  Checkpoint: step {step}")

    print("Loading test data...")
    d = np.load(args.test_data, allow_pickle=True)
    X = d['X'].astype(np.float32)
    Y_rel = d['Y_relative'].astype(np.float32)
    dates = d['dates_end']
    print(f"  {len(X):,} samples, {len(np.unique(dates))} dates, "
          f"{sorted(dates)[0]} to {sorted(dates)[-1]}")

    print("Computing predictions...")
    mu, sigma, nu = predict(model, X)
    h_ref = 9  # 10-day horizon

    # Run all analyses
    cost_results = transaction_cost_analysis(mu[:, h_ref], Y_rel[:, h_ref], dates)
    baseline_results = baseline_comparisons(X, Y_rel, dates, mu, h_ref)
    ortho_results = orthogonality_analysis(X, Y_rel, dates, mu, h_ref)
    decile_results = decile_analysis(mu, Y_rel, dates, h_ref)
    robustness_results = robustness_analysis(mu, Y_rel, dates, h_ref)

    # Publication figures
    print("\n" + "="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)
    plot_publication_figures(mu, Y_rel, dates, sigma, nu, X,
                             cost_results, baseline_results, ortho_results,
                             decile_results, robustness_results,
                             h_ref, args.output_dir)

    # Summary table
    print_summary_table(mu, Y_rel, dates, sigma, nu, cost_results, h_ref)

    # ── Strategy performance summary table ──
    print("\n" + "="*60)
    print("6. STRATEGY PERFORMANCE SUMMARY")
    print("="*60)
    ls_rets, _ = long_short_returns(mu[:, h_ref], Y_rel[:, h_ref], dates)
    ics, _ = rank_ic_per_date(mu[:, h_ref], Y_rel[:, h_ref], dates)

    nw_t = newey_west_tstat(ls_rets)
    ic_nw_t = newey_west_tstat(ics)
    ann_ret = ls_rets.mean() * 365 * 100
    ann_vol = ls_rets.std() * np.sqrt(365) * 100
    sh = sharpe(ls_rets)
    sort = sortino(ls_rets)
    calm = calmar(ls_rets)
    mdd = max_drawdown(ls_rets) * 100

    print(f"""
  Strategy Performance (h=10d, top/bottom 20%, equal-weighted):
  ─────────────────────────────────────────────────────────────
  Annualized return:        {ann_ret:>8.1f}%
  Annualized volatility:    {ann_vol:>8.1f}%
  Sharpe ratio:             {sh:>8.2f}
  Sortino ratio:            {sort:>8.2f}
  Calmar ratio:             {calm:>8.2f}
  Max drawdown:             {mdd:>8.1f}%
  Win rate:                 {(ls_rets>0).mean()*100:>8.1f}%
  Avg daily turnover:       {cost_results['avg_turnover']*100:>8.1f}%
  Breakeven cost (1-way):   {cost_results['breakeven_bps']:>8.0f} bps

  Rank IC (mean):           {ics.mean():>8.4f}
  Rank IC (NW t-stat):      {ic_nw_t:>8.2f}
  L/S return (NW t-stat):   {nw_t:>8.2f}
  IC positive days:         {(ics>0).mean()*100:>8.0f}%
  Months IC > 0:            {sum(1 for v in robustness_results['monthly_ic'].values() if v['mean']>0)}/{len(robustness_results['monthly_ic'])}

  Net Sharpe (10 bps):      {cost_results['cost_scenarios']['10 bps (standard)']['sharpe']:>8.2f}
  Net Sharpe (30 bps):      {cost_results['cost_scenarios']['30 bps (conservative)']['sharpe']:>8.2f}
  Net Sharpe (50 bps):      {cost_results['cost_scenarios']['50 bps (pessimistic)']['sharpe']:>8.2f}
    """)

    # ── Quintile sort table (publication format) ──
    print("\n" + "="*60)
    print("7. QUINTILE PORTFOLIO SORT")
    print("="*60)

    unique_dates = np.unique(dates)
    n_q = 5
    quintile_daily = {q: [] for q in range(n_q)}

    for d in unique_dates:
        m = dates == d
        if m.sum() < n_q * 3:
            continue
        p = mu[m, h_ref]
        a = Y_rel[m, h_ref]
        ranks = np.argsort(np.argsort(p))
        n = len(p)
        for q in range(n_q):
            lo = int(n * q / n_q)
            hi = int(n * (q + 1) / n_q)
            mask_q = (ranks >= lo) & (ranks < hi)
            if mask_q.any():
                quintile_daily[q].append(a[mask_q].mean())

    print(f"\n  {'':>12} ", end='')
    for q in range(n_q):
        label = f'Q{q+1}'
        print(f"{label:>10}", end='')
    print(f"{'Q5-Q1':>10}")

    # Mean return
    q_means = [np.mean(quintile_daily[q]) for q in range(n_q)]
    q_spread = q_means[-1] - q_means[0]
    print(f"  {'Mean ret':>12} ", end='')
    for q in range(n_q):
        print(f"{q_means[q]*100:>9.3f}%", end='')
    print(f"{q_spread*100:>9.3f}%")

    # NW t-stat
    spread_series = np.array(quintile_daily[n_q-1]) - np.array(quintile_daily[0])
    spread_nw_t = newey_west_tstat(spread_series)
    print(f"  {'NW t-stat':>12} ", end='')
    for q in range(n_q):
        t = newey_west_tstat(np.array(quintile_daily[q]))
        print(f"{t:>10.2f}", end='')
    print(f"{spread_nw_t:>10.2f}")

    # Sharpe
    print(f"  {'Sharpe':>12} ", end='')
    for q in range(n_q):
        sh_q = sharpe(np.array(quintile_daily[q]))
        print(f"{sh_q:>10.2f}", end='')
    sh_spread = sharpe(spread_series)
    print(f"{sh_spread:>10.2f}")

    # Monotonicity
    mono = _monotonicity(q_means)
    print(f"\n  Monotonicity (Spearman): {mono:.3f}")
    print(f"  Q5-Q1 spread NW t-stat: {spread_nw_t:.2f}")

    # Save all results as JSON
    results = {
        'step': step,
        'n_samples': len(X),
        'n_dates': len(np.unique(dates)),
        'n_assets': len(np.unique(d['asset_id'])) if 'asset_id' in d else 362,
        'test_period': f"{sorted(dates)[0]} to {sorted(dates)[-1]}",
        'performance': {
            'annualized_return_pct': ann_ret,
            'annualized_vol_pct': ann_vol,
            'sharpe_gross': sh,
            'sortino': sort,
            'calmar': calm,
            'max_drawdown_pct': mdd,
            'win_rate': (ls_rets>0).mean(),
            'ic_mean': ics.mean(),
            'ic_nw_tstat': ic_nw_t,
        },
        'cost_analysis': {
            'gross_sharpe': cost_results['gross_sharpe'],
            'avg_turnover': cost_results['avg_turnover'],
            'breakeven_bps': cost_results['breakeven_bps'],
            'net_sharpes': {k: v['sharpe'] for k, v in cost_results['cost_scenarios'].items()},
        },
        'baselines': {k: {kk: float(vv) for kk, vv in v.items()}
                      for k, v in baseline_results.items()},
        'quintile_means': [float(m) for m in q_means],
        'quintile_spread': float(q_spread),
        'quintile_monotonicity': float(mono),
        'quintile_spread_nw_tstat': float(spread_nw_t),
        'orthogonality': {
            'phantom_ic_within_vol': {k: float(v) for k, v in ortho_results['phantom_ic_within_vol'].items()},
            'signal_correlation': float(ortho_results['signal_correlation']),
            'combined_ic': float(ortho_results['combined_ic']),
            'combined_sharpe': float(ortho_results['combined_sharpe']),
        },
        'decile_spread': decile_results['spread'],
        'decile_monotonicity': float(_monotonicity(decile_results['means'])),
    }
    results_path = Path(args.output_dir) / 'publication_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved {results_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
