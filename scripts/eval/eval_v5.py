#!/usr/bin/env python
"""
Evaluate v5 checkpoint: cross-sectional relative return prediction.

Key metrics:
  - Rank IC (Spearman): does the model rank assets correctly per date?
  - Pearson IC: linear correlation between predicted and actual relative returns
  - Long-short portfolio: buy top quintile, short bottom quintile
  - Per-horizon IC: which horizons have signal?
  - Per-asset-class IC: where is signal concentrated?
  - Standard distributional metrics: PIT, coverage, CRPS

Usage:
  python scripts/eval/eval_v5.py --checkpoint checkpoints_v5/best.pt
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr, t as scipy_t

from src.model import PhantomConfig, PhantomModel


ASSET_NAMES = {0: 'Crypto', 1: 'Equity', 2: 'Forex', 3: 'Commodity'}
ASSET_COLORS = {0: '#F7931A', 1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}


def compute_rank_ic(pred, actual, dates, min_assets=5):
    """Spearman rank IC averaged across dates."""
    unique_dates = np.unique(dates)
    ics = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < min_assets:
            continue
        ic, _ = spearmanr(pred[mask], actual[mask])
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def compute_pearson_ic(pred, actual, dates, min_assets=5):
    """Pearson IC averaged across dates."""
    unique_dates = np.unique(dates)
    ics = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < min_assets:
            continue
        ic, _ = pearsonr(pred[mask], actual[mask])
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def long_short_backtest(pred, actual, dates, quantile=0.2, min_assets=10):
    """Long top quintile, short bottom quintile per date."""
    unique_dates = np.unique(dates)
    daily_returns = []
    valid_dates = []
    for d in unique_dates:
        mask = dates == d
        n = mask.sum()
        if n < min_assets:
            continue
        p = pred[mask]
        a = actual[mask]
        k = max(1, int(n * quantile))
        top_idx = np.argsort(p)[-k:]
        bot_idx = np.argsort(p)[:k]
        ls_ret = a[top_idx].mean() - a[bot_idx].mean()
        daily_returns.append(ls_ret)
        valid_dates.append(d)
    return np.array(daily_returns), np.array(valid_dates)


def main():
    parser = argparse.ArgumentParser(description="Evaluate v5 (relative returns)")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v5/best.pt')
    parser.add_argument('--test_data', type=str, default='data/processed_v5/test.npz')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = PhantomConfig(**{k: v for k, v in cfg_dict.items()
                          if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    step = ckpt['step']
    print(f"Checkpoint: step {step:,}, best_val_loss: {ckpt['best_val_loss']:.4f}")

    # Load test data
    d = np.load(args.test_data, allow_pickle=True)
    X = d['X'].astype(np.float32)
    Y_rel = d['Y_relative'].astype(np.float32)
    Y_abs = d['Y'].astype(np.float32)
    asset_type = d['asset_type'].astype(np.int64)
    dates = d['dates_end'] if 'dates_end' in d else None

    if args.n_samples and args.n_samples < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), args.n_samples, replace=False)
        X, Y_rel, Y_abs, asset_type = X[idx], Y_rel[idx], Y_abs[idx], asset_type[idx]
        if dates is not None:
            dates = dates[idx]

    N = len(X)
    print(f"Test set: {N:,} samples")
    for t in sorted(np.unique(asset_type)):
        n = (asset_type == t).sum()
        print(f"  {ASSET_NAMES.get(t, f'type_{t}')}: {n} ({100*n/N:.1f}%)")

    if dates is not None:
        print(f"  Unique dates: {len(np.unique(dates))}")

    # Get predictions (batched)
    print("Computing predictions...", flush=True)
    all_mu, all_sigma, all_nu = [], [], []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, N, batch_size):
            print(f"\r  Batch {i//batch_size+1}/{(N+batch_size-1)//batch_size}", end='', flush=True)
            x = torch.from_numpy(X[i:i+batch_size])
            log_pi, mu, sigma, nu = model(x)
            all_mu.append(mu.numpy())
            all_sigma.append(sigma.numpy())
            if nu is not None:
                all_nu.append(nu.numpy())
    print()

    mu_np = np.concatenate(all_mu).squeeze(-1)      # (N, 30)
    sigma_np = np.concatenate(all_sigma).squeeze(-1)
    nu_np = np.concatenate(all_nu).squeeze(-1) if all_nu else None

    # Predicted mean per horizon (K=1 Student-t, so mu IS the mean)
    pred_mean = mu_np  # (N, 30)

    print(f"Pred mean: range [{pred_mean.min():.4f}, {pred_mean.max():.4f}], std={pred_mean.std():.4f}")
    print(f"Sigma: range [{sigma_np.min():.4f}, {sigma_np.max():.4f}]")

    # ── NLL and CRPS (vectorized via scipy) ──
    print("Computing losses...", flush=True)
    nll_per_sample = -scipy_t.logpdf(Y_rel, df=nu_np, loc=mu_np, scale=sigma_np).mean(axis=1)
    nll_val = nll_per_sample.mean()
    print(f"NLL: {nll_val:.4f}")

    # ── PIT values (on a reference horizon, e.g. h=10) ──
    h_ref = 9  # index for 10-day horizon
    pit_values = scipy_t.cdf(Y_rel[:, h_ref], df=nu_np[:, h_ref],
                              loc=mu_np[:, h_ref], scale=sigma_np[:, h_ref])

    # ── Coverage ──
    coverage_levels = [0.50, 0.80, 0.90, 0.95]
    coverages = {}
    for level in coverage_levels:
        alpha = (1 - level) / 2
        lo = scipy_t.ppf(alpha, df=nu_np[:, h_ref], loc=mu_np[:, h_ref], scale=sigma_np[:, h_ref])
        hi = scipy_t.ppf(1-alpha, df=nu_np[:, h_ref], loc=mu_np[:, h_ref], scale=sigma_np[:, h_ref])
        coverages[level] = ((Y_rel[:, h_ref] >= lo) & (Y_rel[:, h_ref] <= hi)).mean()

    print(f"\nCoverage (h=10d):")
    for level in coverage_levels:
        print(f"  {level*100:.0f}%: {coverages[level]*100:.1f}%")

    # ── Cross-sectional metrics (need dates) ──
    if dates is None:
        print("\nWARNING: No dates in test data — cannot compute Rank IC / long-short")
        rank_ics_by_h = {}
        ls_returns = np.array([])
        ls_dates = np.array([])
    else:
        # Rank IC per horizon
        print("\nRank IC by horizon:")
        horizons_to_eval = [0, 4, 9, 14, 19, 24, 29]  # 1d, 5d, 10d, 15d, 20d, 25d, 30d
        rank_ics_by_h = {}
        for h_idx in horizons_to_eval:
            ics = compute_rank_ic(pred_mean[:, h_idx], Y_rel[:, h_idx], dates)
            rank_ics_by_h[h_idx+1] = ics
            print(f"  h={h_idx+1:2d}d: IC={ics.mean():.4f} +/- {ics.std():.4f} (n={len(ics)} dates)")

        # Rank IC by asset class (at h=10d)
        print(f"\nRank IC by asset class (h=10d):")
        for t in sorted(np.unique(asset_type)):
            mask = asset_type == t
            ics = compute_rank_ic(pred_mean[mask, h_ref], Y_rel[mask, h_ref], dates[mask])
            name = ASSET_NAMES.get(t, f'type_{t}')
            print(f"  {name}: IC={ics.mean():.4f} +/- {ics.std():.4f} (n={len(ics)})")

        # Long-short backtest (h=10d)
        ls_returns, ls_dates = long_short_backtest(
            pred_mean[:, h_ref], Y_rel[:, h_ref], dates)
        if len(ls_returns) > 0:
            cum_ret = np.cumsum(ls_returns)
            sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8) * np.sqrt(252)
            print(f"\nLong-short (h=10d, top/bottom 20%):")
            print(f"  Mean daily: {ls_returns.mean()*100:.3f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Cumulative: {cum_ret[-1]*100:.1f}%")
            print(f"  Win rate: {(ls_returns > 0).mean()*100:.1f}%")

    # ── Conditional signal ──
    corr_mean = np.corrcoef(pred_mean[:, h_ref], Y_rel[:, h_ref])[0, 1]
    corr_std = np.corrcoef(sigma_np[:, h_ref], np.abs(Y_rel[:, h_ref] - pred_mean[:, h_ref]))[0, 1]
    print(f"\nConditional signal (h=10d):")
    print(f"  Pred mean std: {pred_mean[:, h_ref].std():.4f}")
    print(f"  Corr(pred_mean, actual): {corr_mean:.4f}")
    print(f"  Corr(pred_std, |error|): {corr_std:.4f}")

    # ═══════════ PLOT ═══════════
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Rank IC distribution, IC time series, long-short cumulative
    ax = fig.add_subplot(gs[0, 0])
    if h_ref+1 in rank_ics_by_h and len(rank_ics_by_h[h_ref+1]) > 0:
        ics = rank_ics_by_h[h_ref+1]
        ax.hist(ics, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(ics.mean(), color='red', ls='--', lw=2, label=f'Mean: {ics.mean():.4f}')
        ax.axvline(0, color='gray', ls='--', lw=1)
        ax.legend()
    ax.set_xlabel('Rank IC'); ax.set_ylabel('Count')
    ax.set_title(f'Rank IC Distribution (h=10d)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    if len(rank_ics_by_h) > 0:
        h_vals = sorted(rank_ics_by_h.keys())
        ic_means = [rank_ics_by_h[h].mean() for h in h_vals]
        ic_stds = [rank_ics_by_h[h].std() for h in h_vals]
        ax.errorbar(h_vals, ic_means, yerr=ic_stds, fmt='o-', color='tab:blue',
                    capsize=3, lw=2, ms=6)
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.fill_between(h_vals, [m-s for m,s in zip(ic_means,ic_stds)],
                        [m+s for m,s in zip(ic_means,ic_stds)], alpha=0.2)
    ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Rank IC')
    ax.set_title('Rank IC by Horizon')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if len(ls_returns) > 0:
        cum_ret = np.cumsum(ls_returns) * 100
        ax.plot(range(len(cum_ret)), cum_ret, color='#2196F3', lw=2)
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.fill_between(range(len(cum_ret)), 0, cum_ret,
                        where=cum_ret >= 0, alpha=0.3, color='green')
        ax.fill_between(range(len(cum_ret)), 0, cum_ret,
                        where=cum_ret < 0, alpha=0.3, color='red')
        sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8) * np.sqrt(252)
        ax.set_title(f'Long-Short Cumulative (Sharpe={sharpe:.2f})')
    else:
        ax.set_title('Long-Short (no dates available)')
    ax.set_xlabel('Trading days'); ax.set_ylabel('Cumulative return (%)')
    ax.grid(True, alpha=0.3)

    # Row 2: PIT, coverage, CRPS by asset type
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(pit_values, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Ideal')
    ax.set_xlabel('PIT value'); ax.set_ylabel('Density')
    ax.set_title('PIT Histogram (h=10d, relative returns)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    ax = fig.add_subplot(gs[1, 1])
    levels_pct = [l*100 for l in coverage_levels]
    actual_pct = [coverages[l]*100 for l in coverage_levels]
    ax.plot(levels_pct, actual_pct, 'o-', color='tab:blue', ms=8, lw=2, label='Model')
    ax.plot([40, 100], [40, 100], '--', color='gray', lw=1, label='Perfect')
    for t, a in zip(levels_pct, actual_pct):
        ax.annotate(f'{a:.1f}%', (t, a), textcoords="offset points", xytext=(8, -5), fontsize=9)
    ax.set_xlabel('Target (%)'); ax.set_ylabel('Actual (%)')
    ax.set_title('Coverage Calibration (h=10d)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 100); ax.set_ylim(40, 100)

    ax = fig.add_subplot(gs[1, 2])
    # CRPS by asset type (sample-based)
    rng = np.random.default_rng(42)
    n_mc = 300
    crps_by_asset = {}
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        samples = scipy_t.rvs(df=nu_np[mask, h_ref], loc=mu_np[mask, h_ref],
                               scale=sigma_np[mask, h_ref],
                               size=(n_mc, mask.sum()), random_state=rng).T
        term1 = np.abs(samples - Y_rel[mask, h_ref:h_ref+1]).mean(axis=1)
        s_sorted = np.sort(samples, axis=1)
        diffs = s_sorted[:, 1:] - s_sorted[:, :-1]
        idx_w = np.arange(1, n_mc)
        weights = idx_w * (n_mc - idx_w)
        term2 = (diffs * weights[None, :]).sum(axis=1) * 2.0 / (n_mc * n_mc)
        crps_by_asset[ASSET_NAMES.get(t, f'{t}')] = (term1 - 0.5 * term2).mean()

    names = list(crps_by_asset.keys())
    vals = [crps_by_asset[n] for n in names]
    bar_colors = [ASSET_COLORS.get(i, 'gray') for i in sorted(np.unique(asset_type))]
    bars = ax.bar(names, vals, color=bar_colors, alpha=0.8, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f'{v:.4f}', ha='center', fontsize=10)
    ax.set_ylabel('CRPS'); ax.set_title('CRPS by Asset Type (h=10d)')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3: pred mean vs actual, sigma term structure, IC by asset class
    ax = fig.add_subplot(gs[2, 0])
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        ax.scatter(Y_rel[mask, h_ref][::3], pred_mean[mask, h_ref][::3],
                   alpha=0.1, s=3, color=ASSET_COLORS.get(t, 'gray'),
                   label=ASSET_NAMES.get(t, f'{t}'))
    lim = np.percentile(np.abs(Y_rel[:, h_ref]), 99) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlabel('Actual relative return'); ax.set_ylabel('Predicted mean')
    ax.set_title(f'Pred vs Actual (h=10d, corr={corr_mean:.3f})')
    ax.legend(markerscale=5, fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    ax = fig.add_subplot(gs[2, 1])
    # Sigma term structure
    mean_sigma_by_h = sigma_np.mean(axis=0)  # (30,)
    h_range = np.arange(1, 31)
    ax.plot(h_range, mean_sigma_by_h, 'b-', lw=2, label='Model sigma')
    # Theoretical sqrt(h) scaling
    theoretical = mean_sigma_by_h[0] * np.sqrt(h_range)
    ax.plot(h_range, theoretical, 'r--', lw=1.5, label=r'$\sigma_1 \sqrt{h}$')
    ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Mean sigma')
    ax.set_title('Sigma Term Structure')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    # Mu term structure
    mean_mu_by_h = pred_mean.mean(axis=0)  # (30,)
    std_mu_by_h = pred_mean.std(axis=0)
    ax.plot(h_range, mean_mu_by_h, 'b-', lw=2, label='Mean of pred mu')
    ax.fill_between(h_range, mean_mu_by_h - std_mu_by_h, mean_mu_by_h + std_mu_by_h,
                    alpha=0.3, label='+/- 1 std')
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Predicted mu')
    ax.set_title('Mu Term Structure (spread = signal strength)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Row 4: IC by asset class, long-short daily, summary
    ax = fig.add_subplot(gs[3, 0])
    if dates is not None:
        ic_by_class = {}
        for t in sorted(np.unique(asset_type)):
            mask = asset_type == t
            ics = compute_rank_ic(pred_mean[mask, h_ref], Y_rel[mask, h_ref], dates[mask])
            ic_by_class[ASSET_NAMES.get(t, f'{t}')] = ics
        names_c = list(ic_by_class.keys())
        means_c = [ic_by_class[n].mean() for n in names_c]
        stds_c = [ic_by_class[n].std() for n in names_c]
        bars = ax.bar(names_c, means_c, yerr=stds_c, capsize=5,
                      color=[ASSET_COLORS.get(i, 'gray') for i in sorted(np.unique(asset_type))],
                      alpha=0.8, edgecolor='white')
        for bar, m in zip(bars, means_c):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{m:.4f}', ha='center', fontsize=10)
        ax.axhline(0, color='gray', ls='--')
    ax.set_ylabel('Rank IC'); ax.set_title('Rank IC by Asset Class (h=10d)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[3, 1])
    if len(ls_returns) > 0:
        ax.bar(range(len(ls_returns)), ls_returns * 100,
               color=['green' if r > 0 else 'red' for r in ls_returns], alpha=0.6)
        ax.axhline(0, color='gray', ls='--')
        ax.set_xlabel('Trading day'); ax.set_ylabel('Daily L/S return (%)')
        ax.set_title(f'Long-Short Daily Returns (win rate={100*(ls_returns>0).mean():.0f}%)')
    ax.grid(True, alpha=0.3)

    # Summary
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    lines = [
        f"Checkpoint: step {step:,}",
        f"Test samples: {N:,}",
        "",
        "--- Distributional (h=10d) ---",
        f"NLL:   {nll_val:.4f}",
        f"Pred mean std: {pred_mean[:, h_ref].std():.4f}",
        f"Corr(mean, y): {corr_mean:.4f}",
        f"Corr(std, |err|): {corr_std:.4f}",
        "",
        "--- Coverage (h=10d) ---",
    ]
    for level in coverage_levels:
        lines.append(f"  {level*100:.0f}%: {coverages[level]*100:.1f}%")

    if dates is not None and h_ref+1 in rank_ics_by_h:
        ics = rank_ics_by_h[h_ref+1]
        lines += [
            "",
            "--- Cross-Sectional ---",
            f"Rank IC (10d): {ics.mean():.4f}",
            f"IC t-stat: {ics.mean()/(ics.std()/np.sqrt(len(ics))+1e-8):.2f}",
        ]
        if len(ls_returns) > 0:
            sharpe = ls_returns.mean() / (ls_returns.std()+1e-8) * np.sqrt(252)
            lines += [
                f"L/S Sharpe: {sharpe:.2f}",
                f"L/S cumul: {np.sum(ls_returns)*100:.1f}%",
                f"Win rate: {100*(ls_returns>0).mean():.0f}%",
            ]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle(f'Phantom v5 — Relative Return Evaluation (step {step:,})',
                 fontsize=16, y=0.99)
    out_path = args.output or f'plots/eval_v5_{step}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
