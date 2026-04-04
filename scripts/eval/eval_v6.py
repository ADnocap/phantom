#!/usr/bin/env python
"""
Evaluate v6 checkpoint: crypto-focused with 9-channel features.

Same metrics as v5 (for comparison) plus:
  - Feature ablation: zero each of channels 6-8, measure IC change
  - All metrics are crypto-only

Usage:
  python scripts/eval/eval_v6.py --checkpoint checkpoints_v6/best.pt
  python scripts/eval/eval_v6.py --checkpoint checkpoints_v6/best.pt --test_data data/processed_v6/test.npz
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


def long_short_backtest(pred, actual, dates, quantile=0.2, min_assets=10):
    """Long top quintile, short bottom quintile per date."""
    unique_dates = np.unique(dates)
    daily_returns = []
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
        daily_returns.append(a[top_idx].mean() - a[bot_idx].mean())
    return np.array(daily_returns)


def predict_batched(model, X, batch_size=1024):
    """Get model predictions in batches."""
    all_mu, all_sigma, all_nu = [], [], []
    N = len(X)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.from_numpy(X[i:i+batch_size])
            log_pi, mu, sigma, nu = model(x)
            all_mu.append(mu.numpy())
            all_sigma.append(sigma.numpy())
            if nu is not None:
                all_nu.append(nu.numpy())
    mu_np = np.concatenate(all_mu).squeeze(-1)
    sigma_np = np.concatenate(all_sigma).squeeze(-1)
    nu_np = np.concatenate(all_nu).squeeze(-1) if all_nu else None
    return mu_np, sigma_np, nu_np


def feature_ablation(model, X, Y_rel, dates, h_ref=9, channels=(6, 7)):
    """Zero each channel and measure IC change."""
    channel_names = {6: 'taker_buy_ratio', 7: 'funding_rate'}

    # Baseline IC
    mu_base, _, _ = predict_batched(model, X)
    ics_base = compute_rank_ic(mu_base[:, h_ref], Y_rel[:, h_ref], dates)
    ic_base = ics_base.mean()

    results = {'baseline': ic_base}
    print(f"\nFeature Ablation (h={h_ref+1}d):")
    print(f"  Baseline IC: {ic_base:.4f}")

    for ch in channels:
        X_ablated = X.copy()
        X_ablated[:, :, ch] = 0.0

        mu_abl, _, _ = predict_batched(model, X_ablated)
        ics_abl = compute_rank_ic(mu_abl[:, h_ref], Y_rel[:, h_ref], dates)
        ic_abl = ics_abl.mean()
        delta = ic_base - ic_abl
        name = channel_names.get(ch, f'ch{ch}')
        results[name] = {'ic_without': ic_abl, 'ic_delta': delta}
        print(f"  Without {name}: IC={ic_abl:.4f} (delta={delta:+.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate v6 (crypto + new features)")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v6/best.pt')
    parser.add_argument('--test_data', type=str, default='data/processed_v6/test.npz')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--ablation', action='store_true', default=True,
                        help='Run feature ablation analysis')
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = PhantomConfig(**{k: v for k, v in cfg_dict.items()
                          if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    step = ckpt.get('step', 0)
    phase = ckpt.get('phase', '?')
    print(f"Checkpoint: step {step:,}, phase {phase}, best_val_loss: {ckpt.get('best_val_loss', '?')}")
    print(f"Input channels: {cfg.n_input_channels}")

    # Load test data
    d = np.load(args.test_data, allow_pickle=True)
    X = d['X'].astype(np.float32)
    Y_rel = d['Y_relative'].astype(np.float32)
    Y_abs = d['Y'].astype(np.float32)
    dates = d['dates_end'] if 'dates_end' in d else None

    if args.n_samples and args.n_samples < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), args.n_samples, replace=False)
        X, Y_rel, Y_abs = X[idx], Y_rel[idx], Y_abs[idx]
        if dates is not None:
            dates = dates[idx]

    N = len(X)
    print(f"Test set: {N:,} samples, X shape: {X.shape}")
    if dates is not None:
        print(f"  Unique dates: {len(np.unique(dates))}")
        print(f"  Date range: {sorted(dates)[0]} to {sorted(dates)[-1]}")

    # New channel activity in test set
    for ch, name in [(6, 'taker_buy_ratio'), (7, 'funding_rate'), (8, 'oi_change')]:
        if ch < X.shape[2]:
            nonzero = (X[:, :, ch] != 0).any(axis=1).mean()
            print(f"  Channel {ch} ({name}): {nonzero*100:.1f}% samples have data")

    # Get predictions
    print("Computing predictions...", flush=True)
    mu_np, sigma_np, nu_np = predict_batched(model, X)
    pred_mean = mu_np  # (N, 30), Student-t K=1

    print(f"Pred mean: range [{pred_mean.min():.4f}, {pred_mean.max():.4f}], std={pred_mean.std():.4f}")
    print(f"Sigma: range [{sigma_np.min():.4f}, {sigma_np.max():.4f}]")
    if nu_np is not None:
        print(f"Nu: range [{nu_np.min():.2f}, {nu_np.max():.2f}], mean={nu_np.mean():.2f}")

    # NLL
    h_ref = 9
    nll_val = -scipy_t.logpdf(Y_rel, df=nu_np, loc=mu_np, scale=sigma_np).mean()
    print(f"NLL: {nll_val:.4f}")

    # PIT
    pit_values = scipy_t.cdf(Y_rel[:, h_ref], df=nu_np[:, h_ref],
                              loc=mu_np[:, h_ref], scale=sigma_np[:, h_ref])

    # Coverage
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

    # Cross-sectional metrics
    if dates is not None:
        horizons_to_eval = [0, 4, 9, 14, 19, 24, 29]
        rank_ics_by_h = {}
        print("\nRank IC by horizon:")
        for h_idx in horizons_to_eval:
            ics = compute_rank_ic(pred_mean[:, h_idx], Y_rel[:, h_idx], dates)
            rank_ics_by_h[h_idx+1] = ics
            tstat = ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8)
            print(f"  h={h_idx+1:2d}d: IC={ics.mean():.4f} +/- {ics.std():.4f} "
                  f"(t={tstat:.1f}, n={len(ics)})")

        # Long-short backtest
        ls_returns = long_short_backtest(pred_mean[:, h_ref], Y_rel[:, h_ref], dates)
        if len(ls_returns) > 0:
            sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8) * np.sqrt(252)
            cum_ret = np.cumsum(ls_returns)
            drawdowns = cum_ret - np.maximum.accumulate(cum_ret)
            max_dd = drawdowns.min()
            print(f"\nLong-short (h=10d):")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Cumulative: {cum_ret[-1]*100:.1f}%")
            print(f"  Win rate: {(ls_returns > 0).mean()*100:.0f}%")
            print(f"  Max drawdown: {max_dd*100:.1f}%")
    else:
        rank_ics_by_h = {}
        ls_returns = np.array([])

    # Conditional signal
    corr_mean = np.corrcoef(pred_mean[:, h_ref], Y_rel[:, h_ref])[0, 1]
    corr_std = np.corrcoef(sigma_np[:, h_ref],
                            np.abs(Y_rel[:, h_ref] - pred_mean[:, h_ref]))[0, 1]
    print(f"\nConditional signal (h=10d):")
    print(f"  Pred mean std: {pred_mean[:, h_ref].std():.4f}")
    print(f"  Corr(pred_mean, actual): {corr_mean:.4f}")
    print(f"  Corr(pred_std, |error|): {corr_std:.4f}")

    # Feature ablation
    ablation_results = None
    if args.ablation and dates is not None and X.shape[2] >= 8:
        ablation_results = feature_ablation(model, X, Y_rel, dates, h_ref=h_ref)

    # ═══════════ PLOT ═══════════
    fig = plt.figure(figsize=(24, 24))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: IC distribution, IC by horizon, long-short cumulative
    ax = fig.add_subplot(gs[0, 0])
    if h_ref+1 in rank_ics_by_h and len(rank_ics_by_h[h_ref+1]) > 0:
        ics = rank_ics_by_h[h_ref+1]
        ax.hist(ics, bins=30, alpha=0.7, color='#F7931A', edgecolor='white')
        ax.axvline(ics.mean(), color='red', ls='--', lw=2, label=f'Mean: {ics.mean():.4f}')
        ax.axvline(0, color='gray', ls='--')
        ax.legend()
    ax.set_xlabel('Rank IC'); ax.set_ylabel('Count')
    ax.set_title('Rank IC Distribution (h=10d, crypto)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    if rank_ics_by_h:
        h_vals = sorted(rank_ics_by_h.keys())
        ic_means = [rank_ics_by_h[h].mean() for h in h_vals]
        ic_stds = [rank_ics_by_h[h].std() for h in h_vals]
        ax.errorbar(h_vals, ic_means, yerr=ic_stds, fmt='o-', color='#F7931A',
                    capsize=3, lw=2, ms=6)
        ax.axhline(0, color='gray', ls='--')
        ax.fill_between(h_vals, [m-s for m,s in zip(ic_means,ic_stds)],
                        [m+s for m,s in zip(ic_means,ic_stds)], alpha=0.2, color='#F7931A')
    ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Rank IC')
    ax.set_title('Rank IC by Horizon (crypto)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if len(ls_returns) > 0:
        cum_ret = np.cumsum(ls_returns) * 100
        ax.plot(cum_ret, color='#F7931A', lw=2)
        ax.axhline(0, color='gray', ls='--')
        ax.fill_between(range(len(cum_ret)), 0, cum_ret,
                        where=cum_ret >= 0, alpha=0.3, color='green')
        ax.fill_between(range(len(cum_ret)), 0, cum_ret,
                        where=cum_ret < 0, alpha=0.3, color='red')
        sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8) * np.sqrt(252)
        ax.set_title(f'Long-Short Cumulative (Sharpe={sharpe:.2f})')
    ax.set_xlabel('Trading days'); ax.set_ylabel('Cumulative return (%)')
    ax.grid(True, alpha=0.3)

    # Row 2: PIT, coverage, pred vs actual
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(pit_values, bins=30, density=True, alpha=0.7, color='#F7931A', edgecolor='white')
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Ideal')
    ax.set_xlabel('PIT value'); ax.set_ylabel('Density')
    ax.set_title('PIT Histogram (h=10d)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    ax = fig.add_subplot(gs[1, 1])
    levels_pct = [l*100 for l in coverage_levels]
    actual_pct = [coverages[l]*100 for l in coverage_levels]
    ax.plot(levels_pct, actual_pct, 'o-', color='#F7931A', ms=8, lw=2, label='Model')
    ax.plot([40, 100], [40, 100], '--', color='gray', label='Perfect')
    for t, a in zip(levels_pct, actual_pct):
        ax.annotate(f'{a:.1f}%', (t, a), textcoords="offset points", xytext=(8, -5))
    ax.set_xlabel('Target (%)'); ax.set_ylabel('Actual (%)')
    ax.set_title('Coverage Calibration (h=10d)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 100); ax.set_ylim(40, 100)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(Y_rel[:, h_ref][::3], pred_mean[:, h_ref][::3],
               alpha=0.1, s=3, color='#F7931A')
    lim = np.percentile(np.abs(Y_rel[:, h_ref]), 99) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlabel('Actual relative return'); ax.set_ylabel('Predicted mean')
    ax.set_title(f'Pred vs Actual (h=10d, corr={corr_mean:.3f})')
    ax.grid(True, alpha=0.3)

    # Row 3: Sigma term structure, mu term structure, daily L/S returns
    ax = fig.add_subplot(gs[2, 0])
    mean_sigma_by_h = sigma_np.mean(axis=0)
    h_range = np.arange(1, 31)
    ax.plot(h_range, mean_sigma_by_h, '-', color='#F7931A', lw=2, label='Model sigma')
    theoretical = mean_sigma_by_h[0] * np.sqrt(h_range)
    ax.plot(h_range, theoretical, 'r--', lw=1.5, label=r'$\sigma_1 \sqrt{h}$')
    ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Mean sigma')
    ax.set_title('Sigma Term Structure')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    mean_mu = pred_mean.mean(axis=0)
    std_mu = pred_mean.std(axis=0)
    ax.plot(h_range, mean_mu, '-', color='#F7931A', lw=2, label='Mean of pred mu')
    ax.fill_between(h_range, mean_mu - std_mu, mean_mu + std_mu, alpha=0.3, color='#F7931A')
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Horizon (days)'); ax.set_ylabel('Predicted mu')
    ax.set_title('Mu Term Structure (spread = signal)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    if len(ls_returns) > 0:
        ax.bar(range(len(ls_returns)), ls_returns * 100,
               color=['green' if r > 0 else 'red' for r in ls_returns], alpha=0.6)
        ax.axhline(0, color='gray', ls='--')
        ax.set_xlabel('Trading day'); ax.set_ylabel('Daily L/S return (%)')
        ax.set_title(f'Daily Returns (win rate={100*(ls_returns>0).mean():.0f}%)')
    ax.grid(True, alpha=0.3)

    # Row 4: Feature ablation + IC over time
    ax = fig.add_subplot(gs[3, 0])
    if ablation_results:
        names = ['baseline']
        vals = [ablation_results['baseline']]
        colors = ['#F7931A']
        for ch_name in ['taker_buy_ratio', 'funding_rate']:
            if ch_name in ablation_results:
                names.append(f'no {ch_name}')
                vals.append(ablation_results[ch_name]['ic_without'])
                colors.append('#888888')
        bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{v:.4f}', ha='center', fontsize=9)
    ax.set_ylabel('Rank IC (10d)')
    ax.set_title('Feature Ablation')
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[3, 1])
    if h_ref+1 in rank_ics_by_h:
        ics = rank_ics_by_h[h_ref+1]
        ax.bar(range(len(ics)), ics,
               color=['green' if x > 0 else 'red' for x in ics], alpha=0.5)
        # Rolling mean
        if len(ics) >= 20:
            rolling = np.convolve(ics, np.ones(20)/20, mode='valid')
            ax.plot(range(19, 19+len(rolling)), rolling, 'b-', lw=2, label='20-day rolling')
        ax.axhline(0, color='gray', ls='--')
        ax.axhline(ics.mean(), color='red', ls='--', alpha=0.5, label=f'Mean: {ics.mean():.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Trading day'); ax.set_ylabel('Daily Rank IC')
    ax.set_title('IC Over Time (h=10d)')
    ax.grid(True, alpha=0.3)

    # Row 4, col 3: Summary
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    lines = [
        f"Checkpoint: step {step:,} (phase {phase})",
        f"Test samples: {N:,}",
        f"Input channels: {cfg.n_input_channels}",
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

    if h_ref+1 in rank_ics_by_h:
        ics = rank_ics_by_h[h_ref+1]
        tstat = ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8)
        lines += ["", "--- Cross-Sectional ---",
                   f"Rank IC (10d): {ics.mean():.4f}",
                   f"IC t-stat: {tstat:.2f}"]
    if len(ls_returns) > 0:
        sharpe = ls_returns.mean() / (ls_returns.std()+1e-8) * np.sqrt(252)
        lines += [f"L/S Sharpe: {sharpe:.2f}",
                  f"L/S cumul: {np.sum(ls_returns)*100:.1f}%",
                  f"Win rate: {100*(ls_returns>0).mean():.0f}%"]
    if ablation_results:
        lines += ["", "--- Feature Ablation ---"]
        for ch_name in ['taker_buy_ratio', 'funding_rate']:
            if ch_name in ablation_results:
                d = ablation_results[ch_name]['ic_delta']
                lines.append(f"  {ch_name}: {d:+.4f}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Row 5: v5 vs v6 comparison placeholder (empty for now)
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')
    ax.text(0.5, 0.5,
            'v6 vs v5 comparison: run eval_v5.py on same test period for side-by-side',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color='gray', style='italic')

    fig.suptitle(f'Phantom v6 — Crypto Evaluation (step {step:,})', fontsize=16, y=0.99)
    out_path = args.output or f'plots/eval_v6_{step}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
