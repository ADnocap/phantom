#!/usr/bin/env python
"""
Evaluate v7 checkpoint: 4h-bar crypto predictions.

Adapts v6 eval for 4h horizons:
  - Rank IC per daily timestamp (use 00:00 UTC bars)
  - Display horizons: 6 bars (1d), 18 (3d), 42 (7d), 60 (10d), 90 (15d)
  - Long-short backtest: daily trades using 00:00 UTC bars

Usage:
  python scripts/eval/eval_v7.py --checkpoint checkpoints_v7/best.pt
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

from src.model import PhantomConfig, PhantomModel


# 4h horizons to evaluate (bars → days)
HORIZONS = {
    6: '1d', 12: '2d', 18: '3d', 30: '5d',
    42: '7d', 60: '10d', 90: '15d',
}
H_REF = 60  # 10-day horizon (index 59 in 0-based)
H_REF_IDX = H_REF - 1


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
    unique_dates = np.unique(dates)
    daily_returns = []
    for d in unique_dates:
        mask = dates == d
        n = mask.sum()
        if n < min_assets:
            continue
        p, a = pred[mask], actual[mask]
        k = max(1, int(n * quantile))
        top_idx = np.argsort(p)[-k:]
        bot_idx = np.argsort(p)[:k]
        daily_returns.append(a[top_idx].mean() - a[bot_idx].mean())
    return np.array(daily_returns)


def predict_batched(model, X, batch_size=512):
    all_mu, all_sigma, all_nu = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate v7 (4h bars)")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v7/best.pt')
    parser.add_argument('--test_data', type=str, default='data/processed_v7/test.npz')
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
    step = ckpt.get('step', 0)
    print(f"Checkpoint: step {step:,}, val_loss: {ckpt.get('best_val_loss', '?'):.4f}")
    print(f"Config: context={cfg.context_len}, patch_len={cfg.patch_len}, "
          f"horizons={cfg.max_horizon}, channels={cfg.n_input_channels}")

    # Load test data
    d = np.load(args.test_data, allow_pickle=True)
    X = d['X'].astype(np.float32)
    Y_rel = d['Y_relative'].astype(np.float32)
    dates = d['dates_end']  # daily dates for IC grouping
    timestamps = d['timestamps_end'] if 'timestamps_end' in d else None

    if args.n_samples and args.n_samples < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), args.n_samples, replace=False)
        X, Y_rel, dates = X[idx], Y_rel[idx], dates[idx]
        if timestamps is not None:
            timestamps = timestamps[idx]

    N = len(X)
    print(f"Test set: {N:,} samples, X: {X.shape}")
    print(f"Date range: {sorted(dates)[0]} to {sorted(dates)[-1]}")
    print(f"Unique dates: {len(np.unique(dates))}")

    # Predictions
    print("Computing predictions...", flush=True)
    mu_np, sigma_np, nu_np = predict_batched(model, X)
    pred_mean = mu_np

    print(f"Pred mean: range [{pred_mean.min():.4f}, {pred_mean.max():.4f}], std={pred_mean.std():.4f}")
    print(f"Sigma: range [{sigma_np.min():.4f}, {sigma_np.max():.4f}]")
    if nu_np is not None:
        print(f"Nu: range [{nu_np.min():.2f}, {nu_np.max():.2f}], mean={nu_np.mean():.2f}")

    # NLL (average across all horizons)
    nll_val = -scipy_t.logpdf(Y_rel, df=nu_np, loc=mu_np, scale=sigma_np).mean()
    print(f"NLL: {nll_val:.4f}")

    # Coverage at reference horizon
    coverage_levels = [0.50, 0.80, 0.90, 0.95]
    coverages = {}
    for level in coverage_levels:
        alpha = (1 - level) / 2
        lo = scipy_t.ppf(alpha, df=nu_np[:, H_REF_IDX], loc=mu_np[:, H_REF_IDX], scale=sigma_np[:, H_REF_IDX])
        hi = scipy_t.ppf(1-alpha, df=nu_np[:, H_REF_IDX], loc=mu_np[:, H_REF_IDX], scale=sigma_np[:, H_REF_IDX])
        coverages[level] = ((Y_rel[:, H_REF_IDX] >= lo) & (Y_rel[:, H_REF_IDX] <= hi)).mean()

    print(f"\nCoverage (h=10d/{H_REF} bars):")
    for level in coverage_levels:
        print(f"  {level*100:.0f}%: {coverages[level]*100:.1f}%")

    # Rank IC by horizon (use daily dates for grouping)
    print("\nRank IC by horizon:")
    rank_ics_by_h = {}
    for h_bars, h_name in sorted(HORIZONS.items()):
        h_idx = h_bars - 1
        if h_idx >= Y_rel.shape[1]:
            continue
        ics = compute_rank_ic(pred_mean[:, h_idx], Y_rel[:, h_idx], dates)
        rank_ics_by_h[h_bars] = ics
        tstat = ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8)
        print(f"  h={h_name:>3} ({h_bars:>2} bars): IC={ics.mean():.4f} +/- {ics.std():.4f} (t={tstat:.1f})")

    # Long-short (at 1-day horizon = 6 bars)
    h_ls = 5  # index for 6-bar (1-day) horizon
    ls_returns = long_short_backtest(pred_mean[:, h_ls], Y_rel[:, h_ls], dates)
    if len(ls_returns) > 0:
        sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8) * np.sqrt(252)
        cum_ret = np.cumsum(ls_returns)
        print(f"\nLong-short (h=1d/6 bars):")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Cumulative: {cum_ret[-1]*100:.1f}%")
        print(f"  Win rate: {(ls_returns > 0).mean()*100:.0f}%")

    # Also L/S at 10-day horizon
    ls_10d = long_short_backtest(pred_mean[:, H_REF_IDX], Y_rel[:, H_REF_IDX], dates)
    if len(ls_10d) > 0:
        sharpe_10d = ls_10d.mean() / (ls_10d.std() + 1e-8) * np.sqrt(252)
        print(f"\nLong-short (h=10d/{H_REF} bars):")
        print(f"  Sharpe: {sharpe_10d:.2f}")
        print(f"  Cumulative: {np.sum(ls_10d)*100:.1f}%")
        print(f"  Win rate: {(ls_10d > 0).mean()*100:.0f}%")

    # Conditional signal
    corr_mean = np.corrcoef(pred_mean[:, H_REF_IDX], Y_rel[:, H_REF_IDX])[0, 1]
    corr_std = np.corrcoef(sigma_np[:, H_REF_IDX],
                            np.abs(Y_rel[:, H_REF_IDX] - pred_mean[:, H_REF_IDX]))[0, 1]
    print(f"\nConditional signal (h=10d):")
    print(f"  Pred mean std: {pred_mean[:, H_REF_IDX].std():.4f}")
    print(f"  Corr(mean, actual): {corr_mean:.4f}")
    print(f"  Corr(std, |error|): {corr_std:.4f}")

    # ═══════════ PLOT ═══════════
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: IC distribution, IC by horizon, L/S cumulative
    ax = fig.add_subplot(gs[0, 0])
    if H_REF in rank_ics_by_h:
        ics = rank_ics_by_h[H_REF]
        ax.hist(ics, bins=30, alpha=0.7, color='#F7931A', edgecolor='white')
        ax.axvline(ics.mean(), color='red', ls='--', lw=2, label=f'Mean: {ics.mean():.4f}')
        ax.axvline(0, color='gray', ls='--')
        ax.legend()
    ax.set_xlabel('Rank IC'); ax.set_ylabel('Count')
    ax.set_title(f'Rank IC Distribution (h=10d)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    if rank_ics_by_h:
        h_vals = sorted(rank_ics_by_h.keys())
        ic_means = [rank_ics_by_h[h].mean() for h in h_vals]
        ic_stds = [rank_ics_by_h[h].std() for h in h_vals]
        h_labels = [HORIZONS[h] for h in h_vals]
        ax.errorbar(range(len(h_vals)), ic_means, yerr=ic_stds, fmt='o-',
                    color='#F7931A', capsize=3, lw=2, ms=6)
        ax.set_xticks(range(len(h_vals)))
        ax.set_xticklabels(h_labels)
        ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Horizon'); ax.set_ylabel('Rank IC')
    ax.set_title('Rank IC by Horizon (4h bars)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if len(ls_10d) > 0:
        cum = np.cumsum(ls_10d) * 100
        ax.plot(cum, color='#F7931A', lw=2)
        ax.axhline(0, color='gray', ls='--')
        ax.fill_between(range(len(cum)), 0, cum,
                        where=cum >= 0, alpha=0.3, color='green')
        ax.fill_between(range(len(cum)), 0, cum,
                        where=cum < 0, alpha=0.3, color='red')
        ax.set_title(f'L/S Cumulative h=10d (Sharpe={sharpe_10d:.2f})')
    ax.set_xlabel('Trading days'); ax.set_ylabel('Cumulative (%)')
    ax.grid(True, alpha=0.3)

    # Row 2: PIT, coverage, pred vs actual
    pit = scipy_t.cdf(Y_rel[:, H_REF_IDX], df=nu_np[:, H_REF_IDX],
                       loc=mu_np[:, H_REF_IDX], scale=sigma_np[:, H_REF_IDX])

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(pit, bins=30, density=True, alpha=0.7, color='#F7931A', edgecolor='white')
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Ideal')
    ax.set_xlabel('PIT'); ax.set_ylabel('Density')
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
    ax.set_title('Coverage (h=10d)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(Y_rel[:, H_REF_IDX][::3], pred_mean[:, H_REF_IDX][::3],
               alpha=0.1, s=3, color='#F7931A')
    lim = np.percentile(np.abs(Y_rel[:, H_REF_IDX]), 99) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual (h=10d, corr={corr_mean:.3f})')
    ax.grid(True, alpha=0.3)

    # Row 3: sigma/mu term structure, daily returns
    ax = fig.add_subplot(gs[2, 0])
    mean_sigma = sigma_np.mean(axis=0)
    bars = np.arange(1, cfg.max_horizon + 1)
    ax.plot(bars, mean_sigma, '-', color='#F7931A', lw=2, label='Model sigma')
    theoretical = mean_sigma[0] * np.sqrt(bars)
    ax.plot(bars, theoretical, 'r--', lw=1.5, label=r'$\sigma_1 \sqrt{h}$')
    ax.set_xlabel('Horizon (4h bars)'); ax.set_ylabel('Mean sigma')
    ax.set_title('Sigma Term Structure')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    mean_mu = pred_mean.mean(axis=0)
    std_mu = pred_mean.std(axis=0)
    ax.plot(bars, mean_mu, '-', color='#F7931A', lw=2)
    ax.fill_between(bars, mean_mu - std_mu, mean_mu + std_mu, alpha=0.3, color='#F7931A')
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Horizon (4h bars)'); ax.set_ylabel('Predicted mu')
    ax.set_title('Mu Term Structure')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    if len(ls_10d) > 0:
        ax.bar(range(len(ls_10d)), ls_10d * 100,
               color=['green' if r > 0 else 'red' for r in ls_10d], alpha=0.6)
        ax.axhline(0, color='gray', ls='--')
        ax.set_xlabel('Trading day'); ax.set_ylabel('Daily L/S (%)')
        ax.set_title(f'Daily Returns h=10d (win={100*(ls_10d>0).mean():.0f}%)')
    ax.grid(True, alpha=0.3)

    # Row 4: IC over time + summary
    ax = fig.add_subplot(gs[3, 0:2])
    if H_REF in rank_ics_by_h:
        ics = rank_ics_by_h[H_REF]
        ax.bar(range(len(ics)), ics,
               color=['green' if x > 0 else 'red' for x in ics], alpha=0.5)
        if len(ics) >= 20:
            rolling = np.convolve(ics, np.ones(20)/20, mode='valid')
            ax.plot(range(19, 19+len(rolling)), rolling, 'b-', lw=2, label='20-day rolling')
        ax.axhline(ics.mean(), color='red', ls='--', alpha=0.5, label=f'Mean: {ics.mean():.4f}')
        ax.axhline(0, color='gray', ls='--')
        ax.legend()
    ax.set_xlabel('Trading day'); ax.set_ylabel('Rank IC')
    ax.set_title('IC Over Time (h=10d)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    lines = [
        f"Checkpoint: step {step:,}",
        f"Test: {N:,} samples",
        f"Context: {cfg.context_len} bars (4h)",
        f"Horizons: {cfg.max_horizon} (4h bars)",
        "",
        f"NLL: {nll_val:.4f}",
        f"Pred mean std: {pred_mean[:, H_REF_IDX].std():.4f}",
        f"Corr(mean,y): {corr_mean:.4f}",
        f"Corr(std,|err|): {corr_std:.4f}",
        "",
        "--- Coverage (10d) ---",
    ]
    for level in coverage_levels:
        lines.append(f"  {level*100:.0f}%: {coverages[level]*100:.1f}%")

    if H_REF in rank_ics_by_h:
        ics = rank_ics_by_h[H_REF]
        tstat = ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8)
        lines += ["", f"IC (10d): {ics.mean():.4f} (t={tstat:.1f})"]
    if len(ls_10d) > 0:
        lines += [f"L/S Sharpe: {sharpe_10d:.2f}",
                  f"L/S cum: {np.sum(ls_10d)*100:.1f}%"]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle(f'Phantom v7 — 4h Crypto Evaluation (step {step:,})', fontsize=16, y=0.99)
    out_path = args.output or f'plots/eval_v7_{step}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
