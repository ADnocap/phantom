#!/usr/bin/env python
"""
Evaluate v3 checkpoint on the real multi-asset test set.

Produces a 12-panel evaluation plot:
  Row 1: PIT histogram, coverage calibration, CRPS vs baselines
  Row 2: Predicted distributions per horizon (3d, 5d, 7d)
  Row 3: Predicted mean vs actual, uncertainty calibration, CRPS by asset type
  Row 4: Per-asset-type coverage, NLL by asset type, summary stats

Usage:
  python scripts/eval/eval_v3.py --checkpoint checkpoints_v3/best.pt
  python scripts/eval/eval_v3.py --checkpoint checkpoints_v3/best.pt --test_data data/processed/test.npz
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import t as scipy_t

from src.model import PhantomConfig, PhantomModel
from src.losses import nll_loss, crps_loss


ASSET_NAMES = {0: 'Crypto', 1: 'Equity', 2: 'Forex', 3: 'Commodity'}
ASSET_COLORS = {0: '#F7931A', 1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}


def student_t_cdf(y, mu, sigma, nu):
    """CDF of Student-t(mu, sigma, nu) at point y."""
    return scipy_t.cdf(y, df=nu, loc=mu, scale=sigma)


def student_t_quantile(p, mu, sigma, nu):
    """Quantile of Student-t(mu, sigma, nu)."""
    return scipy_t.ppf(p, df=nu, loc=mu, scale=sigma)


def main():
    parser = argparse.ArgumentParser(description="Evaluate v3 checkpoint on real test data")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v3/best.pt')
    parser.add_argument('--test_data', type=str, default='data/processed/test.npz')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Subsample test set (None = use all)')
    args = parser.parse_args()

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = PhantomConfig(**{k: v for k, v in cfg_dict.items()
                          if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    step = ckpt['step']
    print(f"Loaded checkpoint: step {step:,}, best_val_loss: {ckpt['best_val_loss']:.4f}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"head={cfg.head_type}, channels={cfg.n_input_channels}")

    # ── Load test data ──
    d = np.load(args.test_data, allow_pickle=True)
    X = d['X'].astype(np.float32)
    H = d['H'].astype(np.int64)
    Y = d['Y'].astype(np.float32)
    asset_type = d['asset_type'].astype(np.int64)

    # Subsample for speed (default 20K — full 127K is slow on CPU)
    n_eval = args.n_samples or min(20000, len(X))
    if n_eval < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), n_eval, replace=False)
        X, H, Y, asset_type = X[idx], H[idx], Y[idx], asset_type[idx]

    print(f"Test set: {len(X)} samples")
    for t in sorted(np.unique(asset_type)):
        n = (asset_type == t).sum()
        print(f"  {ASSET_NAMES.get(t, f'type_{t}')}: {n} ({100*n/len(X):.1f}%)")

    # ── Get predictions (batched) ──
    print("Computing predictions...", flush=True)
    all_mu, all_sigma, all_nu = [], [], []
    batch_size = 1024
    n_batches = (len(X) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = i // batch_size + 1
            print(f"\r  Batch {b}/{n_batches}", end='', flush=True)
            x = torch.from_numpy(X[i:i+batch_size])
            h = torch.from_numpy(H[i:i+batch_size])
            log_pi, mu, sigma, nu = model(x, h)
            all_mu.append(mu.numpy())
            all_sigma.append(sigma.numpy())
            if nu is not None:
                all_nu.append(nu.numpy())
    print()

    mu_np = np.concatenate(all_mu)        # (N, 1) for Student-t head
    sigma_np = np.concatenate(all_sigma)
    nu_np = np.concatenate(all_nu) if all_nu else None

    # Squeeze K=1 dimension
    mu_np = mu_np.squeeze(-1) if mu_np.ndim > 1 else mu_np
    sigma_np = sigma_np.squeeze(-1) if sigma_np.ndim > 1 else sigma_np
    if nu_np is not None:
        nu_np = nu_np.squeeze(-1) if nu_np.ndim > 1 else nu_np

    print(f"Mu range: [{mu_np.min():.4f}, {mu_np.max():.4f}], std: {mu_np.std():.4f}")
    print(f"Sigma range: [{sigma_np.min():.4f}, {sigma_np.max():.4f}]")
    if nu_np is not None:
        print(f"Nu range: [{nu_np.min():.2f}, {nu_np.max():.2f}]")

    # ── Losses (vectorized via scipy, not torch MC) ──
    print("Computing losses...", flush=True)
    # NLL: log-pdf of Student-t evaluated at Y
    nll_val = -scipy_t.logpdf(Y, df=nu_np, loc=mu_np, scale=sigma_np).mean() if nu_np is not None else None

    # CRPS: closed-form for Student-t via scipy
    # CRPS(F, y) = E|X-y| - 0.5*E|X-X'|, approximate with samples
    rng_crps = np.random.default_rng(123)
    n_mc = 500
    samples = scipy_t.rvs(df=np.tile(nu_np, (n_mc, 1)).T,
                           loc=np.tile(mu_np, (n_mc, 1)).T,
                           scale=np.tile(sigma_np, (n_mc, 1)).T,
                           random_state=rng_crps) if nu_np is not None else None  # (N, n_mc)
    term1 = np.abs(samples - Y[:, None]).mean(axis=1)  # E|X-y|
    samples_sorted = np.sort(samples, axis=1)
    diffs = samples_sorted[:, 1:] - samples_sorted[:, :-1]
    idx_w = np.arange(1, n_mc)
    weights = idx_w * (n_mc - idx_w)
    term2 = (diffs * weights[None, :]).sum(axis=1) * 2.0 / (n_mc * n_mc)
    crps_per_sample = term1 - 0.5 * term2
    crps_val = crps_per_sample.mean()

    # For per-asset computation, keep torch tensors for NLL
    log_pi_t = torch.zeros(len(Y), 1)
    mu_t = torch.from_numpy(mu_np).unsqueeze(-1)
    sigma_t = torch.from_numpy(sigma_np).unsqueeze(-1)
    nu_t = torch.from_numpy(nu_np).unsqueeze(-1) if nu_np is not None else None
    y_t = torch.from_numpy(Y)

    print(f"\nOverall NLL: {nll_val:.4f}")
    print(f"Overall CRPS: {crps_val:.4f}")

    # ── PIT values ──
    if nu_np is not None:
        pit_values = student_t_cdf(Y, mu_np, sigma_np, nu_np)
    else:
        from scipy.stats import norm
        pit_values = norm.cdf(Y, loc=mu_np, scale=sigma_np)

    # ── Coverage ──
    coverage_levels = [0.50, 0.80, 0.90, 0.95]
    coverages = {}
    for level in coverage_levels:
        alpha = (1 - level) / 2
        if nu_np is not None:
            lo = student_t_quantile(alpha, mu_np, sigma_np, nu_np)
            hi = student_t_quantile(1 - alpha, mu_np, sigma_np, nu_np)
        else:
            from scipy.stats import norm
            lo = norm.ppf(alpha, loc=mu_np, scale=sigma_np)
            hi = norm.ppf(1 - alpha, loc=mu_np, scale=sigma_np)
        inside = ((Y >= lo) & (Y <= hi)).mean()
        coverages[level] = inside

    # ECE
    pit_bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        mask = (pit_values >= pit_bins[i]) & (pit_values < pit_bins[i+1])
        if mask.sum() > 0:
            expected = 0.1
            actual = mask.mean()
            ece += abs(actual - expected)
    ece /= 10

    print(f"\nCalibration:")
    print(f"  ECE: {ece:.4f}")
    for level in coverage_levels:
        print(f"  Coverage {level*100:.0f}%: {coverages[level]*100:.1f}%")

    # ── CRPS baselines ──
    y_mean, y_std = Y.mean(), Y.std()
    crps_marginal = np.mean(np.abs(Y - y_mean)) - y_std / np.sqrt(np.pi)
    crps_per_horizon = 0
    for h in [3, 5, 7]:
        mask = H == h
        yh = Y[mask]
        m, s = yh.mean(), yh.std()
        crps_per_horizon += (np.mean(np.abs(yh - m)) - s / np.sqrt(np.pi)) * mask.sum() / len(Y)

    print(f"\nCRPS comparison:")
    print(f"  Model:                {crps_val:.4f}")
    print(f"  Marginal Gaussian:    {crps_marginal:.4f}")
    print(f"  Per-horizon Gaussian: {crps_per_horizon:.4f}")

    # ── Per-asset CRPS and NLL ──
    crps_by_asset = {}
    nll_by_asset = {}
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        name = ASSET_NAMES.get(t, f'type_{t}')
        nll_by_asset[name] = -scipy_t.logpdf(Y[mask], df=nu_np[mask], loc=mu_np[mask], scale=sigma_np[mask]).mean() if nu_np is not None else 0
        crps_by_asset[name] = crps_per_sample[mask].mean()
        print(f"  {name}: CRPS={crps_by_asset[name]:.4f}, NLL={nll_by_asset[name]:.4f}")

    # ── Derived stats ──
    pred_mean = mu_np
    pred_std = sigma_np
    abs_error = np.abs(Y - pred_mean)
    corr_mean = np.corrcoef(pred_mean, Y)[0, 1]
    corr_std = np.corrcoef(pred_std, abs_error)[0, 1]

    print(f"\nConditional signal:")
    print(f"  Pred mean std:           {pred_mean.std():.4f}")
    print(f"  Corr(pred_mean, actual): {corr_mean:.4f}")
    print(f"  Corr(pred_std, |error|): {corr_std:.4f}")

    # ═══════════ PLOT ═══════════
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Row 1: Calibration ──

    # 1. PIT histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(pit_values, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Ideal (uniform)')
    ax.set_xlabel('PIT value'); ax.set_ylabel('Density')
    ax.set_title(f'PIT Histogram (ECE={ece:.4f})')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    # 2. Coverage calibration
    ax = fig.add_subplot(gs[0, 1])
    levels_pct = [l * 100 for l in coverage_levels]
    actual_pct = [coverages[l] * 100 for l in coverage_levels]
    ax.plot(levels_pct, actual_pct, 'o-', color='tab:blue', ms=8, lw=2, label='Model')
    ax.plot([40, 100], [40, 100], '--', color='gray', lw=1, label='Perfect')
    for t, a in zip(levels_pct, actual_pct):
        ax.annotate(f'{a:.1f}%', (t, a), textcoords="offset points", xytext=(8, -5), fontsize=9)
    ax.set_xlabel('Target (%)'); ax.set_ylabel('Actual (%)')
    ax.set_title('Coverage Calibration')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 100); ax.set_ylim(40, 100)

    # 3. CRPS comparison
    ax = fig.add_subplot(gs[0, 2])
    methods = ['Model\n(Student-t)', 'Marginal\nGaussian', 'Per-horizon\nGaussian']
    vals = [crps_val, crps_marginal, crps_per_horizon]
    colors = ['tab:blue', 'tab:gray', 'tab:gray']
    bars = ax.bar(methods, vals, color=colors, alpha=0.8, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{v:.4f}', ha='center', fontsize=10)
    ax.set_ylabel('CRPS (lower = better)')
    ax.set_title('CRPS vs Baselines')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Row 2: Sample distributions per horizon ──
    rng = np.random.default_rng(42)
    for idx, h_val in enumerate([3, 5, 7]):
        ax = fig.add_subplot(gs[1, idx])
        mask = H == h_val
        sample_idxs = rng.choice(np.where(mask)[0], min(6, mask.sum()), replace=False)
        for i, si in enumerate(sample_idxs):
            if nu_np is not None:
                samples = scipy_t.rvs(df=nu_np[si], loc=mu_np[si], scale=sigma_np[si], size=3000)
            else:
                samples = np.random.normal(mu_np[si], sigma_np[si], 3000)
            ax.hist(samples, bins=60, density=True, alpha=0.3, color=f'C{i}')
            ax.axvline(Y[si], color=f'C{i}', lw=2)
        ax.set_xlabel('Forward log-return'); ax.set_ylabel('Density')
        ax.set_title(f'Predicted Distributions ({h_val}d)')
        ax.grid(True, alpha=0.3)

    # ── Row 3: Conditional signal ──

    # 4. Predicted mean vs actual (colored by asset type)
    ax = fig.add_subplot(gs[2, 0])
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        name = ASSET_NAMES.get(t, f'{t}')
        ax.scatter(Y[mask][::3], pred_mean[mask][::3], alpha=0.1, s=3,
                   color=ASSET_COLORS.get(t, 'gray'), label=name)
    lim = np.percentile(np.abs(Y), 99) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted mean')
    ax.set_title(f'Pred Mean vs Actual (corr={corr_mean:.3f})')
    ax.legend(markerscale=5, fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # 5. Uncertainty calibration
    ax = fig.add_subplot(gs[2, 1])
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        ax.scatter(pred_std[mask][::3], abs_error[mask][::3], alpha=0.1, s=3,
                   color=ASSET_COLORS.get(t, 'gray'))
    ax.plot([0, pred_std.max()], [0, pred_std.max()], 'r--', lw=1, label='Perfect')
    ax.set_xlabel('Predicted std'); ax.set_ylabel('|Error|')
    ax.set_title(f'Uncertainty Calibration (corr={corr_std:.3f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. CRPS by asset type
    ax = fig.add_subplot(gs[2, 2])
    names = list(crps_by_asset.keys())
    crps_vals = [crps_by_asset[n] for n in names]
    bar_colors = [ASSET_COLORS.get(i, 'gray') for i in sorted(np.unique(asset_type))]
    bars = ax.bar(names, crps_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    for bar, v in zip(bars, crps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f'{v:.4f}', ha='center', fontsize=10)
    ax.set_ylabel('CRPS'); ax.set_title('CRPS by Asset Type')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Row 4: Per-asset coverage + NLL + summary ──

    # 7. Coverage by asset type
    ax = fig.add_subplot(gs[3, 0])
    width = 0.18
    x_pos = np.arange(len(coverage_levels))
    for i, t in enumerate(sorted(np.unique(asset_type))):
        mask = asset_type == t
        name = ASSET_NAMES.get(t, f'{t}')
        covs = []
        for level in coverage_levels:
            alpha = (1 - level) / 2
            if nu_np is not None:
                lo = student_t_quantile(alpha, mu_np[mask], sigma_np[mask], nu_np[mask])
                hi = student_t_quantile(1 - alpha, mu_np[mask], sigma_np[mask], nu_np[mask])
            else:
                from scipy.stats import norm
                lo = norm.ppf(alpha, loc=mu_np[mask], scale=sigma_np[mask])
                hi = norm.ppf(1 - alpha, loc=mu_np[mask], scale=sigma_np[mask])
            covs.append(((Y[mask] >= lo) & (Y[mask] <= hi)).mean() * 100)
        ax.bar(x_pos + i * width, covs, width, label=name,
               color=ASSET_COLORS.get(t, 'gray'), alpha=0.8)
    # Target lines
    for j, level in enumerate(coverage_levels):
        ax.hlines(level * 100, j - 0.2, j + len(np.unique(asset_type)) * width,
                  colors='red', ls='--', alpha=0.5)
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels([f'{l*100:.0f}%' for l in coverage_levels])
    ax.set_ylabel('Actual Coverage (%)'); ax.set_title('Coverage by Asset Type')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # 8. NLL by asset type
    ax = fig.add_subplot(gs[3, 1])
    nll_vals = [nll_by_asset[n] for n in names]
    bars = ax.bar(names, nll_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    for bar, v in zip(bars, nll_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                f'{v:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('NLL (lower = better)'); ax.set_title('NLL by Asset Type')
    ax.grid(True, alpha=0.3, axis='y')

    # 9. Summary
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    lines = [
        f"Checkpoint: step {step:,}",
        f"Val loss:   {ckpt['best_val_loss']:.4f}",
        f"Test samples: {len(Y):,}",
        "",
        "--- Overall ---",
        f"NLL:   {nll_val:.4f}",
        f"CRPS:  {crps_val:.4f}",
        f"ECE:   {ece:.4f}",
        "",
        "--- Coverage ---",
    ]
    for level in coverage_levels:
        lines.append(f"  {level*100:.0f}%: {coverages[level]*100:.1f}%")
    lines += [
        "",
        "--- Conditional Signal ---",
        f"Pred mean std:    {pred_mean.std():.4f}",
        f"Corr(mean, y):    {corr_mean:.4f}",
        f"Corr(std, |err|): {corr_std:.4f}",
        "",
        "--- Baselines ---",
        f"Marginal Gaussian: {crps_marginal:.4f}",
        f"Per-horizon Gauss: {crps_per_horizon:.4f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle(f'Phantom v3 Evaluation — Real Multi-Asset Test Set (step {step:,})',
                 fontsize=16, y=0.99)
    out_path = args.output or f'plots/eval_v3_{step}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
