"""
Visualize model predictions on real BTC data.

Shows:
  1. BTC price paths with context / prediction boundary + predicted distribution fan
  2. Calibration curve with ECE (Expected Calibration Error)
  3. Brier-style reliability diagram

Usage:
  python visualize_btc.py --checkpoint checkpoints_ft/best.pt
  python visualize_btc.py --checkpoint checkpoints/best.pt --cfg_scale 2.0
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm as scipy_norm

from src.model import PhantomConfig, PhantomModel
from src.btc_data import fetch_btc_daily, temporal_split


def mog_cdf(y, pi, mu, sigma):
    return np.sum(pi * scipy_norm.cdf(y, loc=mu, scale=sigma))


def mog_quantile(p, pi, mu, sigma):
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mog_cdf(mid, pi, mu, sigma) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_mog(pi, mu, sigma, n=10000):
    c = np.random.choice(len(pi), size=n, p=pi)
    return np.random.normal(mu[c], sigma[c])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_ft/best.pt')
    parser.add_argument('--btc_cache', type=str, default='data/btc_daily.npz')
    parser.add_argument('--cfg_scale', type=float, default=0.0,
                        help='CFG guidance scale (0 = disabled)')
    parser.add_argument('--output', type=str, default='plots/finetune_btc_predictions.png')
    args = parser.parse_args()

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = PhantomConfig(**{k: v for k, v in ckpt['config'].items()
                          if k in PhantomConfig.__dataclass_fields__})
    cfg.cond_drop_prob = 0.0
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model: step {ckpt['step']:,}, K={cfg.n_components}")

    use_cfg = args.cfg_scale > 0 and hasattr(model, 'forward_cfg')

    # ── Load BTC data ──
    btc = fetch_btc_daily(cache_path=args.btc_cache)
    dates = btc['dates']
    closes = btc['closes']
    log_returns = btc['log_returns']
    context_len = cfg.context_len

    # Use test period (2023-07 onwards)
    test_start_idx = np.searchsorted(dates, '2023-07-01')
    max_h = 7

    # ── Generate predictions for all test windows ──
    print("Computing predictions on test set...")
    all_pits = []        # PIT values for calibration
    all_coverages = {}   # {level: [0/1 for each sample]}
    levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    for l in levels:
        all_coverages[l] = []

    # Store predictions for path visualization
    viz_indices = []
    viz_data = []

    n_test = len(log_returns) - test_start_idx - context_len - max_h
    horizons = [3, 5, 7]

    for i in range(0, n_test, 3):  # stride 3 to reduce computation
        start = test_start_idx + i
        ctx = log_returns[start:start + context_len]

        for h in horizons:
            if start + context_len + h > len(log_returns):
                continue

            fwd = log_returns[start + context_len:start + context_len + h].sum()
            x = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)
            h_t = torch.tensor([h])

            with torch.no_grad():
                if use_cfg:
                    log_pi, mu, sigma = model.forward_cfg(x, h_t, args.cfg_scale)
                else:
                    log_pi, mu, sigma = model(x, h_t)

            pi = torch.exp(log_pi)[0].numpy()
            mu_np = mu[0].numpy()
            sigma_np = sigma[0].numpy()

            # PIT
            pit = mog_cdf(fwd, pi, mu_np, sigma_np)
            all_pits.append(pit)

            # Coverage
            for level in levels:
                alpha = (1 - level) / 2
                q_lo = mog_quantile(alpha, pi, mu_np, sigma_np)
                q_hi = mog_quantile(1 - alpha, pi, mu_np, sigma_np)
                all_coverages[level].append(1 if q_lo <= fwd <= q_hi else 0)

            # Save some for visualization (spread across time, h=5 only)
            if h == 5 and len(viz_data) < 8 and i % 60 == 0:
                viz_data.append({
                    'start': start,
                    'ctx': ctx.copy(),
                    'fwd_actual': fwd,
                    'h': h,
                    'pi': pi, 'mu': mu_np, 'sigma': sigma_np,
                    'date_ctx_end': dates[start + context_len - 1],
                    'date_pred_end': dates[min(start + context_len + h - 1, len(dates) - 1)],
                    'closes_ctx': closes[start:start + context_len],
                    'closes_fwd': closes[start + context_len - 1:start + context_len + h],
                })

    all_pits = np.array(all_pits)
    print(f"Computed {len(all_pits)} predictions")

    # ── Calibration metrics ──
    # ECE: bin PIT values, check uniformity
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.histogram(all_pits, bins=bin_edges)[0]
    bin_freqs = bin_counts / len(all_pits)
    expected_freq = 1.0 / n_bins
    ece = np.mean(np.abs(bin_freqs - expected_freq))

    # Coverage-based calibration
    actual_coverages = {l: np.mean(all_coverages[l]) for l in levels}

    # Brier-style score for each coverage level
    # Brier = mean((actual_coverage - target_coverage)^2)
    brier = np.mean([(actual_coverages[l] - l) ** 2 for l in levels])

    print(f"\nCalibration metrics:")
    print(f"  ECE (PIT uniformity): {ece:.4f}")
    print(f"  Brier (coverage):     {brier:.6f}")
    for l in [0.50, 0.80, 0.90, 0.95]:
        print(f"  Coverage {l*100:.0f}%: {actual_coverages[l]*100:.1f}%")

    # ── PLOT ──
    n_viz = min(len(viz_data), 6)
    fig = plt.figure(figsize=(24, 6 * (n_viz // 2 + 2)))
    gs = GridSpec(n_viz // 2 + 2, 2, figure=fig, hspace=0.4, wspace=0.25)

    # ── Top row: Calibration curve + PIT histogram ──

    # Calibration reliability diagram
    ax = fig.add_subplot(gs[0, 0])
    target_covs = [l * 100 for l in levels]
    actual_covs = [actual_coverages[l] * 100 for l in levels]
    ax.plot(target_covs, actual_covs, 'o-', color='tab:blue', markersize=6, linewidth=2, label='Model')
    ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=1, label='Perfect')
    ax.fill_between([0, 100], [0, 100], alpha=0.05, color='gray')
    for t, a in zip(target_covs, actual_covs):
        if t in [50, 80, 90, 95]:
            ax.annotate(f'{a:.1f}%', (t, a), textcoords='offset points', xytext=(8, -5), fontsize=8)
    ax.set_xlabel('Target Coverage (%)')
    ax.set_ylabel('Actual Coverage (%)')
    ax.set_title(f'Calibration Curve — ECE={ece:.4f}, Brier={brier:.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # PIT histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(all_pits, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Ideal (uniform)')
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title(f'PIT Histogram — Real BTC Test Set ({len(all_pits)} predictions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # ── Path visualizations ──
    for idx in range(n_viz):
        row = idx // 2 + 1
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        d = viz_data[idx]
        ctx_days = np.arange(len(d['ctx']))
        fwd_days = np.arange(len(d['ctx']) - 1, len(d['ctx']) - 1 + len(d['closes_fwd']))

        # Normalize prices to start at 1.0 for visual clarity
        base_price = d['closes_ctx'][0]
        ctx_prices = d['closes_ctx'] / base_price
        fwd_prices = d['closes_fwd'] / base_price

        # Context path (solid blue)
        ax.plot(ctx_days, ctx_prices, linewidth=1.5, color='tab:blue', label='Context (model sees)')

        # Future actual path (solid green)
        ax.plot(fwd_days, fwd_prices, linewidth=2.5, color='tab:green', label='Actual future')

        # Vertical separator
        ax.axvline(len(d['ctx']) - 1, color='black', linewidth=2, linestyle='-', alpha=0.5)

        # Predicted distribution fan (shaded quantile regions)
        last_price_log = np.log(d['closes_ctx'][-1] / base_price)
        quantile_levels = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]
        q_values = [mog_quantile(q, d['pi'], d['mu'], d['sigma']) for q in quantile_levels]

        # Convert log-return quantiles to price levels
        pred_x = len(d['ctx']) - 1 + d['h']  # end of prediction period
        mid_x = len(d['ctx']) - 1 + d['h'] / 2  # middle for fan shape

        for i, (q_lo_lvl, q_hi_lvl, alpha_val, color) in enumerate([
            (0, 6, 0.15, 'tab:red'),    # 2.5%-97.5%
            (1, 5, 0.20, 'tab:red'),    # 10%-90%
            (2, 4, 0.30, 'tab:red'),    # 25%-75%
        ]):
            price_lo = ctx_prices[-1] * np.exp(q_values[q_lo_lvl])
            price_hi = ctx_prices[-1] * np.exp(q_values[q_hi_lvl])
            # Draw as a polygon from context end to prediction end
            xs = [len(d['ctx']) - 1, pred_x, pred_x, len(d['ctx']) - 1]
            ys = [ctx_prices[-1], price_lo, price_hi, ctx_prices[-1]]
            ax.fill(xs, ys, alpha=alpha_val, color=color,
                    label=f'{quantile_levels[q_lo_lvl]*100:.0f}-{quantile_levels[q_hi_lvl]*100:.0f}%' if idx == 0 else None)

        # Median prediction line
        price_median = ctx_prices[-1] * np.exp(q_values[3])
        ax.plot([len(d['ctx']) - 1, pred_x], [ctx_prices[-1], price_median],
                linewidth=1.5, color='tab:red', linestyle='--', label='Predicted median' if idx == 0 else None)

        ax.set_xlabel('Day')
        ax.set_ylabel('Normalized Price')
        ax.set_title(f'{d["date_ctx_end"]} → {d["date_pred_end"]} ({d["h"]}d) | '
                     f'Actual={d["fwd_actual"]:.3f}')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    cfg_str = f' (CFG s={args.cfg_scale})' if use_cfg else ''
    fig.suptitle(f'Phantom — Real BTC Predictions{cfg_str} — Test Period (2023-07 to 2026)',
                 fontsize=16, y=0.99)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
