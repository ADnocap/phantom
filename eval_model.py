"""Evaluate best.pt checkpoint on fresh synthetic test data."""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm as scipy_norm

from src.model import PhantomConfig, PhantomModel
from src.data import make_validation_batch
from src.losses import combined_loss


def mog_cdf(y, pi, mu, sigma):
    return np.sum(pi * scipy_norm.cdf(y, loc=mu, scale=sigma))


def mog_quantile(p, pi, mu, sigma, lo=-5, hi=5):
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.sum(pi * scipy_norm.cdf(mid, loc=mu, scale=sigma)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_mog(pi, mu, sigma, n=5000):
    components = np.random.choice(len(pi), size=n, p=pi)
    return np.random.normal(mu[components], sigma[components])


def main():
    # ── Load model ──
    ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    print(f"Checkpoint step: {ckpt['step']:,}, epoch: {ckpt['epoch']}, best_val_loss: {ckpt['best_val_loss']:.4f}")

    cfg = PhantomConfig(**{k: v for k, v in cfg_dict.items() if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Generate test data (different seed from training) ──
    print("Generating test set...")
    test_x, test_h, test_y = make_validation_batch(n_samples=8192, context_len=75, seed=7777)

    # ── Get predictions ──
    with torch.no_grad():
        log_pi, mu, sigma = model(test_x, test_h)

    pi = torch.exp(log_pi)
    pi_np, mu_np, sigma_np = pi.numpy(), mu.numpy(), sigma.numpy()
    y_np, h_np = test_y.numpy(), test_h.numpy()

    print(f"Pi range: [{pi_np.min():.4f}, {pi_np.max():.4f}]")
    print(f"Mu range: [{mu_np.min():.4f}, {mu_np.max():.4f}]")
    print(f"Sigma range: [{sigma_np.min():.4f}, {sigma_np.max():.4f}]")

    # ── Loss ──
    with torch.no_grad():
        loss, nll, crps_val = combined_loss(log_pi, mu, sigma, test_y, alpha=0.3)
    print(f"Test loss: {loss.item():.4f}, NLL: {nll.item():.4f}, CRPS: {crps_val.item():.4f}")

    # ── PIT values ──
    pit_values = np.array([mog_cdf(y_np[i], pi_np[i], mu_np[i], sigma_np[i]) for i in range(len(y_np))])

    # ── Coverage ──
    coverages = {}
    for level in [0.50, 0.80, 0.90, 0.95]:
        alpha = (1 - level) / 2
        inside = 0
        for i in range(len(y_np)):
            lo = mog_quantile(alpha, pi_np[i], mu_np[i], sigma_np[i])
            hi = mog_quantile(1 - alpha, pi_np[i], mu_np[i], sigma_np[i])
            if lo <= y_np[i] <= hi:
                inside += 1
        cov = inside / len(y_np)
        coverages[level] = cov
        print(f"Coverage {level*100:.0f}%: {cov*100:.1f}% (target: {level*100:.0f}%)")

    # ── CRPS baselines ──
    y_mean, y_std = y_np.mean(), y_np.std()
    crps_gaussian = np.mean(np.abs(y_np - y_mean) - y_std * (1 / np.sqrt(np.pi)))

    crps_per_horizon = 0
    for h in [3, 5, 7]:
        mask = h_np == h
        yh = y_np[mask]
        m, s = yh.mean(), yh.std()
        crps_h = np.mean(np.abs(yh - m) - s * (1 / np.sqrt(np.pi)))
        crps_per_horizon += crps_h * mask.sum() / len(y_np)

    crps_model = crps_val.item()
    print(f"\nCRPS comparison:")
    print(f"  Model MoG:            {crps_model:.4f}")
    print(f"  Marginal Gaussian:    {crps_gaussian:.4f}")
    print(f"  Per-horizon Gaussian: {crps_per_horizon:.4f}")

    # ── Derived stats ──
    pred_mean = (pi_np * mu_np).sum(axis=1)
    pred_var = (pi_np * (sigma_np**2 + mu_np**2)).sum(axis=1) - pred_mean**2
    pred_std = np.sqrt(np.maximum(pred_var, 1e-8))
    abs_error = np.abs(y_np - pred_mean)
    eff_k = np.exp(-np.sum(pi_np * np.log(pi_np + 1e-10), axis=1))

    # ── Plot ──
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. PIT histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(pit_values, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Ideal (uniform)')
    ax.set_xlabel('PIT value')
    ax.set_ylabel('Density')
    ax.set_title('PIT Histogram (Calibration)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # 2. Coverage
    ax = fig.add_subplot(gs[0, 1])
    levels = sorted(coverages.keys())
    actual = [coverages[l] * 100 for l in levels]
    target = [l * 100 for l in levels]
    ax.plot(target, actual, 'o-', color='tab:blue', markersize=8, linewidth=2, label='Model')
    ax.plot([40, 100], [40, 100], '--', color='gray', linewidth=1, label='Perfect calibration')
    for t, a in zip(target, actual):
        ax.annotate(f'{a:.1f}%', (t, a), textcoords="offset points", xytext=(8, -5), fontsize=9)
    ax.set_xlabel('Target coverage (%)')
    ax.set_ylabel('Actual coverage (%)')
    ax.set_title('Coverage Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 100)
    ax.set_ylim(40, 100)

    # 3. CRPS comparison
    ax = fig.add_subplot(gs[0, 2])
    methods = ['Model\n(MoG K=8)', 'Marginal\nGaussian', 'Per-horizon\nGaussian']
    crps_vals = [crps_model, crps_gaussian, crps_per_horizon]
    colors = ['tab:blue', 'tab:gray', 'tab:gray']
    bars = ax.bar(methods, crps_vals, color=colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, crps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=10)
    ax.set_ylabel('CRPS (lower = better)')
    ax.set_title('CRPS vs Baselines')
    ax.grid(True, alpha=0.3, axis='y')

    # 4-6. Sample predicted distributions per horizon
    np.random.seed(42)
    for idx, h_val in enumerate([3, 5, 7]):
        ax = fig.add_subplot(gs[1, idx])
        mask = (h_np == h_val)
        sample_idxs = np.where(mask)[0][:6]
        for i, si in enumerate(sample_idxs):
            samples = sample_mog(pi_np[si], mu_np[si], sigma_np[si], n=3000)
            color = f'C{i}'
            ax.hist(samples, bins=60, density=True, alpha=0.3, color=color)
            ax.axvline(y_np[si], color=color, linewidth=2, linestyle='-')
        ax.set_xlabel('Forward log-return')
        ax.set_ylabel('Density')
        ax.set_title(f'Predicted Distributions ({h_val}d horizon)')
        ax.grid(True, alpha=0.3)

    # 7. Predicted mean vs actual
    ax = fig.add_subplot(gs[2, 0])
    for h_val, color, label in [(3, 'tab:blue', '3d'), (5, 'tab:orange', '5d'), (7, 'tab:green', '7d')]:
        mask = h_np == h_val
        ax.scatter(y_np[mask][::3], pred_mean[mask][::3], alpha=0.15, s=3, color=color, label=label)
    lim = max(abs(y_np.min()), abs(y_np.max()), abs(pred_mean.min()), abs(pred_mean.max())) * 0.8
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted mean')
    ax.set_title('Predicted Mean vs Actual')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # 8. Uncertainty calibration
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(pred_std[::3], abs_error[::3], alpha=0.1, s=3, color='purple')
    ax.plot([0, pred_std.max()], [0, pred_std.max()], 'r--', linewidth=1, label='Perfect')
    ax.set_xlabel('Predicted std')
    ax.set_ylabel('|Actual - Predicted mean|')
    ax.set_title('Uncertainty Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Effective component count
    ax = fig.add_subplot(gs[2, 2])
    ax.hist(eff_k, bins=50, density=True, alpha=0.7, color='teal', edgecolor='white')
    ax.axvline(eff_k.mean(), color='red', linestyle='--', label=f'Mean: {eff_k.mean():.1f}')
    ax.set_xlabel('Effective # components')
    ax.set_ylabel('Density')
    ax.set_title(f'MoG Component Usage (K=8, eff={eff_k.mean():.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Phantom Model Evaluation -- best.pt (step {ckpt['step']:,}, val_loss={ckpt['best_val_loss']:.4f})",
        fontsize=15, y=0.99,
    )
    plt.savefig('model_eval.png', dpi=150, bbox_inches='tight')
    print("\nSaved model_eval.png")


if __name__ == "__main__":
    main()
