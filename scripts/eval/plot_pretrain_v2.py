#!/usr/bin/env python
"""
Plot Phantom v2 pre-training metrics from the CSV log.

Usage:
  python scripts/eval/plot_pretrain_v2.py
  python scripts/eval/plot_pretrain_v2.py --log logs/pretrain_v2_train_log.csv --output plots/pretrain_v2_live.png
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_and_split(csv_path):
    """Load CSV and separate training vs validation rows."""
    df = pd.read_csv(csv_path)

    # Validation rows have NaN in grad_norm (they use different metric keys)
    if 'grad_norm' in df.columns:
        train = df[df['grad_norm'].notna()].copy().reset_index(drop=True)
        val = df[df['grad_norm'].isna()].copy().reset_index(drop=True)
    else:
        train = df.copy()
        val = pd.DataFrame()

    return train, val


def plot_training(train, val, output_path, title_suffix=''):
    has_nu = 'mean_nu' in train.columns and train['mean_nu'].notna().any()
    has_aux = 'loss_aux' in train.columns

    fig, axes = plt.subplots(4, 4, figsize=(26, 22))
    fig.suptitle(f'Phantom v2 Pre-training{title_suffix}', fontsize=16, y=0.98)

    steps = train['step'].values

    # ── Row 1: Core losses ──
    ax = axes[0, 0]
    ax.plot(steps, train['loss'], 'b-', lw=1.5, label='Total')
    if has_aux:
        ax.plot(steps, train['loss_main'], 'r-', lw=1.5, label='Main')
    ax.set_ylabel('Loss'); ax.set_title('Total Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 1]
    ax.plot(steps, train['ed'], color='#2196F3', lw=1.5)
    ax.set_ylabel('Energy Distance'); ax.set_title('Energy Distance')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 2]
    ax.plot(steps, train['nll'], color='#4CAF50', lw=1.5)
    ax.set_ylabel('NLL'); ax.set_title('NLL (random branch)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 3]
    if has_aux:
        ax.plot(steps, train['loss_aux'], color='purple', lw=1.5, label='Aux total')
        ax.plot(steps, train['loss_sde'], color='#FF9800', lw=1.5, label='SDE CE')
        ax.plot(steps, train['vol_mse'], color='#00BCD4', lw=1.5, label='Vol MSE')
        ax.legend(fontsize=8)
    ax.set_ylabel('Loss'); ax.set_title('Auxiliary Losses')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ── Row 2: Auxiliary task metrics ──
    ax = axes[1, 0]
    ax.plot(steps, train['sde_acc'] * 100, color='#FF9800', lw=1.5)
    n_sde = 7 if train['sde_acc'].iloc[0] < 0.2 else 5
    ax.axhline(100 / n_sde, color='red', ls='--', alpha=0.5, label=f'Random ({100/n_sde:.0f}%)')
    ax.set_ylabel('Accuracy (%)'); ax.set_title('SDE Classification Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 1]
    ax.plot(steps, train['vol_mse'], color='#00BCD4', lw=1.5)
    ax.set_ylabel('MSE'); ax.set_title('Volatility Regression MSE')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 2]
    if 'loss_sde' in train.columns:
        ax.plot(steps, train['loss_sde'], color='#FF9800', lw=1.5)
    ax.set_ylabel('Cross Entropy'); ax.set_title('SDE Classification Loss')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 3]
    ax.plot(steps, train['lr'], 'k-', lw=1.5)
    ax.set_ylabel('Learning Rate'); ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # ── Row 3: Head statistics ──
    ax = axes[2, 0]
    if 'mean_mu' in train.columns:
        ax.plot(steps, train['mean_mu'], color='#2196F3', lw=1.5)
    ax.set_ylabel('|mu| mean'); ax.set_title('Mean |Component Location|')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 1]
    if 'mean_sigma' in train.columns:
        ax.plot(steps, train['mean_sigma'], color='#F44336', lw=1.5)
    ax.set_ylabel('sigma mean'); ax.set_title('Mean Component Scale')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 2]
    if has_nu:
        ax.plot(steps, train['mean_nu'], color='#4CAF50', lw=1.5)
        ax.axhline(30, color='gray', ls='--', alpha=0.5, label='nu=30 (~Gaussian)')
        ax.axhline(5, color='red', ls='--', alpha=0.5, label='nu=5 (heavy tails)')
        ax.axhline(2.01, color='darkred', ls='--', alpha=0.5, label='nu=2 (min)')
        ax.legend(fontsize=8)
    ax.set_ylabel('nu (df)'); ax.set_title('Mean Student-t Degrees of Freedom')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 3]
    if 'eff_k' in train.columns:
        ax.plot(steps, train['eff_k'], color='#009688', lw=1.5)
        K = 5
        ax.axhline(K, color='gray', ls='--', alpha=0.5, label=f'K={K} (max)')
        ax.axhline(1, color='red', ls='--', alpha=0.5, label='K=1 (collapsed)')
        ax.set_ylim(0, K + 0.5)
        ax.legend(fontsize=8)
    ax.set_ylabel('Effective K'); ax.set_title('Effective Component Count')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ── Row 4: Training dynamics + summary ──
    ax = axes[3, 0]
    if 'grad_norm' in train.columns:
        ax.plot(steps, train['grad_norm'], color='#9C27B0', lw=1, alpha=0.7)
        # Rolling mean
        if len(steps) > 5:
            window = max(3, len(steps) // 10)
            rolling = train['grad_norm'].rolling(window, min_periods=1).mean()
            ax.plot(steps, rolling, color='#9C27B0', lw=2, label=f'Rolling {window}')
            ax.legend(fontsize=8)
    ax.set_ylabel('Grad Norm'); ax.set_title('Gradient Norm (before clip)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[3, 1]
    if 'steps/s' in train.columns:
        ax.plot(steps, train['steps/s'], 'k-', lw=1.5)
    ax.set_ylabel('Steps/s'); ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # Loss decomposition
    ax = axes[3, 2]
    if has_aux:
        ed = train['ed'].values
        nll_scaled = np.abs(train['nll'].values * 0.1)
        aux_scaled = train['loss_aux'].values * 0.5
        ax.plot(steps, ed, lw=1.5, label='ED', color='#2196F3')
        ax.plot(steps, nll_scaled, lw=1.5, label='0.1*|NLL|', color='#4CAF50')
        ax.plot(steps, aux_scaled, lw=1.5, label='0.5*Aux', color='purple')
        ax.legend(fontsize=8)
    ax.set_ylabel('Loss'); ax.set_title('Weighted Loss Components')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # Summary panel
    ax = axes[3, 3]
    ax.axis('off')
    last = train.iloc[-1]
    total_steps = 117180
    pct = last['step'] / total_steps * 100
    eta_hrs = last.get('eta_min', 0) / 60 if 'eta_min' in last else 0

    lines = [
        f"Step: {int(last['step']):,} / {total_steps:,} ({pct:.1f}%)",
        f"Epoch: {int(last['epoch'])}",
    ]
    if 'steps/s' in last:
        lines.append(f"Throughput: {last['steps/s']:.1f} steps/s")
    if eta_hrs > 0:
        lines.append(f"ETA: {eta_hrs:.1f} hrs")
    lines.append("")
    lines.append("--- Latest Metrics ---")
    lines.append(f"Energy Distance:  {last['ed']:.6f}")
    lines.append(f"NLL:              {last['nll']:.4f}")
    lines.append(f"SDE Accuracy:     {last['sde_acc']*100:.1f}%")
    lines.append(f"Vol MSE:          {last['vol_mse']:.4f}")
    if 'eff_k' in last:
        lines.append(f"Effective K:      {last['eff_k']:.2f} / 5")
    if has_nu:
        lines.append(f"Mean nu:          {last['mean_nu']:.1f}")
    if 'grad_norm' in last:
        lines.append(f"Grad Norm:        {last['grad_norm']:.2f}")

    summary = "\n".join(lines)
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Phantom v2 pre-training metrics")
    parser.add_argument('--log', type=str, default='logs/pretrain_v2_train_log.csv',
                        help='Path to training CSV log')
    parser.add_argument('--output', type=str, default='plots/pretrain_v2_live.png',
                        help='Output plot path')
    parser.add_argument('--title', type=str, default='',
                        help='Extra title suffix')
    args = parser.parse_args()

    train, val = load_and_split(args.log)
    print(f"Loaded {len(train)} training rows, {len(val)} validation rows")
    print(f"Steps: {int(train['step'].iloc[0])} to {int(train['step'].iloc[-1])}")

    plot_training(train, val, args.output, title_suffix=args.title)


if __name__ == "__main__":
    main()
