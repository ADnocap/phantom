#!/usr/bin/env python
"""
Compare multiple Phantom v2 experiment runs side-by-side.

Usage:
  python scripts/eval/plot_experiments.py
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_train(csv_path):
    df = pd.read_csv(csv_path)
    if 'grad_norm' in df.columns:
        return df[df['grad_norm'].notna()].copy().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='plots/experiments_comparison.png')
    args = parser.parse_args()

    experiments = {
        'Exp1: Student-t+Gumbel': 'logs/exp1_train_log.csv',
        'Exp2: MultiScale+Decomp': 'logs/exp2_train_log.csv',
        'Exp3: Full v2 LR=1e-4': 'logs/exp3_train_log.csv',
        'ExpA: CRPS-avg+Contr': 'logs/expA_train_log.csv',
        'ExpB: FiLM+CRPS-avg': 'logs/expB_train_log.csv',
        'ExpC: All fixes': 'logs/expC_train_log.csv',
        'ExpD: MomentMatch': 'logs/expD_train_log.csv',
    }

    # High-contrast distinct colors
    colors = [
        '#1f77b4',  # Exp1: blue
        '#ff7f0e',  # Exp2: orange
        '#2ca02c',  # Exp3: green
        '#d62728',  # ExpA: red
        '#9467bd',  # ExpB: purple
        '#17becf',  # ExpC: cyan
        '#e377c2',  # ExpD: magenta/pink
    ]
    data = {}
    for name, path in experiments.items():
        try:
            data[name] = load_train(path)
            print(f'{name}: {len(data[name])} rows, steps {int(data[name]["step"].iloc[0])}-{int(data[name]["step"].iloc[-1])}')
        except Exception as e:
            print(f'{name}: FAILED to load ({e})')

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle('Phantom v2 — Experiment Comparison', fontsize=16, y=0.98)

    metrics = [
        ('ed', 'Energy Distance', False),
        ('nll', 'NLL (random branch)', False),
        ('loss', 'Total Loss', False),
        ('sde_acc', 'SDE Accuracy', True),
        ('vol_mse', 'Volatility MSE', False),
        ('eff_k', 'Effective K', False),
        ('mean_sigma', 'Mean Sigma', False),
        ('grad_norm', 'Grad Norm', False),
    ]

    # Check if any experiment has mean_nu
    has_nu = any('mean_nu' in df.columns and df['mean_nu'].notna().any()
                 for df in data.values())
    if has_nu:
        metrics.append(('mean_nu', 'Mean Nu (Student-t df)', False))

    for idx, (col, title, is_pct) in enumerate(metrics):
        row, c = divmod(idx, 3)
        ax = axes[row, c]

        for (name, df), color in zip(data.items(), colors):
            if col not in df.columns or df[col].isna().all():
                continue
            vals = df[col].values * (100 if is_pct else 1)
            steps = df['step'].values

            # Raw data (transparent)
            ax.plot(steps, vals, color=color, alpha=0.3, lw=0.8)
            # Rolling mean
            window = max(5, len(vals) // 20)
            rolling = pd.Series(vals).rolling(window, min_periods=1).mean()
            ax.plot(steps, rolling, color=color, lw=2, label=name.split('\n')[0])

        if col == 'sde_acc':
            ax.axhline(14.3, color='red', ls='--', alpha=0.4, label='Random (14%)')
        if col == 'eff_k':
            ax.axhline(5, color='gray', ls='--', alpha=0.4)
            ax.set_ylim(0, 5.5)

        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Summary table in the last cell if we have < 9 metrics
    if len(metrics) < 9:
        ax = axes[2, 2]
        ax.axis('off')
        lines = []
        for name, df in data.items():
            last = df.iloc[-1]
            short = name.split('\n')[0]
            lines.append(f'{short}:')
            lines.append(f'  Step {int(last["step"]):,}  ED={last["ed"]:.5f}  NLL={last["nll"]:.3f}')
            lines.append(f'  SDE={last["sde_acc"]*100:.1f}%  EffK={last.get("eff_k", 0):.1f}  GradN={last.get("grad_norm", 0):.1f}')
            lines.append('')
        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
