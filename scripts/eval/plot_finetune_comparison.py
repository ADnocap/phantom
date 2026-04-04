#!/usr/bin/env python
"""Compare fine-tuning experiments side-by-side."""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_ft_log(csv_path):
    df = pd.read_csv(csv_path)
    train = df[df['loss'] < 0].copy().reset_index(drop=True)
    val = df[df['loss'] > 0].copy().reset_index(drop=True)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='plots/finetune_comparison.png')
    args = parser.parse_args()

    experiments = {
        'FT-D: Baseline': 'logs/ft_expD_log.csv',
        'FT-F: Aggressive encoder': 'logs/ftF_log.csv',
        'FT-H: Heavy NLL + anneal': 'logs/ftH_log.csv',
    }

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    data = {}
    for name, path in experiments.items():
        if Path(path).exists():
            train, val = load_ft_log(path)
            data[name] = {'train': train, 'val': val}
            print(f'{name}: {len(train)} train rows, {len(val)} val rows, steps {int(train["step"].iloc[0])}-{int(train["step"].iloc[-1])}')
        else:
            print(f'{name}: NOT FOUND')

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Phantom Fine-tuning — Experiment Comparison', fontsize=16, y=0.98)

    # Real CRPS (training)
    ax = axes[0, 0]
    for (name, d), color in zip(data.items(), colors):
        train = d['train']
        steps = train['step'].values
        w = max(3, len(steps) // 20)
        ax.plot(steps, train['real_crps'].rolling(w, min_periods=1).mean(),
                color=color, lw=2, label=name.split(':')[0])
    ax.set_xlabel('Step'); ax.set_ylabel('CRPS')
    ax.set_title('Real BTC CRPS (train)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Val CRPS
    ax = axes[0, 1]
    for (name, d), color in zip(data.items(), colors):
        val = d['val']
        if len(val) > 0:
            ax.plot(val['step'], val['loss'], 'o-', color=color, lw=2,
                    markersize=4, label=name.split(':')[0])
    ax.set_xlabel('Step'); ax.set_ylabel('Val CRPS')
    ax.set_title('Validation CRPS')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Synthetic ED
    ax = axes[0, 2]
    for (name, d), color in zip(data.items(), colors):
        train = d['train']
        if 'synth_ed' in train.columns:
            steps = train['step'].values
            w = max(3, len(steps) // 20)
            ax.plot(steps, train['synth_ed'].rolling(w, min_periods=1).mean(),
                    color=color, lw=2, label=name.split(':')[0])
    ax.set_xlabel('Step'); ax.set_ylabel('ED')
    ax.set_title('Synthetic ED (forgetting check)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Total loss
    ax = axes[1, 0]
    for (name, d), color in zip(data.items(), colors):
        train = d['train']
        steps = train['step'].values
        w = max(3, len(steps) // 20)
        ax.plot(steps, train['loss'].rolling(w, min_periods=1).mean(),
                color=color, lw=2, label=name.split(':')[0])
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Real fraction (if annealing)
    ax = axes[1, 1]
    for (name, d), color in zip(data.items(), colors):
        train = d['train']
        if 'real_frac' in train.columns and train['real_frac'].notna().any():
            ax.plot(train['step'], train['real_frac'] * 100, color=color, lw=2,
                    label=name.split(':')[0])
    ax.set_xlabel('Step'); ax.set_ylabel('Real %')
    ax.set_title('Real Data Fraction')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Summary
    ax = axes[1, 2]
    ax.axis('off')
    lines = []
    for name, d in data.items():
        val = d['val']
        train = d['train']
        last = train.iloc[-1]
        short = name.split(':')[0]
        lines.append(f'{short} (step {int(last["step"]):,}):')
        lines.append(f'  Real CRPS: {last["real_crps"]:.4f}')
        if len(val) > 0:
            lines.append(f'  Best Val:  {val["loss"].min():.4f}')
            lines.append(f'  Last Val:  {val["loss"].iloc[-1]:.4f}')
        lines.append('')
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
