#!/usr/bin/env python
"""
Plot Phantom fine-tuning metrics from CSV log.

Usage:
  python scripts/eval/plot_finetune.py
  python scripts/eval/plot_finetune.py --log logs/ft_expD_log.csv --output plots/finetune_expD.png
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='logs/ft_expD_log.csv')
    parser.add_argument('--output', type=str, default='plots/finetune_expD.png')
    parser.add_argument('--title', type=str, default='ExpD Fine-tuning on Real BTC')
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    print(f'Loaded {len(df)} rows, steps {int(df["step"].iloc[0])}-{int(df["step"].iloc[-1])}')

    # Separate training and validation rows
    # Val rows have fewer columns or specific patterns
    # The CSV has: step, loss, real_crps, synth_ed, steps/s, real_frac
    # Val rows have: step, val_crps, val_nll (only 3 fields)
    # Detect by checking if real_frac is NaN or column count

    # Actually the val rows get shoehorned into the same columns
    # Val has val_crps, val_nll → mapped to loss, real_crps columns
    # Detect by: val rows have very different 'loss' values (0.03x vs -0.06x)

    train = df[df['loss'] < 0].copy().reset_index(drop=True)  # training rows have negative loss
    val = df[df['loss'] > 0].copy().reset_index(drop=True)    # val rows have positive loss (val_crps)

    print(f'Training rows: {len(train)}, Validation rows: {len(val)}')

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(args.title, fontsize=16, y=0.98)

    steps = train['step'].values

    # Real CRPS
    ax = axes[0, 0]
    ax.plot(steps, train['real_crps'], color='#d62728', lw=1.5, alpha=0.5)
    if len(steps) > 5:
        w = max(3, len(steps) // 15)
        ax.plot(steps, train['real_crps'].rolling(w, min_periods=1).mean(),
                color='#d62728', lw=2.5, label='Train CRPS (rolling)')
    if len(val) > 0:
        ax.scatter(val['step'], val['loss'], color='blue', s=50, zorder=5, label='Val CRPS')
        for _, r in val.iterrows():
            ax.annotate(f'{r["loss"]:.4f}', (r['step'], r['loss']),
                       textcoords='offset points', xytext=(5, 5), fontsize=8, color='blue')
    ax.set_xlabel('Step'); ax.set_ylabel('CRPS')
    ax.set_title('Real BTC CRPS')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Synthetic ED
    ax = axes[0, 1]
    ax.plot(steps, train['synth_ed'], color='#1f77b4', lw=1.5, alpha=0.5)
    if len(steps) > 5:
        ax.plot(steps, train['synth_ed'].rolling(w, min_periods=1).mean(),
                color='#1f77b4', lw=2.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Energy Distance')
    ax.set_title('Synthetic ED (catastrophic forgetting check)')
    ax.grid(True, alpha=0.3)

    # Total loss
    ax = axes[0, 2]
    ax.plot(steps, train['loss'], color='#2ca02c', lw=1.5, alpha=0.5)
    if len(steps) > 5:
        ax.plot(steps, train['loss'].rolling(w, min_periods=1).mean(),
                color='#2ca02c', lw=2.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)

    # Real fraction (annealing)
    ax = axes[1, 0]
    if 'real_frac' in train.columns:
        ax.plot(steps, train['real_frac'] * 100, color='#ff7f0e', lw=2)
        ax.set_ylabel('Real Data %')
        ax.set_title('Real Fraction Annealing')
        ax.set_ylim(0, 100)
    ax.set_xlabel('Step'); ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1, 1]
    if 'steps/s' in train.columns:
        ax.plot(steps, train['steps/s'], color='k', lw=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Steps/s')
    ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 2]
    ax.axis('off')
    last_train = train.iloc[-1]
    lines = [
        f"Step: {int(last_train['step']):,} / 10,000",
        f"Real CRPS: {last_train['real_crps']:.4f}",
        f"Synth ED: {last_train['synth_ed']:.6f}",
        f"Real fraction: {last_train.get('real_frac', 0.3)*100:.0f}%",
        "",
    ]
    if len(val) > 0:
        lines.append(f"Best Val CRPS: {val['loss'].min():.4f}")
        lines.append(f"Last Val CRPS: {val['loss'].iloc[-1]:.4f}")
        lines.append(f"Val NLL: {val['real_crps'].iloc[-1]:.4f}")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=13,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
