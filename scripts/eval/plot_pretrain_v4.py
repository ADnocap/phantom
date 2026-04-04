#!/usr/bin/env python
"""
Plot Phantom v4 multi-horizon pre-training metrics from the CSV log.

v4 predicts distributions at all 30 horizons simultaneously.
Key new metrics: mean_mse (horizon-weighted), pred_mean_std (is mu alive?).

Usage:
  python scripts/eval/plot_pretrain_v4.py --log logs/v4/train_log.csv
"""

import argparse
import csv as csvmod
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


VAL_COLUMNS = ['step', 'val_loss', 'val_nll', 'val_crps', 'val_mean_mse',
               'val_pred_mean_std', 'val_vol_mse', 'val_asset_acc',
               'val_mean_sigma', 'val_mean_mu', 'val_mean_nu']


def load_and_split(csv_path):
    """Load training CSV and separate val rows (different column count)."""
    # Try separate val_log.csv first
    val_path = csv_path.replace('train_log.csv', 'val_log.csv')
    try:
        val = pd.read_csv(val_path)
        train = pd.read_csv(csv_path)
        return train, val
    except FileNotFoundError:
        pass

    # Parse raw CSV — train rows have more columns than val rows
    train_rows, val_rows = [], []
    with open(csv_path) as f:
        reader = csvmod.reader(f)
        header = next(reader)
        n_train_cols = len(header)
        for row in reader:
            if len(row) == n_train_cols:
                train_rows.append(row)
            else:
                val_rows.append(row)

    train = pd.DataFrame(train_rows, columns=header)
    for c in train.columns:
        train[c] = pd.to_numeric(train[c], errors='coerce')

    if val_rows:
        n_val_cols = len(val_rows[0])
        val_header = VAL_COLUMNS[:n_val_cols]
        val = pd.DataFrame(val_rows, columns=val_header)
        for c in val.columns:
            val[c] = pd.to_numeric(val[c], errors='coerce')
    else:
        val = pd.DataFrame()

    return train, val


def rolling(series, window=None):
    if window is None:
        window = max(3, len(series) // 20)
    return series.rolling(window, min_periods=1).mean()


def plot_training(train, val, output_path):
    fig, axes = plt.subplots(4, 4, figsize=(26, 22))
    fig.suptitle('Phantom v4 — Multi-Horizon Curve Pretraining (1-30 days)',
                 fontsize=16, y=0.98)

    steps = train['step'].values
    w = max(3, len(steps) // 20)

    # ── Row 1: Core losses ──
    ax = axes[0, 0]
    ax.plot(steps, train['loss'], 'b-', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['loss'], w), 'b-', lw=2, label='Total')
    if 'loss_main' in train.columns:
        ax.plot(steps, rolling(train['loss_main'], w), 'r-', lw=2, label='Main')
    ax.set_ylabel('Loss'); ax.set_title('Total Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 1]
    ax.plot(steps, train['nll'], color='#4CAF50', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['nll'], w), color='#4CAF50', lw=2, label='train')
    if len(val) > 0 and 'val_nll' in val.columns:
        ax.scatter(val['step'], val['val_nll'], c='red', s=20, zorder=5, label='val')
    ax.legend(fontsize=8)
    ax.set_ylabel('NLL'); ax.set_title('NLL (all 30 horizons)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 2]
    ax.plot(steps, train['ed'], color='#2196F3', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['ed'], w), color='#2196F3', lw=2, label='train')
    if len(val) > 0 and 'val_crps' in val.columns:
        ax.scatter(val['step'], val['val_crps'], c='red', s=20, zorder=5, label='val')
    ax.legend(fontsize=8)
    ax.set_ylabel('CRPS'); ax.set_title('CRPS (7 horizon subset)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ★ KEY V4 METRIC: Mean MSE
    ax = axes[0, 3]
    if 'mean_mse' in train.columns:
        ax.plot(steps, train['mean_mse'], color='#E91E63', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['mean_mse'], w), color='#E91E63', lw=2, label='train')
    if len(val) > 0 and 'val_mean_mse' in val.columns:
        ax.scatter(val['step'], val['val_mean_mse'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('MSE'); ax.set_title('Mean MSE (sqrt(h)-weighted)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ── Row 2: Auxiliary + key v4 metric ──

    # ★ KEY V4 METRIC: pred_mean_std
    ax = axes[1, 0]
    if 'pred_mean_std' in train.columns:
        ax.plot(steps, train['pred_mean_std'], color='#9C27B0', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['pred_mean_std'], w), color='#9C27B0', lw=2, label='train')
    if len(val) > 0 and 'val_pred_mean_std' in val.columns:
        ax.scatter(val['step'], val['val_pred_mean_std'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.axhline(0.001, color='gray', ls='--', alpha=0.5, label='v3 level (0.001)')
    ax.legend(fontsize=8)
    ax.set_ylabel('Std of pred mean'); ax.set_title('Pred Mean Std (is mu alive?)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 1]
    ax.plot(steps, train['sde_acc'] * 100, color='#FF9800', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['sde_acc'] * 100, w), color='#FF9800', lw=2)
    ax.axhline(25, color='red', ls='--', alpha=0.5, label='Random (25%)')
    if len(val) > 0 and 'val_asset_acc' in val.columns:
        ax.scatter(val['step'], val['val_asset_acc'] * 100, c='red', s=20, zorder=5, label='val')
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Asset-Type Classification')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 2]
    ax.plot(steps, train['vol_mse'], color='#00BCD4', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['vol_mse'], w), color='#00BCD4', lw=2)
    if len(val) > 0 and 'val_vol_mse' in val.columns:
        ax.scatter(val['step'], val['val_vol_mse'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('MSE'); ax.set_title('Volatility Regression MSE')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 3]
    ax.plot(steps, train['lr'], 'k-', lw=1.5)
    ax.set_ylabel('LR'); ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # ── Row 3: Head statistics ──
    ax = axes[2, 0]
    ax.plot(steps, train['mean_mu'], color='#2196F3', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['mean_mu'], w), color='#2196F3', lw=2)
    ax.set_ylabel('|mu| mean'); ax.set_title('Mean |Component Location|')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 1]
    ax.plot(steps, train['mean_sigma'], color='#F44336', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['mean_sigma'], w), color='#F44336', lw=2)
    if len(val) > 0 and 'val_mean_sigma' in val.columns:
        ax.scatter(val['step'], val['val_mean_sigma'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('sigma'); ax.set_title('Mean Component Scale')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 2]
    has_nu = 'mean_nu' in train.columns and train['mean_nu'].notna().any()
    if has_nu:
        ax.plot(steps, train['mean_nu'], color='#4CAF50', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['mean_nu'], w), color='#4CAF50', lw=2)
        ax.axhline(30, color='gray', ls='--', alpha=0.5, label='~Gaussian')
        ax.axhline(5, color='red', ls='--', alpha=0.5, label='heavy tails')
        ax.axhline(2.01, color='darkred', ls='--', alpha=0.5, label='min')
        ax.legend(fontsize=8)
    ax.set_ylabel('nu'); ax.set_title('Student-t Degrees of Freedom')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 3]
    if 'loss_aux' in train.columns:
        ax.plot(steps, rolling(train['loss_aux'], w), color='purple', lw=2, label='Aux total')
        ax.plot(steps, rolling(train['loss_sde'], w), color='#FF9800', lw=2, label='Asset CE')
        ax.plot(steps, rolling(train['vol_mse'], w), color='#00BCD4', lw=2, label='Vol MSE')
        ax.legend(fontsize=8)
    ax.set_ylabel('Loss'); ax.set_title('Auxiliary Losses')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ── Row 4: Training dynamics + validation + summary ──
    ax = axes[3, 0]
    ax.plot(steps, train['grad_norm'], color='#9C27B0', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['grad_norm'], w), color='#9C27B0', lw=2)
    ax.set_ylabel('Grad Norm'); ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[3, 1]
    if 'steps/s' in train.columns:
        ax.plot(steps, train['steps/s'], 'k-', lw=1.5)
    ax.set_ylabel('Steps/s'); ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # Validation loss
    ax = axes[3, 2]
    if len(val) > 0 and 'val_loss' in val.columns:
        ax.plot(val['step'], val['val_loss'], 'ro-', ms=4, lw=1.5, label='val_loss')
        if 'val_crps' in val.columns:
            ax.plot(val['step'], val['val_crps'], 'bs-', ms=4, lw=1.5, label='val_crps')
        if 'val_mean_mse' in val.columns:
            ax.plot(val['step'], val['val_mean_mse'], 'g^-', ms=4, lw=1.5, label='val_mean_mse')
        best_idx = val['val_loss'].idxmin()
        best_val = val.loc[best_idx, 'val_loss']
        best_step = val.loc[best_idx, 'step']
        ax.axhline(best_val, color='green', ls='--', alpha=0.5,
                   label=f'Best: {best_val:.4f} @ {int(best_step)}')
        ax.legend(fontsize=7)
    ax.set_ylabel('Loss'); ax.set_title('Validation Metrics')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # Summary
    ax = axes[3, 3]
    ax.axis('off')
    last = train.iloc[-1]
    total_steps = 40500
    pct = last['step'] / total_steps * 100
    eta_hrs = last.get('eta_min', 0) / 60

    lines = [
        f"Step: {int(last['step']):,} / {total_steps:,} ({pct:.1f}%)",
        f"Epoch: {int(last['epoch'])}",
    ]
    if 'steps/s' in last and last['steps/s'] > 0:
        lines.append(f"Throughput: {last['steps/s']:.1f} steps/s")
    if eta_hrs > 0:
        lines.append(f"ETA: {eta_hrs:.1f} hrs")
    lines += [
        "",
        "--- Latest Training ---",
        f"NLL:              {last['nll']:.4f}",
        f"CRPS:             {last['ed']:.6f}",
        f"Mean MSE:         {last.get('mean_mse', 0):.6f}",
        f"Pred Mean Std:    {last.get('pred_mean_std', 0):.6f}",
        f"Asset Accuracy:   {last['sde_acc']*100:.1f}%",
        f"Vol MSE:          {last['vol_mse']:.4f}",
    ]
    if has_nu:
        lines.append(f"Mean nu:          {last['mean_nu']:.1f}")
    lines.append(f"Grad Norm:        {last['grad_norm']:.1f}")

    if len(val) > 0:
        lv = val.iloc[-1]
        lines += [
            "",
            "--- Latest Validation ---",
        ]
        if 'val_nll' in lv:
            lines.append(f"Val NLL:          {lv['val_nll']:.4f}")
        if 'val_crps' in lv:
            lines.append(f"Val CRPS:         {lv['val_crps']:.6f}")
        if 'val_mean_mse' in lv:
            lines.append(f"Val Mean MSE:     {lv['val_mean_mse']:.6f}")
        if 'val_pred_mean_std' in lv:
            lines.append(f"Val PredMeanStd:  {lv['val_pred_mean_std']:.6f}")
        if 'val_asset_acc' in lv:
            lines.append(f"Val Asset Acc:    {lv['val_asset_acc']*100:.1f}%")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Phantom v4 pre-training metrics")
    parser.add_argument('--log', type=str, default='logs/v4/train_log.csv')
    parser.add_argument('--output', type=str, default='plots/pretrain_v4_live.png')
    args = parser.parse_args()

    train, val = load_and_split(args.log)
    print(f"Loaded {len(train)} training rows, {len(val)} validation rows")
    if len(train) > 0:
        print(f"Steps: {int(train['step'].iloc[0])} to {int(train['step'].iloc[-1])}")
    plot_training(train, val, args.output)


if __name__ == "__main__":
    main()
