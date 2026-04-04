#!/usr/bin/env python
"""
Plot Phantom v3 real-data pre-training metrics from the CSV log.

v3 trains on real multi-asset OHLCV data with NLL+CRPS loss (no branches).
Key metrics: CRPS, NLL, asset-type accuracy, sign accuracy, realized vol MSE.

Usage:
  python scripts/eval/plot_pretrain_v3.py --log logs/v3/train_log.csv
  python scripts/eval/plot_pretrain_v3.py --log logs/v3/train_log.csv --output plots/pretrain_v3_live.png
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
    if 'grad_norm' in df.columns:
        train = df[df['grad_norm'].notna()].copy().reset_index(drop=True)
        val = df[df['grad_norm'].isna()].copy().reset_index(drop=True)
    else:
        train = df.copy()
        val = pd.DataFrame()
    return train, val


def rolling(series, window=None):
    """Compute rolling mean with auto window size."""
    if window is None:
        window = max(3, len(series) // 20)
    return series.rolling(window, min_periods=1).mean()


def plot_training(train, val, output_path, title_suffix=''):
    has_nu = 'mean_nu' in train.columns and train['mean_nu'].notna().any()

    fig, axes = plt.subplots(4, 4, figsize=(26, 22))
    fig.suptitle(f'Phantom v3 — Real Multi-Asset Pretraining{title_suffix}',
                 fontsize=16, y=0.98)

    steps = train['step'].values
    w = max(3, len(steps) // 20)

    # ── Row 1: Core losses ──
    ax = axes[0, 0]
    ax.plot(steps, train['loss'], 'b-', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['loss'], w), 'b-', lw=2, label='Total')
    if 'loss_main' in train.columns:
        ax.plot(steps, rolling(train['loss_main'], w), 'r-', lw=2, label='Main (NLL+CRPS)')
    ax.set_ylabel('Loss'); ax.set_title('Total Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 1]
    ax.plot(steps, train['nll'], color='#4CAF50', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['nll'], w), color='#4CAF50', lw=2)
    if len(val) > 0 and 'val_nll' in val.columns:
        ax.scatter(val['step'], val['val_nll'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('NLL'); ax.set_title('NLL (primary loss)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 2]
    # 'ed' column stores CRPS in v3 mode
    ax.plot(steps, train['ed'], color='#2196F3', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['ed'], w), color='#2196F3', lw=2)
    if len(val) > 0 and 'val_crps' in val.columns:
        ax.scatter(val['step'], val['val_crps'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('CRPS'); ax.set_title('CRPS (secondary loss)')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[0, 3]
    if 'loss_aux' in train.columns:
        ax.plot(steps, rolling(train['loss_aux'], w), color='purple', lw=2, label='Aux total')
    if 'loss_sde' in train.columns:
        ax.plot(steps, rolling(train['loss_sde'], w), color='#FF9800', lw=2, label='Asset CE')
    if 'vol_mse' in train.columns:
        ax.plot(steps, rolling(train['vol_mse'], w), color='#00BCD4', lw=2, label='Vol MSE')
    ax.legend(fontsize=8)
    ax.set_ylabel('Loss'); ax.set_title('Auxiliary Losses')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ── Row 2: Auxiliary tasks ──
    ax = axes[1, 0]
    ax.plot(steps, train['sde_acc'] * 100, color='#FF9800', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['sde_acc'] * 100, w), color='#FF9800', lw=2)
    ax.axhline(25, color='red', ls='--', alpha=0.5, label='Random (25%)')
    if len(val) > 0 and 'val_asset_acc' in val.columns:
        ax.scatter(val['step'], val['val_asset_acc'] * 100, c='red', s=20, zorder=5, label='val')
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Asset-Type Classification')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 1]
    ax.plot(steps, train['vol_mse'], color='#00BCD4', lw=0.8, alpha=0.4)
    ax.plot(steps, rolling(train['vol_mse'], w), color='#00BCD4', lw=2)
    if len(val) > 0 and 'val_vol_mse' in val.columns:
        ax.scatter(val['step'], val['val_vol_mse'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('MSE'); ax.set_title('Volatility Regression MSE')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 2]
    if len(val) > 0 and 'val_sign_acc' in val.columns:
        ax.scatter(val['step'], val['val_sign_acc'] * 100, c='#E91E63', s=30, zorder=5)
    ax.axhline(50, color='red', ls='--', alpha=0.5, label='Random (50%)')
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Return-Sign Classification (val)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[1, 3]
    ax.plot(steps, train['lr'], 'k-', lw=1.5)
    ax.set_ylabel('Learning Rate'); ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # ── Row 3: Head statistics ──
    ax = axes[2, 0]
    if 'mean_mu' in train.columns:
        ax.plot(steps, train['mean_mu'], color='#2196F3', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['mean_mu'], w), color='#2196F3', lw=2)
    ax.set_ylabel('|mu| mean'); ax.set_title('Mean |Component Location|')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 1]
    if 'mean_sigma' in train.columns:
        ax.plot(steps, train['mean_sigma'], color='#F44336', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['mean_sigma'], w), color='#F44336', lw=2)
    if len(val) > 0 and 'val_mean_sigma' in val.columns:
        ax.scatter(val['step'], val['val_mean_sigma'], c='red', s=20, zorder=5, label='val')
        ax.legend(fontsize=8)
    ax.set_ylabel('sigma'); ax.set_title('Mean Component Scale')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 2]
    if has_nu:
        ax.plot(steps, train['mean_nu'], color='#4CAF50', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['mean_nu'], w), color='#4CAF50', lw=2)
        ax.axhline(30, color='gray', ls='--', alpha=0.5, label='nu=30 (~Gaussian)')
        ax.axhline(5, color='red', ls='--', alpha=0.5, label='nu=5 (heavy tails)')
        ax.axhline(2.01, color='darkred', ls='--', alpha=0.5, label='nu=2 (min)')
        ax.legend(fontsize=8)
    ax.set_ylabel('nu (df)'); ax.set_title('Mean Student-t Degrees of Freedom')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[2, 3]
    if 'eff_k' in train.columns:
        ax.plot(steps, train['eff_k'], color='#009688', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['eff_k'], w), color='#009688', lw=2)
        ax.axhline(1, color='red', ls='--', alpha=0.5, label='K=1 (Student-t head)')
        ax.legend(fontsize=8)
    ax.set_ylabel('Effective K'); ax.set_title('Effective Component Count')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # ── Row 4: Training dynamics + validation + summary ──
    ax = axes[3, 0]
    if 'grad_norm' in train.columns:
        ax.plot(steps, train['grad_norm'], color='#9C27B0', lw=0.8, alpha=0.4)
        ax.plot(steps, rolling(train['grad_norm'], w), color='#9C27B0', lw=2)
    ax.set_ylabel('Grad Norm'); ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    ax = axes[3, 1]
    if 'steps/s' in train.columns:
        ax.plot(steps, train['steps/s'], 'k-', lw=1.5)
    ax.set_ylabel('Steps/s'); ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # Validation CRPS over time
    ax = axes[3, 2]
    if len(val) > 0 and 'val_loss' in val.columns:
        ax.plot(val['step'], val['val_loss'], 'ro-', ms=4, lw=1.5, label='val_loss')
        if 'val_crps' in val.columns:
            ax.plot(val['step'], val['val_crps'], 'bs-', ms=4, lw=1.5, label='val_crps')
        best_idx = val['val_loss'].idxmin()
        best_step = val.loc[best_idx, 'step']
        best_val = val.loc[best_idx, 'val_loss']
        ax.axhline(best_val, color='green', ls='--', alpha=0.5,
                   label=f'Best: {best_val:.4f} @ {int(best_step)}')
        ax.legend(fontsize=8)
    ax.set_ylabel('Loss'); ax.set_title('Validation Loss')
    ax.grid(True, alpha=0.3); ax.set_xlabel('Step')

    # Summary panel
    ax = axes[3, 3]
    ax.axis('off')
    last = train.iloc[-1]
    total_steps = int(last.get('eta_min', 0) / 60 * last.get('steps/s', 1) * 3600 + last['step']) if last.get('steps/s', 0) > 0 and last.get('eta_min', 0) > 0 else 60980
    pct = last['step'] / total_steps * 100
    eta_hrs = last.get('eta_min', 0) / 60

    lines = [
        f"Step: {int(last['step']):,} / ~{total_steps:,} ({pct:.1f}%)",
        f"Epoch: {int(last['epoch'])}",
    ]
    if 'steps/s' in last and last['steps/s'] > 0:
        lines.append(f"Throughput: {last['steps/s']:.1f} steps/s")
    if eta_hrs > 0:
        lines.append(f"ETA: {eta_hrs:.1f} hrs")
    lines.append("")
    lines.append("─── Latest Training ───")
    lines.append(f"NLL:              {last['nll']:.4f}")
    lines.append(f"CRPS:             {last['ed']:.6f}")
    lines.append(f"Asset Accuracy:   {last['sde_acc']*100:.1f}%")
    lines.append(f"Vol MSE:          {last['vol_mse']:.4f}")
    if has_nu:
        lines.append(f"Mean nu:          {last['mean_nu']:.1f}")
    lines.append(f"Grad Norm:        {last['grad_norm']:.2f}")

    if len(val) > 0:
        last_val = val.iloc[-1]
        lines.append("")
        lines.append("─── Latest Validation ───")
        if 'val_crps' in last_val:
            lines.append(f"Val CRPS:         {last_val['val_crps']:.6f}")
        if 'val_nll' in last_val:
            lines.append(f"Val NLL:          {last_val['val_nll']:.4f}")
        if 'val_asset_acc' in last_val:
            lines.append(f"Val Asset Acc:    {last_val['val_asset_acc']*100:.1f}%")
        if 'val_sign_acc' in last_val:
            lines.append(f"Val Sign Acc:     {last_val['val_sign_acc']*100:.1f}%")

    summary = "\n".join(lines)
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Phantom v3 pre-training metrics")
    parser.add_argument('--log', type=str, default='logs/v3/train_log.csv',
                        help='Path to training CSV log')
    parser.add_argument('--output', type=str, default='plots/pretrain_v3_live.png',
                        help='Output plot path')
    parser.add_argument('--title', type=str, default='',
                        help='Extra title suffix')
    args = parser.parse_args()

    train, val = load_and_split(args.log)
    print(f"Loaded {len(train)} training rows, {len(val)} validation rows")
    if len(train) > 0:
        print(f"Steps: {int(train['step'].iloc[0])} to {int(train['step'].iloc[-1])}")
    plot_training(train, val, args.output, title_suffix=args.title)


if __name__ == "__main__":
    main()
