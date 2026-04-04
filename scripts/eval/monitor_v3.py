#!/usr/bin/env python
"""
Pull v3 training logs from LaRuche and plot live metrics.

Fetches the latest train_log.csv from the HPC cluster, then runs
plot_pretrain_v3.py to generate an updated metrics plot.

Usage:
  python scripts/eval/monitor_v3.py
  python scripts/eval/monitor_v3.py --job_dir logs/v3
  python scripts/eval/monitor_v3.py --slurm_log logs/v3_498017.out
"""

import argparse
import subprocess
import sys
from pathlib import Path


REMOTE = "dalbanal@ruche.mesocentre.universite-paris-saclay.fr"
REMOTE_PROJECT = "/gpfs/workdir/dalbanal/phantom"


def scp(remote_path, local_path):
    """Download a file from LaRuche."""
    cmd = f"scp {REMOTE}:{remote_path} {local_path}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  SCP failed: {r.stderr.strip()}")
        return False
    return True


def ssh(command):
    """Run a command on LaRuche and return stdout."""
    cmd = f'ssh {REMOTE} "{command}"'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Monitor v3 training on LaRuche")
    parser.add_argument('--job_dir', type=str, default='logs/v3',
                        help='Remote log directory (relative to project)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v3',
                        help='Remote checkpoint directory')
    parser.add_argument('--slurm_log', type=str, default=None,
                        help='SLURM .out log to show (e.g. logs/v3_498017.out)')
    parser.add_argument('--output', type=str, default='plots/pretrain_v3_live.png',
                        help='Local output plot path')
    args = parser.parse_args()

    local_log = Path('logs/v3/train_log.csv')
    local_log.parent.mkdir(parents=True, exist_ok=True)

    # Check job status
    print("=== SLURM Job Status ===")
    status = ssh("squeue -u dalbanal --format='%.10i %.9P %.12j %.8T %.10M %.6D %R'")
    print(status if status else "  No running jobs")

    # Pull training log
    remote_log = f"{REMOTE_PROJECT}/{args.job_dir}/train_log.csv"
    print(f"\n=== Pulling {remote_log} ===")
    if scp(remote_log, str(local_log)):
        import pandas as pd
        df = pd.read_csv(local_log)
        train = df[df['grad_norm'].notna()] if 'grad_norm' in df.columns else df
        val = df[df['grad_norm'].isna()] if 'grad_norm' in df.columns else pd.DataFrame()
        print(f"  {len(train)} training rows, {len(val)} validation rows")
        if len(train) > 0:
            last = train.iloc[-1]
            print(f"  Latest step: {int(last['step'])}")
            print(f"  NLL: {last['nll']:.4f} | CRPS: {last['ed']:.6f}")
            if 'sde_acc' in last:
                print(f"  Asset Acc: {last['sde_acc']*100:.1f}%")
        if len(val) > 0:
            best_idx = val['val_loss'].idxmin() if 'val_loss' in val.columns else 0
            best = val.iloc[best_idx]
            print(f"  Best val_loss: {best['val_loss']:.4f} @ step {int(best['step'])}")
    else:
        print("  Could not pull log — training may not have started yet")
        return

    # Check for checkpoints
    print(f"\n=== Checkpoints ===")
    ckpts = ssh(f"ls -lh {REMOTE_PROJECT}/{args.checkpoint_dir}/*.pt 2>/dev/null")
    print(ckpts if ckpts else "  No checkpoints yet")

    # Pull SLURM output log (last 30 lines)
    if args.slurm_log:
        print(f"\n=== SLURM Output (last 30 lines) ===")
        output = ssh(f"tail -30 {REMOTE_PROJECT}/{args.slurm_log} 2>/dev/null")
        print(output if output else "  Log not found")
    else:
        # Auto-detect latest slurm log
        latest = ssh(f"ls -t {REMOTE_PROJECT}/logs/v3_*.out 2>/dev/null | head -1")
        if latest:
            print(f"\n=== SLURM Output: {Path(latest).name} (last 20 lines) ===")
            output = ssh(f"tail -20 {latest}")
            print(output)

    # Plot metrics
    print(f"\n=== Generating plot ===")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, 'scripts/eval/plot_pretrain_v3.py',
        '--log', str(local_log),
        '--output', args.output,
    ])

    print(f"\nDone. Plot saved to {args.output}")


if __name__ == '__main__':
    main()
