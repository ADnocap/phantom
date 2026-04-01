#!/usr/bin/env python
"""
Synthetic pre-training script for Phantom.

Trains the transformer on synthetic SDE trajectories (Phase 1 of training).
Supports two data modes:
  - shards: load pre-generated .npz files from disk
  - online: generate fresh SDE samples on-the-fly (JointFM-style infinite stream)

Usage:
  # From pre-generated shards:
  python train_pretrain.py --data_mode shards --data_dir data/

  # On-the-fly generation (no pre-generated data needed):
  python train_pretrain.py --data_mode online --samples_per_epoch 1000000

  # Resume from checkpoint:
  python train_pretrain.py --resume checkpoints/latest.pt

  # Custom model size:
  python train_pretrain.py --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import PhantomConfig, PhantomModel
from src.losses import combined_loss, quantile_loss
from src.data import ShardDataset, OnlineDataset, make_validation_batch


# ── Cosine schedule with linear warmup ──────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


# ── Logging ─────────────────────────────────────────────────────────

class Logger:
    """Simple CSV + console logger."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "train_log.csv"
        self._header_written = False

    def log(self, metrics: dict, step: int, console: bool = True):
        if not self._header_written:
            with open(self.csv_path, 'w') as f:
                f.write("step," + ",".join(metrics.keys()) + "\n")
            self._header_written = True

        with open(self.csv_path, 'a') as f:
            vals = ",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in metrics.values())
            f.write(f"{step},{vals}\n")

        if console:
            parts = [f"step {step:>7d}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts))


# ── Validation ──────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_x, val_h, val_y, device, alpha,
             entropy_coeff=0.01, sharpness_coeff=0.1, batch_size=512):
    """Run validation on a fixed synthetic batch."""
    model.eval()
    head_type = model.cfg.head_type
    n = val_x.size(0)
    total_loss, total_nll, total_crps = 0.0, 0.0, 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        h = val_h[i:i+batch_size].to(device)
        y = val_y[i:i+batch_size].to(device)

        if head_type == 'quantile':
            q_pred = model(x, h)
            loss = quantile_loss(q_pred, y, model.cfg.quantiles)
            total_loss += loss.item()
            total_nll += loss.item()
            total_crps += loss.item()
        else:
            log_pi, mu, sigma = model(x, h)
            loss, nll, crps = combined_loss(log_pi, mu, sigma, y, alpha=alpha,
                                            entropy_coeff=entropy_coeff,
                                            sharpness_coeff=sharpness_coeff)
            total_loss += loss.item()
            total_nll += nll.item()
            total_crps += crps.item()
        n_batches += 1

    model.train()
    return {
        'val_loss': total_loss / n_batches,
        'val_nll': total_nll / n_batches,
        'val_crps': total_crps / n_batches,
    }


# ── Checkpoint ──────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scaler, step, epoch, config, best_val_loss):
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'config': vars(config),
        'best_val_loss': best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and ckpt['scaler_state_dict'] is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['step'], ckpt['epoch'], ckpt.get('best_val_loss', float('inf'))


# ── Main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phantom synthetic pre-training")

    # Data
    p.add_argument('--data_mode', type=str, default='online', choices=['shards', 'online'],
                   help='Data source: shards (pre-generated) or online (on-the-fly)')
    p.add_argument('--data_dir', type=str, default='data/',
                   help='Directory with .npz shards (for data_mode=shards)')
    p.add_argument('--samples_per_epoch', type=int, default=1_000_000,
                   help='Samples per epoch for online mode')
    p.add_argument('--n_workers', type=int, default=4,
                   help='DataLoader workers')

    # Model architecture
    p.add_argument('--context_len', type=int, default=60)
    p.add_argument('--patch_len', type=int, default=5)
    p.add_argument('--patch_stride', type=int, default=5)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--n_components', type=int, default=5)
    p.add_argument('--head_type', type=str, default='mog', choices=['mog', 'quantile'],
                   help='Forecast head: mog (Mixture of Gaussians) or quantile')

    # Training
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4, help='Peak learning rate')
    p.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR at end of cosine decay')
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=2.0, help='CRPS weight in combined loss (>=1 for CRPS-dominant)')
    p.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy regularization on mixture weights')
    p.add_argument('--sharpness_coeff', type=float, default=0.1, help='Sharpness penalty on predicted variance')
    p.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')

    # Mixed precision
    p.add_argument('--amp', action='store_true', default=True,
                   help='Use automatic mixed precision')
    p.add_argument('--no_amp', action='store_true', help='Disable AMP')

    # Logging & checkpointing
    p.add_argument('--log_every', type=int, default=100, help='Log every N steps')
    p.add_argument('--val_every', type=int, default=2000, help='Validate every N steps')
    p.add_argument('--save_every', type=int, default=5000, help='Checkpoint every N steps')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    p.add_argument('--log_dir', type=str, default='logs/')
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    p.add_argument('--val_samples', type=int, default=4096, help='Validation set size')

    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None, help='Device (auto-detected if omitted)')

    return p.parse_args()


def main():
    args = parse_args()

    if args.no_amp:
        args.amp = False

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Seed ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Model config ──
    cfg = PhantomConfig(
        context_len=args.context_len,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_components=args.n_components,
        head_type=args.head_type,
    )

    model = PhantomModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | patches={cfg.n_patches} | K={cfg.n_components}")

    # ── Data ──
    if args.data_mode == 'shards':
        dataset = ShardDataset(args.data_dir, context_len=cfg.context_len)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=device.type == 'cuda',
            drop_last=True,
            persistent_workers=args.n_workers > 0,
        )
    else:
        dataset = OnlineDataset(
            context_len=cfg.context_len,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            pin_memory=device.type == 'cuda',
            drop_last=True,
            persistent_workers=args.n_workers > 0,
        )

    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"Data: mode={args.data_mode} | {len(dataset):,} samples/epoch | {steps_per_epoch:,} steps/epoch")
    print(f"Training: {args.epochs} epochs | {total_steps:,} total steps | batch={args.batch_size}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── AMP scaler ──
    use_amp = args.amp and device.type == 'cuda'
    # Determine amp dtype: bf16 if supported, else fp16
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            scaler = None  # bf16 doesn't need gradient scaling
            print("AMP: bfloat16")
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler('cuda')
            print("AMP: float16 with GradScaler")
    else:
        amp_dtype = torch.float32
        scaler = None
        print("AMP: disabled")

    # ── Resume ──
    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step, start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler
        )
        print(f"  Resumed at step {start_step}, epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Validation data (fixed, for consistent tracking) ──
    print(f"Generating validation set ({args.val_samples} samples)...")
    val_x, val_h, val_y = make_validation_batch(
        n_samples=args.val_samples,
        context_len=cfg.context_len,
        seed=args.seed + 1234,
    )

    # ── Logging ──
    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(ckpt_dir / "config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    # ── Training loop ──
    model.train()
    global_step = start_step
    running_loss = 0.0
    running_nll = 0.0
    running_crps = 0.0
    running_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print("Starting pre-training")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        # For online mode: re-seed each epoch so workers produce different data
        if args.data_mode == 'online':
            dataset.seed = args.seed + epoch * 1000

        for batch_idx, (x, h, y) in enumerate(loader):
            # ── LR schedule ──
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            x = x.to(device, non_blocking=True)
            h = h.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ── Forward + loss ──
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if args.head_type == 'quantile':
                    q_pred = model(x, h)
                    loss = quantile_loss(q_pred, y, cfg.quantiles)
                    nll = loss
                    crps = loss
                else:
                    log_pi, mu, sigma = model(x, h)
                    loss, nll, crps = combined_loss(log_pi, mu, sigma, y, alpha=args.alpha,
                                                    entropy_coeff=args.entropy_coeff,
                                                    sharpness_coeff=args.sharpness_coeff)
                loss = loss / args.grad_accum

            # ── Backward ──
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ── Optimizer step (every grad_accum steps) ──
            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            # ── Tracking ──
            running_loss += loss.item() * args.grad_accum
            running_nll += nll.item()
            running_crps += crps.item()
            running_count += 1
            global_step += 1

            # ── Console logging ──
            if global_step % args.log_every == 0:
                avg_loss = running_loss / running_count
                avg_nll = running_nll / running_count
                avg_crps = running_crps / running_count
                elapsed = time.time() - t_start
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0

                logger.log({
                    'epoch': epoch,
                    'lr': lr,
                    'loss': avg_loss,
                    'nll': avg_nll,
                    'crps': avg_crps,
                    'steps/s': steps_per_sec,
                    'eta_min': eta / 60,
                }, step=global_step)

                running_loss = 0.0
                running_nll = 0.0
                running_crps = 0.0
                running_count = 0

            # ── Validation ──
            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_x, val_h, val_y, device, args.alpha,
                                      args.entropy_coeff, args.sharpness_coeff)
                logger.log(val_metrics, step=global_step)

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, optimizer, scaler,
                        global_step, epoch, cfg, best_val_loss,
                    )
                    print(f"  >> New best val_loss: {best_val_loss:.4f}")

            # ── Periodic checkpoint ──
            if global_step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / "latest.pt", model, optimizer, scaler,
                    global_step, epoch, cfg, best_val_loss,
                )

        # ── End of epoch ──
        print(f"\n--- Epoch {epoch+1}/{args.epochs} complete (step {global_step}) ---\n")

        # Save latest at end of epoch
        save_checkpoint(
            ckpt_dir / "latest.pt", model, optimizer, scaler,
            global_step, epoch + 1, cfg, best_val_loss,
        )

    # ── Final → save as latest ──
    save_checkpoint(
        ckpt_dir / "latest.pt", model, optimizer, scaler,
        global_step, args.epochs, cfg, best_val_loss,
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete. {global_step:,} steps in {elapsed/60:.1f} minutes.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
