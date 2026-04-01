#!/usr/bin/env python
"""
Synthetic pre-training script for Phantom.

Trains the transformer on synthetic SDE trajectories with JointFM-style
branched future paths and energy distance loss.

Usage:
  # On-the-fly generation (recommended):
  python train_pretrain.py --data_mode online --samples_per_epoch 1000000

  # From pre-generated shards:
  python train_pretrain.py --data_mode shards --data_dir data/

  # Resume from checkpoint:
  python train_pretrain.py --resume checkpoints/latest.pt
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
from src.losses import combined_loss, crps_loss
from src.data import ShardDataset, OnlineDataset, make_validation_batch


# ── Cosine schedule with linear warmup ──────────────────────────────

def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
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
    def __init__(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "train_log.csv"
        self._header_written = False

    def log(self, metrics, step, console=True):
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
def validate(model, val_x, val_h, val_yb, device,
             n_model_samples=256, nll_weight=0.1, batch_size=512):
    """Run validation on a fixed synthetic batch with branched futures."""
    model.eval()
    n = val_x.size(0)
    total_loss, total_ed, total_nll = 0.0, 0.0, 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        h = val_h[i:i+batch_size].to(device)
        yb = val_yb[i:i+batch_size].to(device)

        log_pi, mu, sigma = model(x, h)
        loss, ed, nll = combined_loss(log_pi, mu, sigma, yb,
                                      n_model_samples=n_model_samples,
                                      nll_weight=nll_weight)
        total_loss += loss.item()
        total_ed += ed.item()
        total_nll += nll.item()
        n_batches += 1

    model.train()
    return {
        'val_loss': total_loss / n_batches,
        'val_ed': total_ed / n_batches,
        'val_nll': total_nll / n_batches,
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
    p.add_argument('--data_mode', type=str, default='online', choices=['shards', 'online'])
    p.add_argument('--data_dir', type=str, default='data/')
    p.add_argument('--samples_per_epoch', type=int, default=1_000_000)
    p.add_argument('--n_workers', type=int, default=4)
    p.add_argument('--n_branches', type=int, default=128,
                   help='Branched future paths per sample (JointFM-style)')

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

    # Training
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--grad_accum', type=int, default=1)

    # Energy distance loss
    p.add_argument('--n_model_samples', type=int, default=256,
                   help='Samples from MoG for energy distance')
    p.add_argument('--nll_weight', type=float, default=0.1,
                   help='Weight for auxiliary NLL loss (gradient flow to pi)')

    # Mixed precision
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')

    # Logging & checkpointing
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--val_every', type=int, default=2000)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    p.add_argument('--log_dir', type=str, default='logs/')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--val_samples', type=int, default=2048)

    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)

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
    )

    model = PhantomModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | patches={cfg.n_patches} | K={cfg.n_components}")

    # ── Data ──
    if args.data_mode == 'shards':
        dataset = ShardDataset(args.data_dir, context_len=cfg.context_len)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.n_workers, pin_memory=device.type == 'cuda',
            drop_last=True, persistent_workers=args.n_workers > 0,
        )
    else:
        dataset = OnlineDataset(
            context_len=cfg.context_len,
            n_branches=args.n_branches,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            num_workers=args.n_workers, pin_memory=device.type == 'cuda',
            drop_last=True, persistent_workers=args.n_workers > 0,
        )

    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"Data: mode={args.data_mode} | {len(dataset):,} samples/epoch | {steps_per_epoch:,} steps/epoch")
    print(f"Training: {args.epochs} epochs | {total_steps:,} total steps | batch={args.batch_size}")
    print(f"Branches: {args.n_branches} | Model samples: {args.n_model_samples} | NLL weight: {args.nll_weight}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    # ── AMP ──
    use_amp = args.amp and device.type == 'cuda'
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            scaler = None
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
    start_step, start_epoch, best_val_loss = 0, 0, float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step, start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler)
        print(f"  Resumed at step {start_step}, epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Validation data ──
    print(f"Generating validation set ({args.val_samples} samples, {args.n_branches} branches)...")
    val_x, val_h, val_yb = make_validation_batch(
        n_samples=args.val_samples,
        context_len=cfg.context_len,
        n_branches=args.n_branches,
        seed=args.seed + 1234,
    )

    # ── Logging ──
    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    # ── Training loop ──
    model.train()
    global_step = start_step
    running_loss, running_ed, running_nll = 0.0, 0.0, 0.0
    running_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print("Starting pre-training (JointFM-style energy distance)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        if args.data_mode == 'online':
            dataset.seed = args.seed + epoch * 1000

        for batch_idx, (x, h, yb) in enumerate(loader):
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            x = x.to(device, non_blocking=True)
            h = h.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # ── Forward + loss ──
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                log_pi, mu, sigma = model(x, h)
                loss, ed, nll = combined_loss(
                    log_pi, mu, sigma, yb,
                    n_model_samples=args.n_model_samples,
                    nll_weight=args.nll_weight,
                )
                loss = loss / args.grad_accum

            # ── Backward ──
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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
            running_ed += ed.item()
            running_nll += nll.item()
            running_count += 1
            global_step += 1

            # ── Logging ──
            if global_step % args.log_every == 0:
                avg_loss = running_loss / running_count
                avg_ed = running_ed / running_count
                avg_nll = running_nll / running_count
                elapsed = time.time() - t_start
                sps = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / sps if sps > 0 else 0

                logger.log({
                    'epoch': epoch, 'lr': lr,
                    'loss': avg_loss, 'ed': avg_ed, 'nll': avg_nll,
                    'steps/s': sps, 'eta_min': eta / 60,
                }, step=global_step)

                running_loss, running_ed, running_nll, running_count = 0.0, 0.0, 0.0, 0

            # ── Validation ──
            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_x, val_h, val_yb, device,
                                       args.n_model_samples, args.nll_weight)
                logger.log(val_metrics, step=global_step)

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, optimizer, scaler,
                        global_step, epoch, cfg, best_val_loss)
                    print(f"  >> New best val_loss: {best_val_loss:.4f}")

            # ── Periodic checkpoint ──
            if global_step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / "latest.pt", model, optimizer, scaler,
                    global_step, epoch, cfg, best_val_loss)

        # ── End of epoch ──
        print(f"\n--- Epoch {epoch+1}/{args.epochs} complete (step {global_step}) ---\n")
        save_checkpoint(
            ckpt_dir / "latest.pt", model, optimizer, scaler,
            global_step, epoch + 1, cfg, best_val_loss)

    # ── Final ──
    save_checkpoint(
        ckpt_dir / "latest.pt", model, optimizer, scaler,
        global_step, args.epochs, cfg, best_val_loss)

    elapsed = time.time() - t_start
    print(f"\nTraining complete. {global_step:,} steps in {elapsed/60:.1f} minutes.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
