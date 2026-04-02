#!/usr/bin/env python
"""
Synthetic pre-training for Phantom with three anti-collapse mechanisms:
  1. Auxiliary tasks (SDE type classifier + vol regressor)
  2. Condition dropout (classifier-free guidance training)
  3. Cross-attention decoder (architectural conditioning)

Usage:
  python train_pretrain.py --data_mode online --samples_per_epoch 1000000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import PhantomConfig, PhantomModel
from src.losses import combined_loss
from src.data import OnlineDataset, make_validation_batch


def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


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


@torch.no_grad()
def validate(model, val_x, val_h, val_yb, val_sde, val_rv, device,
             n_model_samples=256, nll_weight=0.1, batch_size=512):
    model.eval()
    n = val_x.size(0)
    total_loss, total_ed, total_nll = 0.0, 0.0, 0.0
    total_sde_acc, total_vol_mse = 0.0, 0.0
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        h = val_h[i:i+batch_size].to(device)
        yb = val_yb[i:i+batch_size].to(device)
        sde = val_sde[i:i+batch_size].to(device)
        rv = val_rv[i:i+batch_size].to(device)

        log_pi, mu, sigma = model(x, h)
        loss, ed, nll = combined_loss(log_pi, mu, sigma, yb,
                                      n_model_samples=n_model_samples,
                                      nll_weight=nll_weight)

        # Auxiliary metrics
        sde_logits, vol_pred = model.forward_auxiliary(x)
        sde_acc = (sde_logits.argmax(dim=-1) == sde).float().mean()
        vol_mse = F.mse_loss(vol_pred, rv)

        total_loss += loss.item()
        total_ed += ed.item()
        total_nll += nll.item()
        total_sde_acc += sde_acc.item()
        total_vol_mse += vol_mse.item()
        n_batches += 1

    model.train()
    return {
        'val_loss': total_loss / n_batches,
        'val_ed': total_ed / n_batches,
        'val_nll': total_nll / n_batches,
        'val_sde_acc': total_sde_acc / n_batches,
        'val_vol_mse': total_vol_mse / n_batches,
    }


def save_checkpoint(path, model, optimizer, scaler, step, epoch, config, best_val_loss):
    torch.save({
        'step': step, 'epoch': epoch,
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


def parse_args():
    p = argparse.ArgumentParser(description="Phantom pre-training")

    p.add_argument('--data_mode', type=str, default='online', choices=['shards', 'online'])
    p.add_argument('--data_dir', type=str, default='data/')
    p.add_argument('--samples_per_epoch', type=int, default=1_000_000)
    p.add_argument('--n_workers', type=int, default=4)
    p.add_argument('--n_branches', type=int, default=128)

    p.add_argument('--context_len', type=int, default=60)
    p.add_argument('--patch_len', type=int, default=5)
    p.add_argument('--patch_stride', type=int, default=5)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--n_components', type=int, default=5)
    p.add_argument('--n_decoder_layers', type=int, default=2)
    p.add_argument('--cond_drop_prob', type=float, default=0.15)

    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--grad_accum', type=int, default=1)

    p.add_argument('--n_model_samples', type=int, default=256)
    p.add_argument('--nll_weight', type=float, default=0.1)
    p.add_argument('--aux_weight', type=float, default=0.5,
                   help='Weight for auxiliary losses (SDE classification + vol regression)')

    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')

    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--val_every', type=int, default=2000)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    p.add_argument('--log_dir', type=str, default='logs/pretrain/')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--val_samples', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    if args.no_amp:
        args.amp = False

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = PhantomConfig(
        context_len=args.context_len, patch_len=args.patch_len,
        patch_stride=args.patch_stride, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        dropout=args.dropout, n_components=args.n_components,
        n_decoder_layers=args.n_decoder_layers,
        cond_drop_prob=args.cond_drop_prob,
    )

    model = PhantomModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | patches={cfg.n_patches} | K={cfg.n_components}")
    print(f"Decoder layers: {cfg.n_decoder_layers} | Cond drop: {cfg.cond_drop_prob}")

    # Data
    dataset = OnlineDataset(
        context_len=cfg.context_len, n_branches=args.n_branches,
        samples_per_epoch=args.samples_per_epoch, seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.n_workers,
        pin_memory=device.type == 'cuda', drop_last=True,
        persistent_workers=args.n_workers > 0,
    )

    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"Data: {len(dataset):,} samples/epoch | {steps_per_epoch:,} steps/epoch")
    print(f"Training: {args.epochs} epochs | {total_steps:,} total steps")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    use_amp = args.amp and device.type == 'cuda'
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
        print("AMP: bfloat16")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
        print("AMP: float16")
    else:
        amp_dtype = torch.float32
        scaler = None
        use_amp = False

    start_step, start_epoch, best_val_loss = 0, 0, float('inf')
    if args.resume:
        start_step, start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler)
        print(f"Resumed at step {start_step}, epoch {start_epoch}")

    print(f"Generating validation set ({args.val_samples} samples, {args.n_branches} branches)...")
    val_x, val_h, val_yb, val_sde, val_rv = make_validation_batch(
        n_samples=args.val_samples, context_len=cfg.context_len,
        n_branches=args.n_branches, seed=args.seed + 1234,
    )

    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    model.train()
    global_step = start_step
    run_loss, run_ed, run_nll, run_sde_acc, run_vol = 0., 0., 0., 0., 0.
    run_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print("Starting pre-training (cross-attn + aux tasks + cond dropout)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        dataset.seed = args.seed + epoch * 1000

        for batch_idx, (x, h, yb, sde_labels, rv_labels) in enumerate(loader):
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            x = x.to(device, non_blocking=True)
            h = h.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sde_labels = sde_labels.to(device, non_blocking=True)
            rv_labels = rv_labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Main loss: energy distance + aux NLL
                log_pi, mu, sigma = model(x, h)
                loss_main, ed, nll = combined_loss(
                    log_pi, mu, sigma, yb,
                    n_model_samples=args.n_model_samples,
                    nll_weight=args.nll_weight,
                )

                # Auxiliary losses: SDE classification + vol regression
                sde_logits, vol_pred = model.forward_auxiliary(x)
                loss_sde = F.cross_entropy(sde_logits, sde_labels)
                loss_vol = F.mse_loss(vol_pred, rv_labels)
                loss_aux = loss_sde + loss_vol

                loss = loss_main + args.aux_weight * loss_aux
                loss = loss / args.grad_accum

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

            # Track
            sde_acc = (sde_logits.detach().argmax(-1) == sde_labels).float().mean().item()
            run_loss += loss.item() * args.grad_accum
            run_ed += ed.item()
            run_nll += nll.item()
            run_sde_acc += sde_acc
            run_vol += loss_vol.item()
            run_count += 1
            global_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.time() - t_start
                sps = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / sps if sps > 0 else 0

                logger.log({
                    'epoch': epoch, 'lr': lr,
                    'loss': run_loss / run_count,
                    'ed': run_ed / run_count,
                    'nll': run_nll / run_count,
                    'sde_acc': run_sde_acc / run_count,
                    'vol_mse': run_vol / run_count,
                    'steps/s': sps, 'eta_min': eta / 60,
                }, step=global_step)
                run_loss, run_ed, run_nll, run_sde_acc, run_vol, run_count = 0, 0, 0, 0, 0, 0

            if global_step % args.val_every == 0:
                val_m = validate(model, val_x, val_h, val_yb, val_sde, val_rv,
                                 device, args.n_model_samples, args.nll_weight)
                logger.log(val_m, step=global_step)
                if val_m['val_loss'] < best_val_loss:
                    best_val_loss = val_m['val_loss']
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scaler,
                                    global_step, epoch, cfg, best_val_loss)
                    print(f"  >> New best val_loss: {best_val_loss:.4f}")

            if global_step % args.save_every == 0:
                save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler,
                                global_step, epoch, cfg, best_val_loss)

        print(f"\n--- Epoch {epoch+1}/{args.epochs} complete (step {global_step}) ---\n")
        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler,
                        global_step, epoch + 1, cfg, best_val_loss)

    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler,
                    global_step, args.epochs, cfg, best_val_loss)
    elapsed = time.time() - t_start
    print(f"\nDone. {global_step:,} steps in {elapsed/60:.1f} min. Best val: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
