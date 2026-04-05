#!/usr/bin/env python
"""
Phantom v7: 4h-bar crypto training from v5 checkpoint.

Single-phase training (no warmup needed — patch_embed is re-initialized).
Transfers encoder/decoder/head from v5, re-inits patch_embed + pos_enc.

Usage:
  python scripts/train/train_v7.py --v5_checkpoint checkpoints_v5/best.pt
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
from src.real_data import RealAssetDatasetV5
from src.losses import combined_loss_v4, encoder_variance_penalty


def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


class Logger:
    def __init__(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "train_log.csv"
        self.val_csv_path = log_dir / "val_log.csv"
        self._header_written = False
        self._val_header_written = False

    def log(self, metrics, step, console=True):
        is_val = any(k.startswith('val_') for k in metrics.keys())
        csv_path = self.val_csv_path if is_val else self.csv_path
        header_attr = '_val_header_written' if is_val else '_header_written'
        if not getattr(self, header_attr):
            with open(csv_path, 'w') as f:
                f.write("step," + ",".join(metrics.keys()) + "\n")
            setattr(self, header_attr, True)
        with open(csv_path, 'a') as f:
            vals = ",".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                           for v in metrics.values())
            f.write(f"{step},{vals}\n")
        if console:
            parts = [f"step {step:>7d}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts))


def transfer_v5_weights(model, v5_path):
    """Transfer v5 weights: encoder/decoder/head exact, patch_embed/pos_enc/horizon re-init."""
    ckpt = torch.load(v5_path, map_location='cpu', weights_only=False)
    v5_state = ckpt['model_state_dict']
    v7_state = model.state_dict()

    loaded, skipped, reinit = 0, 0, 0
    for k, v in v5_state.items():
        if k not in v7_state:
            skipped += 1
            continue

        if v7_state[k].shape == v.shape:
            v7_state[k] = v
            loaded += 1
        elif k == 'horizon_embed.weight':
            # v5: (31, 512), v7: (91, 512) — copy first 31 rows
            n_copy = min(v.shape[0], v7_state[k].shape[0])
            v7_state[k][:n_copy] = v[:n_copy]
            reinit += 1
            print(f"  Partial: {k} {v.shape} -> {v7_state[k].shape} (copied {n_copy} rows)")
        else:
            # patch_embed.proj, pos_enc — different sizes, keep random init
            reinit += 1
            print(f"  Re-init: {k} {v.shape} -> {v7_state[k].shape}")

    model.load_state_dict(v7_state)
    print(f"Transferred: {loaded} exact, {reinit} re-init/partial, {skipped} skipped")


@torch.no_grad()
def validate(model, val_x, val_y, device, batch_size=256,
             nll_weight=1.0, crps_weight=0.5, mean_mse_weight=0.3,
             horizon_weighting='sqrt', min_mse_horizon=30):
    model.eval()
    n = len(val_x)
    accum = {}
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        y = val_y[i:i+batch_size].to(device)

        enc_out = model.encode(x)
        log_pi, mu, sigma, nu = model.decode_curve(enc_out)

        loss, nll, crps, mse = combined_loss_v4(
            log_pi, mu, sigma, y, nu,
            nll_weight=nll_weight, crps_weight=crps_weight,
            mean_mse_weight=mean_mse_weight,
            horizon_weighting=horizon_weighting,
            min_mse_horizon=min_mse_horizon)

        pi = log_pi.exp()
        pred_mean = (pi * mu).sum(dim=-1)

        for k, v in [('val_loss', loss), ('val_nll', nll), ('val_crps', crps),
                      ('val_mean_mse', mse),
                      ('val_pred_mean_std', pred_mean.std().item()),
                      ('val_mean_sigma', sigma.mean().item())]:
            val_v = v.item() if torch.is_tensor(v) else v
            accum[k] = accum.get(k, 0.0) + val_v
        if nu is not None:
            accum['val_mean_nu'] = accum.get('val_mean_nu', 0.0) + nu.mean().item()
        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in accum.items()}


def save_checkpoint(path, model, optimizer, step, config, best_val_loss):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
        'best_val_loss': best_val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Phantom v7 training (4h bars)")
    parser.add_argument('--v5_checkpoint', type=str, default='checkpoints_v5/best.pt')
    parser.add_argument('--data_dir', type=str, default='data/processed_v7')

    # Architecture
    parser.add_argument('--context_len', type=int, default=720)
    parser.add_argument('--patch_len', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_decoder_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_horizon', type=int, default=90)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss
    parser.add_argument('--nll_weight', type=float, default=1.0)
    parser.add_argument('--crps_weight', type=float, default=0.5)
    parser.add_argument('--mean_mse_weight', type=float, default=0.3)
    parser.add_argument('--horizon_weighting', type=str, default='sqrt')
    parser.add_argument('--min_mse_horizon', type=int, default=30,
                        help='Only apply mean MSE on horizons >= this (30 bars ≈ 5 days)')
    parser.add_argument('--enc_var_weight', type=float, default=0.1)
    parser.add_argument('--cond_drop_prob', type=float, default=0.15)

    # Early stopping
    parser.add_argument('--patience', type=int, default=10)

    # Logistics
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--val_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v7')
    parser.add_argument('--log_dir', type=str, default='logs/v7')
    parser.add_argument('--val_samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')

    args = parser.parse_args()
    if args.no_amp:
        args.amp = False

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Model
    cfg = PhantomConfig(
        context_len=args.context_len,
        patch_len=args.patch_len, patch_stride=args.patch_len,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        n_decoder_layers=args.n_decoder_layers,
        head_type='student_t', n_components=1,
        n_input_channels=6,
        max_horizon=args.max_horizon, multi_horizon=True,
        use_asset_classifier=False,
        cond_drop_prob=args.cond_drop_prob,
    )

    model = PhantomModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | patches={cfg.n_patches} | "
          f"context={cfg.context_len} | patch_len={cfg.patch_len} | horizons={cfg.max_horizon}")

    # Transfer weights
    print(f"\nTransferring from {args.v5_checkpoint}...")
    transfer_v5_weights(model, args.v5_checkpoint)
    model = model.to(device)

    # Data
    data_dir = Path(args.data_dir)
    dataset = RealAssetDatasetV5(str(data_dir / 'train.npz'))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers,
        pin_memory=device.type == 'cuda', drop_last=True,
        persistent_workers=args.n_workers > 0,
    )

    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = min(args.max_steps, steps_per_epoch * args.epochs)
    print(f"Data: {len(dataset):,} samples | {steps_per_epoch:,} steps/epoch")
    print(f"Training: up to {total_steps:,} steps, patience={args.patience}")

    # Validation
    val_ds = RealAssetDatasetV5(str(data_dir / 'val.npz'))
    n_val = min(args.val_samples, len(val_ds))
    idx = np.random.RandomState(args.seed + 1234).choice(len(val_ds), n_val, replace=False)
    val_x = torch.from_numpy(val_ds.X[idx].astype(np.float32))
    val_y = torch.from_numpy(val_ds.Y_relative[idx].astype(np.float32))
    print(f"Validation: {n_val} samples")

    # AMP
    use_amp = args.amp and device.type == 'cuda'
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("AMP: bfloat16")
    elif use_amp:
        amp_dtype = torch.float16
        print("AMP: float16")
    else:
        amp_dtype = torch.float32
        use_amp = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95))

    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    # Training
    print(f"\n{'='*60}")
    print(f"Starting v7 training (4h bars, {cfg.n_patches} patches, {cfg.max_horizon} horizons)")
    print(f"{'='*60}\n")

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    no_improve_count = 0
    run = {'loss': 0., 'nll': 0., 'crps': 0., 'mean_mse': 0.,
           'pred_mean_std': 0., 'grad_norm': 0., 'mean_nu': 0.}
    run_count = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        for batch in loader:
            if global_step >= total_steps:
                break

            x, y_curve, _, rv = batch
            x = x.to(device, non_blocking=True)
            y_curve = y_curve.to(device, non_blocking=True)

            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g['lr'] = lr

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                enc_out = model.encode(x)

                if model.training and cfg.cond_drop_prob > 0:
                    mask = torch.rand(x.size(0), 1, 1, device=enc_out.device) > cfg.cond_drop_prob
                    enc_out = enc_out * mask

                log_pi, mu, sigma, nu = model.decode_curve(enc_out)

                loss, nll, crps, mse = combined_loss_v4(
                    log_pi, mu, sigma, y_curve, nu,
                    nll_weight=args.nll_weight, crps_weight=args.crps_weight,
                    mean_mse_weight=args.mean_mse_weight,
                    horizon_weighting=args.horizon_weighting,
                    min_mse_horizon=args.min_mse_horizon)

                if args.enc_var_weight > 0:
                    loss = loss + args.enc_var_weight * encoder_variance_penalty(enc_out)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pi = log_pi.exp()
                pred_mean = (pi * mu).sum(dim=-1)

            run['loss'] += loss.item()
            run['nll'] += nll.item()
            run['crps'] += crps.item()
            run['mean_mse'] += mse.item()
            run['pred_mean_std'] += pred_mean.std().item()
            run['grad_norm'] += grad_norm.item()
            if nu is not None:
                run['mean_nu'] += nu.mean().item()
            run_count += 1
            global_step += 1

            if global_step % args.log_every == 0 and run_count > 0:
                metrics = {k: v / run_count for k, v in run.items()}
                metrics['lr'] = lr
                logger.log(metrics, global_step)
                run = {k: 0. for k in run}
                run_count = 0

            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_x, val_y, device,
                                       nll_weight=args.nll_weight,
                                       crps_weight=args.crps_weight,
                                       mean_mse_weight=args.mean_mse_weight,
                                       min_mse_horizon=args.min_mse_horizon)
                logger.log(val_metrics, global_step)

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    no_improve_count = 0
                    save_checkpoint(ckpt_dir / 'best.pt', model, optimizer,
                                    global_step, cfg, best_val_loss)
                    print(f"  ** New best: val_loss={best_val_loss:.4f}")
                else:
                    no_improve_count += 1
                    print(f"  No improvement ({no_improve_count}/{args.patience})")

                if no_improve_count >= args.patience:
                    print(f"\nEarly stopping at step {global_step}")
                    break

            if global_step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f'step_{global_step}.pt', model, optimizer,
                                global_step, cfg, best_val_loss)

        if no_improve_count >= args.patience or global_step >= total_steps:
            break

    save_checkpoint(ckpt_dir / 'last.pt', model, optimizer,
                    global_step, cfg, best_val_loss)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {global_step:,} steps, best_val_loss={best_val_loss:.4f}")
    print(f"Time: {elapsed/60:.1f} min | Best: {ckpt_dir / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
