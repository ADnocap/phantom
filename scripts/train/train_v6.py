#!/usr/bin/env python
"""
Phantom v6: Crypto-focused fine-tuning from v5 checkpoint.

Two-phase training:
  Phase A (warmup):  Only patch_embed + head unfrozen (new channel weights learn)
  Phase B (full):    All layers unfrozen with LLRD

Key changes from v5:
  - 9 input channels (6 OHLCV + taker buy ratio + funding rate + OI change)
  - Crypto-only data (no equity/forex/commodity)
  - No asset classifier (single asset class)
  - Random feature masking on channels 6-8 for regularization
  - Early stopping with patience

Usage:
  python scripts/train/train_v6.py --v5_checkpoint checkpoints_v5/best.pt
  python scripts/train/train_v6.py --v5_checkpoint checkpoints_v5/best.pt --phase_a_steps 5000
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


def transfer_v5_weights(model, v5_path, device='cpu'):
    """Load v5 checkpoint and transfer weights with zero-padding for new channels."""
    ckpt = torch.load(v5_path, map_location=device, weights_only=False)
    v5_state = ckpt['model_state_dict']
    v6_state = model.state_dict()

    loaded, skipped, padded = 0, 0, 0
    for k, v in v5_state.items():
        if k not in v6_state:
            skipped += 1
            continue

        if v6_state[k].shape == v.shape:
            v6_state[k] = v
            loaded += 1
        elif k == 'patch_embed.proj.weight':
            # v5: (512, 30) -> v6: (512, 45) — zero-pad new channel columns
            old_dim = v.shape[1]
            v6_state[k][:, :old_dim] = v
            v6_state[k][:, old_dim:] = 0.0
            padded += 1
            print(f"  Padded: {k} {v.shape} -> {v6_state[k].shape}")
        elif k == 'asset_classifier.2.weight' or k == 'asset_classifier.2.bias':
            # Skip asset classifier (not used in v6)
            skipped += 1
        else:
            # Generic partial copy for any other size mismatch
            slices = tuple(slice(0, min(s, t))
                           for s, t in zip(v.shape, v6_state[k].shape))
            v6_state[k][slices] = v[slices]
            padded += 1
            print(f"  Partial: {k} {v.shape} -> {v6_state[k].shape}")

    model.load_state_dict(v6_state)
    step = ckpt.get('step', 0)
    print(f"Transferred v5 weights: {loaded} exact, {padded} padded, {skipped} skipped")
    return step


def make_param_groups(model, phase, lr_head, lr_encoder, llrd=0.8):
    """Create parameter groups with layer-wise learning rate decay."""
    if phase == 'A':
        # Only patch_embed and head
        groups = [
            {'params': model.patch_embed.parameters(), 'lr': lr_head},
            {'params': model.head.parameters(), 'lr': lr_head * 0.3},
        ]
        # Freeze everything else
        for name, param in model.named_parameters():
            if not name.startswith('patch_embed.') and not name.startswith('head.'):
                param.requires_grad = False
        return groups

    # Phase B: unfreeze all, LLRD
    for param in model.parameters():
        param.requires_grad = True

    groups = []

    # Head + decoder: full LR
    groups.append({
        'params': list(model.head.parameters()),
        'lr': lr_head,
        'name': 'head',
    })
    groups.append({
        'params': list(model.decoder_norm.parameters()),
        'lr': lr_head,
        'name': 'decoder_norm',
    })
    for i, layer in enumerate(model.decoder_layers):
        groups.append({
            'params': list(layer.parameters()),
            'lr': lr_head * 0.8,
            'name': f'decoder_{i}',
        })

    # Encoder layers: LLRD (deeper = lower LR)
    n_enc = len(model.encoder.layers)
    for i, layer in enumerate(model.encoder.layers):
        layer_lr = lr_encoder * (llrd ** (n_enc - 1 - i))
        groups.append({
            'params': list(layer.parameters()),
            'lr': layer_lr,
            'name': f'encoder_{i}',
        })

    # Encoder norm
    groups.append({
        'params': list(model.encoder.norm.parameters()),
        'lr': lr_encoder,
        'name': 'encoder_norm',
    })

    # Patch embed + pos_enc: lowest LR
    groups.append({
        'params': list(model.patch_embed.parameters()),
        'lr': lr_encoder * (llrd ** n_enc),
        'name': 'patch_embed',
    })
    if model.pos_enc.requires_grad:
        groups.append({
            'params': [model.pos_enc],
            'lr': lr_encoder * (llrd ** n_enc),
            'name': 'pos_enc',
        })

    # Auxiliary heads (vol regressor) — if present
    if hasattr(model, 'vol_regressor'):
        groups.append({
            'params': list(model.vol_regressor.parameters()),
            'lr': lr_head * 0.5,
            'name': 'vol_regressor',
        })

    return groups


def apply_feature_masking(x, mask_prob=0.15, mask_channels=(6, 7)):
    """Randomly zero-out new feature channels for regularization."""
    if mask_prob <= 0 or not mask_channels:
        return x
    B = x.size(0)
    for ch in mask_channels:
        mask = torch.rand(B, 1, device=x.device) < mask_prob
        x[:, :, ch] = x[:, :, ch] * (~mask).float()
    return x


@torch.no_grad()
def validate(model, val_x, val_y, val_rv, device, batch_size=512,
             nll_weight=1.0, crps_weight=0.5, mean_mse_weight=1.0,
             horizon_weighting='sqrt', min_mse_horizon=5):
    """Run validation and return metrics dict."""
    model.eval()
    n = len(val_x)
    accum = {}
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        y = val_y[i:i+batch_size].to(device)
        rv = val_rv[i:i+batch_size].to(device)

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
        pred_mean_std = pred_mean.std().item()
        mean_sigma = sigma.mean().item()

        for k, v in [('val_loss', loss), ('val_nll', nll), ('val_crps', crps),
                      ('val_mean_mse', mse), ('val_pred_mean_std', pred_mean_std),
                      ('val_mean_sigma', mean_sigma)]:
            val_v = v.item() if torch.is_tensor(v) else v
            accum[k] = accum.get(k, 0.0) + val_v

        if nu is not None:
            accum['val_mean_nu'] = accum.get('val_mean_nu', 0.0) + nu.mean().item()

        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in accum.items()}


def save_checkpoint(path, model, optimizer, step, config, best_val_loss, phase):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
        'best_val_loss': best_val_loss,
        'phase': phase,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Phantom v6 training")
    parser.add_argument('--v5_checkpoint', type=str, default='checkpoints_v5/best.pt')
    parser.add_argument('--data_dir', type=str, default='data/processed_v6')

    # Architecture (must match v5 except n_input_channels)
    parser.add_argument('--context_len', type=int, default=120)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_decoder_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_horizon', type=int, default=30)

    # Training phases
    parser.add_argument('--phase_a_steps', type=int, default=5000,
                        help='Steps for Phase A (patch embed warmup)')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Maximum total steps (Phase A + B)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs (will early-stop before this)')

    # Learning rates
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_encoder', type=float, default=5e-5)
    parser.add_argument('--lr_warmup_a', type=float, default=3e-4,
                        help='Peak LR for Phase A (patch embed)')
    parser.add_argument('--llrd', type=float, default=0.8,
                        help='Layer-wise LR decay factor')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss
    parser.add_argument('--nll_weight', type=float, default=1.0)
    parser.add_argument('--crps_weight', type=float, default=0.5)
    parser.add_argument('--mean_mse_weight', type=float, default=1.0)
    parser.add_argument('--horizon_weighting', type=str, default='sqrt')
    parser.add_argument('--min_mse_horizon', type=int, default=5)
    parser.add_argument('--enc_var_weight', type=float, default=0.1)
    parser.add_argument('--cond_drop_prob', type=float, default=0.10)

    # Regularization
    parser.add_argument('--feature_mask_prob', type=float, default=0.15,
                        help='Probability of zeroing new channels 6-8 per sample')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stop after this many val checks without improvement')

    # Logistics
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--val_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2500)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v6')
    parser.add_argument('--log_dir', type=str, default='logs/v6')
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

    # ── Model ──
    cfg = PhantomConfig(
        context_len=args.context_len, patch_len=5, patch_stride=5,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        n_decoder_layers=args.n_decoder_layers,
        head_type='student_t', n_components=1,
        n_input_channels=8,  # 6 OHLCV + taker buy + funding rate
        max_horizon=args.max_horizon, multi_horizon=True,
        use_asset_classifier=False,  # crypto only
        cond_drop_prob=args.cond_drop_prob,
    )

    model = PhantomModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | 8 input channels")

    # Transfer v5 weights
    print(f"\nTransferring weights from {args.v5_checkpoint}...")
    transfer_v5_weights(model, args.v5_checkpoint, device='cpu')
    model = model.to(device)

    # ── Data ──
    data_dir = Path(args.data_dir)
    dataset = RealAssetDatasetV5(str(data_dir / 'train.npz'))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers,
        pin_memory=device.type == 'cuda', drop_last=True,
        persistent_workers=args.n_workers > 0,
    )

    steps_per_epoch = len(dataset) // args.batch_size
    print(f"Data: {len(dataset):,} samples | {steps_per_epoch:,} steps/epoch")
    print(f"X channels: {dataset.X.shape[-1]}")

    # Validation set
    val_ds = RealAssetDatasetV5(str(data_dir / 'val.npz'))
    n_val = min(args.val_samples, len(val_ds))
    idx = np.random.RandomState(args.seed + 1234).choice(len(val_ds), n_val, replace=False)
    val_x = torch.from_numpy(val_ds.X[idx].astype(np.float32))
    val_y = torch.from_numpy(val_ds.Y_relative[idx].astype(np.float32))
    val_rv = torch.from_numpy(val_ds.realized_vol[idx].astype(np.float32))
    print(f"Validation: {n_val} samples")

    # AMP setup
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

    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════
    #  PHASE A: Patch embed warmup
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"PHASE A: Patch embed warmup ({args.phase_a_steps} steps)")
    print(f"{'='*60}\n")

    param_groups_a = make_param_groups(model, 'A', args.lr_warmup_a, args.lr_encoder)
    optimizer = torch.optim.AdamW(param_groups_a, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Phase A): {n_trainable:,} / {n_params:,}")

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    no_improve_count = 0
    run = {'loss': 0., 'nll': 0., 'crps': 0., 'mean_mse': 0.,
           'pred_mean_std': 0., 'grad_norm': 0., 'mean_nu': 0.}
    run_count = 0
    t_start = time.time()

    phase_a_done = False
    for epoch in range(args.epochs):
        for batch in loader:
            if global_step >= args.phase_a_steps:
                phase_a_done = True
                break

            x, y_curve, _, rv = batch
            x = x.to(device, non_blocking=True)
            y_curve = y_curve.to(device, non_blocking=True)

            # Feature masking (channels 6-8)
            if args.feature_mask_prob > 0:
                x = apply_feature_masking(x, args.feature_mask_prob)

            lr = get_lr(global_step, args.warmup_steps, args.phase_a_steps,
                        args.lr_warmup_a, args.min_lr)
            for g in optimizer.param_groups:
                g['lr'] = lr

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                enc_out = model.encode(x)
                log_pi, mu, sigma, nu = model.decode_curve(enc_out)

                loss, nll, crps, mse = combined_loss_v4(
                    log_pi, mu, sigma, y_curve, nu,
                    nll_weight=args.nll_weight, crps_weight=args.crps_weight,
                    mean_mse_weight=args.mean_mse_weight,
                    horizon_weighting=args.horizon_weighting,
                    min_mse_horizon=args.min_mse_horizon)

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

            # Log
            if global_step % args.log_every == 0 and run_count > 0:
                metrics = {k: v / run_count for k, v in run.items()}
                metrics['lr'] = lr
                metrics['phase'] = 'A'
                logger.log(metrics, global_step)
                run = {k: 0. for k in run}
                run_count = 0

            # Validate
            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_x, val_y, val_rv, device,
                                       nll_weight=args.nll_weight,
                                       crps_weight=args.crps_weight,
                                       mean_mse_weight=args.mean_mse_weight)
                logger.log(val_metrics, global_step)

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(ckpt_dir / 'best.pt', model, optimizer,
                                    global_step, cfg, best_val_loss, 'A')
                    print(f"  ** New best: val_loss={best_val_loss:.4f}")

        if phase_a_done:
            break

    print(f"\nPhase A done: {global_step} steps, best_val_loss={best_val_loss:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  PHASE B: Full fine-tuning with LLRD
    # ═══════════════════════════════════════════════════════════════

    total_steps = args.max_steps
    remaining_steps = total_steps - global_step

    print(f"\n{'='*60}")
    print(f"PHASE B: Full fine-tuning (up to {remaining_steps} steps, patience={args.patience})")
    print(f"{'='*60}\n")

    param_groups_b = make_param_groups(model, 'B', args.lr_head, args.lr_encoder, args.llrd)
    optimizer = torch.optim.AdamW(param_groups_b, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Phase B): {n_trainable:,} / {n_params:,}")

    # Print LR schedule
    for g in param_groups_b:
        name = g.get('name', '?')
        print(f"  {name}: lr={g['lr']:.2e}")

    no_improve_count = 0
    run = {'loss': 0., 'nll': 0., 'crps': 0., 'mean_mse': 0.,
           'pred_mean_std': 0., 'grad_norm': 0., 'mean_nu': 0.}
    run_count = 0

    for epoch in range(args.epochs):
        for batch in loader:
            if global_step >= total_steps:
                print(f"\nReached max steps ({total_steps})")
                break

            x, y_curve, _, rv = batch
            x = x.to(device, non_blocking=True)
            y_curve = y_curve.to(device, non_blocking=True)

            if args.feature_mask_prob > 0:
                x = apply_feature_masking(x, args.feature_mask_prob)

            # Cosine LR decay for Phase B
            phase_b_step = global_step - args.phase_a_steps
            lr_scale = get_lr(phase_b_step, args.warmup_steps, remaining_steps,
                              1.0, args.min_lr / args.lr_head)
            for g in optimizer.param_groups:
                base_lr = g.get('_base_lr', g['lr'])
                if '_base_lr' not in g:
                    g['_base_lr'] = g['lr']
                g['lr'] = base_lr * lr_scale

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                enc_out = model.encode(x)

                # Condition dropout
                if model.training and cfg.cond_drop_prob > 0:
                    mask = torch.rand(x.size(0), 1, 1, device=enc_out.device) > cfg.cond_drop_prob
                    enc_out_masked = enc_out * mask
                else:
                    enc_out_masked = enc_out

                log_pi, mu, sigma, nu = model.decode_curve(enc_out_masked)

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

            # Log
            if global_step % args.log_every == 0 and run_count > 0:
                metrics = {k: v / run_count for k, v in run.items()}
                metrics['lr'] = optimizer.param_groups[0]['lr']
                metrics['phase'] = 'B'
                logger.log(metrics, global_step)
                run = {k: 0. for k in run}
                run_count = 0

            # Validate
            if global_step % args.val_every == 0:
                val_metrics = validate(model, val_x, val_y, val_rv, device,
                                       nll_weight=args.nll_weight,
                                       crps_weight=args.crps_weight,
                                       mean_mse_weight=args.mean_mse_weight)
                logger.log(val_metrics, global_step)

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    no_improve_count = 0
                    save_checkpoint(ckpt_dir / 'best.pt', model, optimizer,
                                    global_step, cfg, best_val_loss, 'B')
                    print(f"  ** New best: val_loss={best_val_loss:.4f}")
                else:
                    no_improve_count += 1
                    print(f"  No improvement ({no_improve_count}/{args.patience})")

                if no_improve_count >= args.patience:
                    print(f"\nEarly stopping: no improvement for {args.patience} val checks")
                    break

            # Periodic checkpoint
            if global_step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f'step_{global_step}.pt', model, optimizer,
                                global_step, cfg, best_val_loss, 'B')

        if no_improve_count >= args.patience or global_step >= total_steps:
            break

    # Final save
    save_checkpoint(ckpt_dir / 'last.pt', model, optimizer,
                    global_step, cfg, best_val_loss, 'B')

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete")
    print(f"  Total steps: {global_step:,}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Best checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
