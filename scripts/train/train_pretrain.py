#!/usr/bin/env python
"""
Synthetic pre-training for Phantom with three anti-collapse mechanisms:
  1. Auxiliary tasks (SDE type classifier + vol regressor)
  2. Condition dropout (classifier-free guidance training)
  3. Cross-attention decoder (architectural conditioning)

v2 additions:
  4. Student-t mixture head (--use_student_t)
  5. Gumbel-Softmax energy distance (--use_gumbel_softmax)
  6. Auxiliary quantile loss (--quantile_weight)
  7. Multi-scale patching (--patch_sizes)
  8. Multi-channel input features (--n_input_channels)
  9. Series decomposition (--use_decomposition)
  10. New SDE families (--sde_version v2)

Usage:
  # v1 (original):
  python train_pretrain.py --data_mode online --samples_per_epoch 1000000

  # v2 (all improvements):
  python train_pretrain.py --data_mode online --use_student_t --use_gumbel_softmax \\
    --quantile_weight 0.2 --use_decomposition --patch_sizes 3 5 15 \\
    --n_input_channels 4 --sde_version v2 --context_len 75
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
        self.val_csv_path = log_dir / "val_log.csv"
        self._header_written = False
        self._val_header_written = False

    def log(self, metrics, step, console=True):
        # Detect if this is a validation row (keys start with 'val_')
        is_val = any(k.startswith('val_') for k in metrics.keys())
        csv_path = self.val_csv_path if is_val else self.csv_path

        if is_val:
            if not self._val_header_written:
                with open(csv_path, 'w') as f:
                    f.write("step," + ",".join(metrics.keys()) + "\n")
                self._val_header_written = True
        else:
            if not self._header_written:
                with open(csv_path, 'w') as f:
                    f.write("step," + ",".join(metrics.keys()) + "\n")
                self._header_written = True

        with open(csv_path, 'a') as f:
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
             n_model_samples=256, nll_weight=0.1, batch_size=512,
             use_gumbel_softmax=False, quantile_weight=0.0):
    from src.losses import crps_loss, quantile_loss
    model.eval()
    n = val_x.size(0)
    accum = {}
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        h = val_h[i:i+batch_size].to(device)
        yb = val_yb[i:i+batch_size].to(device)
        sde = val_sde[i:i+batch_size].to(device)
        rv = val_rv[i:i+batch_size].to(device)

        log_pi, mu, sigma, nu = model(x, h)
        loss, ed, nll = combined_loss(log_pi, mu, sigma, yb,
                                      n_model_samples=n_model_samples,
                                      nll_weight=nll_weight,
                                      nu=nu,
                                      use_gumbel_softmax=use_gumbel_softmax,
                                      quantile_weight=quantile_weight)

        # Pick a random branch as scalar target for CRPS
        B, N = yb.shape
        rand_idx = torch.randint(N, (B,), device=yb.device)
        y_single = yb[torch.arange(B, device=yb.device), rand_idx]
        crps_val = crps_loss(log_pi, mu, sigma, y_single, nu=nu)

        # Auxiliary
        sde_logits, vol_pred, _, _ = model.forward_auxiliary(x)
        sde_acc = (sde_logits.argmax(dim=-1) == sde).float().mean()
        vol_mse = F.mse_loss(vol_pred, rv)

        # Head stats
        pi = log_pi.exp()
        eff_k = torch.exp(-torch.sum(pi * log_pi, dim=-1)).mean()  # effective components
        mean_mu = mu.abs().mean()
        mean_sigma = sigma.mean()

        # Per-horizon ED
        for h_val in [3, 5, 7]:
            mask = (h == h_val)
            if mask.any():
                h_key = f'val_ed_h{h_val}'
                from src.losses import energy_distance_loss
                h_ed = energy_distance_loss(log_pi[mask], mu[mask], sigma[mask],
                                            yb[mask], n_model_samples, nu=nu[mask] if nu is not None else None)
                accum[h_key] = accum.get(h_key, 0.0) + h_ed.item()

        # Accumulate
        for k, v in [('val_loss', loss), ('val_ed', ed), ('val_nll', nll),
                      ('val_crps', crps_val), ('val_sde_acc', sde_acc),
                      ('val_vol_mse', vol_mse), ('val_eff_k', eff_k),
                      ('val_mean_mu', mean_mu), ('val_mean_sigma', mean_sigma)]:
            accum[k] = accum.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        if nu is not None:
            mean_nu = nu.mean()
            accum['val_mean_nu'] = accum.get('val_mean_nu', 0.0) + mean_nu.item()

        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def validate_v3(model, val_x, val_h, val_y, val_asset, val_rv, device,
                nll_weight=1.0, crps_weight=0.5, batch_size=512):
    """Validation for v3 real data mode (single-target)."""
    from src.losses import combined_loss_v3, crps_loss
    model.eval()
    n = val_x.size(0)
    accum = {}
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        h = val_h[i:i+batch_size].to(device)
        y = val_y[i:i+batch_size].to(device)
        asset = val_asset[i:i+batch_size].to(device)
        rv = val_rv[i:i+batch_size].to(device)

        log_pi, mu, sigma, nu = model(x, h)
        loss, nll, crps = combined_loss_v3(
            log_pi, mu, sigma, y, nu,
            nll_weight=nll_weight, crps_weight=crps_weight)

        # Auxiliary
        _, vol_pred, asset_logits, sign_logits = model.forward_auxiliary(x)
        vol_mse = F.mse_loss(vol_pred, rv)
        asset_acc = (asset_logits.argmax(-1) == asset).float().mean() if asset_logits is not None else torch.tensor(0.)
        sign_labels = (y > 0).long()
        sign_acc = (sign_logits.argmax(-1) == sign_labels).float().mean() if sign_logits is not None else torch.tensor(0.)

        # Head stats
        pi = log_pi.exp()
        eff_k = torch.exp(-torch.sum(pi * log_pi, dim=-1)).mean()
        mean_mu = mu.abs().mean()
        mean_sigma = sigma.mean()

        for k, v in [('val_loss', loss), ('val_nll', nll), ('val_crps', crps),
                      ('val_vol_mse', vol_mse), ('val_asset_acc', asset_acc),
                      ('val_sign_acc', sign_acc), ('val_eff_k', eff_k),
                      ('val_mean_mu', mean_mu), ('val_mean_sigma', mean_sigma)]:
            accum[k] = accum.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        if nu is not None:
            accum['val_mean_nu'] = accum.get('val_mean_nu', 0.0) + nu.mean().item()

        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def validate_v4(model, val_x, val_y, val_asset, val_rv, device,
                nll_weight=1.0, crps_weight=0.5, mean_mse_weight=1.0,
                horizon_weighting='sqrt', batch_size=512):
    """Validation for v4 multi-horizon curve prediction."""
    from src.losses import combined_loss_v4
    model.eval()
    n = val_x.size(0)
    accum = {}
    n_batches = 0

    for i in range(0, n, batch_size):
        x = val_x[i:i+batch_size].to(device)
        y_curve = val_y[i:i+batch_size].to(device)
        asset = val_asset[i:i+batch_size].to(device)
        rv = val_rv[i:i+batch_size].to(device)

        log_pi, mu, sigma, nu = model(x)  # multi-horizon: (B, H, K)
        loss, nll, crps, mse = combined_loss_v4(
            log_pi, mu, sigma, y_curve, nu,
            nll_weight=nll_weight, crps_weight=crps_weight,
            mean_mse_weight=mean_mse_weight, horizon_weighting=horizon_weighting)

        # Pred mean stats (key metric — is mu alive?)
        pi = log_pi.exp()
        pred_mean = (pi * mu).sum(dim=-1)  # (B, H)
        pred_mean_std = pred_mean.std().item()

        # Auxiliary
        _, vol_pred, asset_logits, sign_logits = model.forward_auxiliary(x)
        vol_mse = F.mse_loss(vol_pred, rv)
        asset_acc = (asset_logits.argmax(-1) == asset).float().mean() if asset_logits is not None else torch.tensor(0.)

        # Mean sigma and nu (averaged across horizons)
        mean_sigma = sigma.mean().item()
        mean_mu = mu.abs().mean().item()

        for k, v in [('val_loss', loss), ('val_nll', nll), ('val_crps', crps),
                      ('val_mean_mse', mse), ('val_pred_mean_std', pred_mean_std),
                      ('val_vol_mse', vol_mse), ('val_asset_acc', asset_acc),
                      ('val_mean_sigma', mean_sigma), ('val_mean_mu', mean_mu)]:
            val = v.item() if torch.is_tensor(v) else v
            accum[k] = accum.get(k, 0.0) + val

        if nu is not None:
            accum['val_mean_nu'] = accum.get('val_mean_nu', 0.0) + nu.mean().item()

        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in accum.items()}


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

    p.add_argument('--data_mode', type=str, default='online',
                   choices=['shards', 'online', 'real_assets', 'v4_real_assets', 'v5_real_assets'])
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
    p.add_argument('--resume', type=str, default=None,
                   help='Resume full training state from checkpoint')
    p.add_argument('--init_from', type=str, default=None,
                   help='Initialize model weights from checkpoint (partial transfer, no optimizer)')
    p.add_argument('--val_samples', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)

    # ── v2 flags ──
    p.add_argument('--use_student_t', action='store_true', default=False,
                   help='Use Student-t mixture head instead of Gaussian')
    p.add_argument('--head_type', type=str, default='mog',
                   choices=['mog', 'mot', 'student_t'],
                   help='Head type: mog (Gaussian mix), mot (Student-t mix), student_t (single)')
    p.add_argument('--use_gumbel_softmax', action='store_true', default=False,
                   help='Use Gumbel-Softmax for differentiable energy distance')
    p.add_argument('--quantile_weight', type=float, default=0.0,
                   help='Weight for auxiliary quantile loss (0 = disabled)')
    p.add_argument('--patch_sizes', type=int, nargs='+', default=None,
                   help='Multi-scale patch sizes (e.g. 3 5 15). None = single-scale')
    p.add_argument('--n_input_channels', type=int, default=1,
                   help='Input channels (1=returns, 4=returns+3 vol features)')
    p.add_argument('--use_decomposition', action='store_true', default=False,
                   help='Use in-model series decomposition')
    p.add_argument('--decomp_kernel', type=int, default=5,
                   help='Kernel size for decomposition moving average')
    p.add_argument('--sde_version', type=str, default='v1', choices=['v1', 'v2', 'v3'],
                   help='SDE version: v1=original, v2=+MRW/FracOU, v3=+GARCH/Momentum (non-Markovian)')

    # ── Phase 6: conditional signal fixes ──
    p.add_argument('--use_crps_avg', action='store_true', default=False,
                   help='Use CRPS-avg over branches instead of ED (zero noise)')
    p.add_argument('--use_film', action='store_true', default=False,
                   help='Use FiLM conditioning (multiplicative, prevents bypass)')
    p.add_argument('--contrastive_weight', type=float, default=0.0,
                   help='Weight for InfoNCE contrastive loss (0 = disabled)')
    p.add_argument('--enc_var_weight', type=float, default=0.0,
                   help='Weight for encoder variance penalty (0 = disabled)')
    p.add_argument('--mean_match_weight', type=float, default=0.0,
                   help='Weight for mean matching loss (0 = disabled)')
    p.add_argument('--var_match_weight', type=float, default=0.0,
                   help='Weight for variance matching loss (0 = disabled)')

    # ── v3 flags (real multi-asset pretraining) ──
    p.add_argument('--real_data_dir', type=str, default='data/processed/',
                   help='Directory with train.npz/val.npz for real_assets mode')
    p.add_argument('--nll_weight_v3', type=float, default=1.0,
                   help='NLL weight for v3/v4 real data training')
    p.add_argument('--crps_weight_v3', type=float, default=0.5,
                   help='CRPS weight for v3/v4 real data training')
    p.add_argument('--asset_cls_weight', type=float, default=0.3,
                   help='Weight for asset-type + vol regressor auxiliary')
    p.add_argument('--sign_cls_weight', type=float, default=0.1,
                   help='Weight for return-sign classifier auxiliary')

    # ── v4 flags (multi-horizon curve prediction) ──
    p.add_argument('--max_horizon', type=int, default=30,
                   help='Maximum prediction horizon for v4 (default 30)')
    p.add_argument('--mean_mse_weight', type=float, default=1.0,
                   help='Weight for horizon-weighted mean MSE loss (v4)')
    p.add_argument('--horizon_weighting', type=str, default='sqrt',
                   choices=['sqrt', 'linear', 'log', 'uniform'],
                   help='How to weight mean MSE across horizons (v4)')
    p.add_argument('--min_mse_horizon', type=int, default=0,
                   help='Only apply mean MSE on horizons >= this (v4, 0=all)')
    p.add_argument('--synth_fraction', type=float, default=0.0,
                   help='Fraction of batch from synthetic GARCH/Momentum SDEs (v4)')

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

    # Determine n_sde_types from sde_version
    n_sde_types = 7 if args.sde_version in ('v2', 'v3') else 5

    # v3/v4/v5 real_assets mode overrides
    is_v3 = args.data_mode == 'real_assets'
    is_v4 = args.data_mode == 'v4_real_assets'
    is_v5 = args.data_mode == 'v5_real_assets'
    is_real = is_v3 or is_v4 or is_v5
    if is_real:
        args.n_input_channels = 6  # Force 6-channel OHLCV features

    cfg = PhantomConfig(
        context_len=args.context_len, patch_len=args.patch_len,
        patch_stride=args.patch_stride, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        dropout=args.dropout, n_components=args.n_components,
        n_decoder_layers=args.n_decoder_layers,
        cond_drop_prob=args.cond_drop_prob,
        n_sde_types=n_sde_types,
        # v2
        use_student_t=(args.head_type == 'mot' or args.use_student_t),
        patch_sizes=args.patch_sizes,
        n_input_channels=args.n_input_channels,
        use_decomposition=args.use_decomposition,
        decomp_kernel=args.decomp_kernel,
        use_film=args.use_film,
        head_type=args.head_type,
        # v3
        use_asset_classifier=is_real,
        use_sign_classifier=is_v3,
        # v4/v5
        max_horizon=args.max_horizon,
        multi_horizon=(is_v4 or is_v5),
    )

    model = PhantomModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | patches={cfg.n_patches} | K={cfg.n_components}")
    print(f"Decoder layers: {cfg.n_decoder_layers} | Cond drop: {cfg.cond_drop_prob}")
    if args.use_student_t:
        print("Head: Mixture of Student-t")
    if args.patch_sizes:
        print(f"Multi-scale patches: {args.patch_sizes}")
    if args.n_input_channels > 1:
        print(f"Input channels: {args.n_input_channels}")
    if args.use_decomposition:
        print(f"Series decomposition: kernel={args.decomp_kernel}")
    if args.use_gumbel_softmax:
        print("Energy distance: Gumbel-Softmax (differentiable)")
    if args.quantile_weight > 0:
        print(f"Quantile loss weight: {args.quantile_weight}")
    if args.sde_version == 'v2':
        print("SDE families: v2 (+ MRW, Fractional OU)")

    # Data
    if is_v5:
        from src.real_data import RealAssetDatasetV5
        real_data_dir = Path(args.real_data_dir)
        dataset = RealAssetDatasetV5(str(real_data_dir / 'train.npz'))
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.n_workers,
            pin_memory=device.type == 'cuda', drop_last=True,
            persistent_workers=args.n_workers > 0,
        )
        _synth_loader = None
    elif is_v4 or is_v5:
        from src.real_data import RealAssetDatasetV4
        real_data_dir = Path(args.real_data_dir)
        real_dataset = RealAssetDatasetV4(str(real_data_dir / 'train.npz'))

        if args.synth_fraction > 0:
            # Mixed real + synthetic dataloader
            from src.data import SyntheticCurveDataset
            from torch.utils.data import ConcatDataset

            n_real = len(real_dataset)
            n_synth = int(n_real * args.synth_fraction / (1 - args.synth_fraction))
            synth_dataset = SyntheticCurveDataset(
                context_len=args.context_len, max_horizon=args.max_horizon,
                samples_per_epoch=n_synth, seed=args.seed,
            )
            print(f"Mixing: {n_real:,} real + {n_synth:,} synthetic "
                  f"({args.synth_fraction*100:.0f}% synth)")

            # Use separate loaders and interleave in training loop
            real_loader = DataLoader(
                real_dataset, batch_size=int(args.batch_size * (1 - args.synth_fraction)),
                shuffle=True, num_workers=args.n_workers,
                pin_memory=device.type == 'cuda', drop_last=True,
                persistent_workers=args.n_workers > 0,
            )
            synth_loader = DataLoader(
                synth_dataset, batch_size=int(args.batch_size * args.synth_fraction),
                num_workers=0, pin_memory=device.type == 'cuda', drop_last=True,
            )
            dataset = real_dataset  # for len() computation
            loader = real_loader
            _synth_loader = synth_loader
        else:
            dataset = real_dataset
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.n_workers,
                pin_memory=device.type == 'cuda', drop_last=True,
                persistent_workers=args.n_workers > 0,
            )
            _synth_loader = None
    elif is_v3:
        from src.real_data import RealAssetDataset
        real_data_dir = Path(args.real_data_dir)
        dataset = RealAssetDataset(str(real_data_dir / 'train.npz'))
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.n_workers,
            pin_memory=device.type == 'cuda', drop_last=True,
            persistent_workers=args.n_workers > 0,
        )
    else:
        dataset = OnlineDataset(
            context_len=cfg.context_len, n_branches=args.n_branches,
            samples_per_epoch=args.samples_per_epoch, seed=args.seed,
            n_input_channels=args.n_input_channels,
            sde_version=args.sde_version,
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
    elif args.init_from:
        # Partial weight transfer (e.g., v3 → v4)
        ckpt = torch.load(args.init_from, map_location='cpu', weights_only=False)
        src_state = ckpt['model_state_dict']
        tgt_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in src_state.items():
            if k in tgt_state and tgt_state[k].shape == v.shape:
                tgt_state[k] = v
                loaded += 1
            elif k in tgt_state:
                # Partial copy for size mismatches (e.g., horizon_embed, pos_enc)
                src_shape, tgt_shape = v.shape, tgt_state[k].shape
                slices = tuple(slice(0, min(s, t)) for s, t in zip(src_shape, tgt_shape))
                tgt_state[k][slices] = v[slices]
                loaded += 1
                print(f"  Partial load: {k} {src_shape} → {tgt_shape}")
            else:
                skipped += 1
        model.load_state_dict(tgt_state)
        print(f"Initialized from {args.init_from}: {loaded} params loaded, {skipped} skipped")

    if is_v5:
        from src.real_data import RealAssetDatasetV5
        real_data_dir = Path(args.real_data_dir)
        val_ds = RealAssetDatasetV5(str(real_data_dir / 'val.npz'))
        n_val = min(args.val_samples, len(val_ds))
        idx = np.random.RandomState(args.seed + 1234).choice(len(val_ds), n_val, replace=False)
        val_x = torch.from_numpy(val_ds.X[idx].astype(np.float32))
        val_y_curve = torch.from_numpy(val_ds.Y_relative[idx].astype(np.float32))
        val_asset = torch.from_numpy(val_ds.asset_type[idx].astype(np.int64))
        val_rv = torch.from_numpy(val_ds.realized_vol[idx].astype(np.float32))
        print(f"Loaded v5 validation set: {n_val} samples (relative targets)")
    elif is_v4 or is_v5:
        from src.real_data import RealAssetDatasetV4
        real_data_dir = Path(args.real_data_dir)
        val_ds = RealAssetDatasetV4(str(real_data_dir / 'val.npz'))
        n_val = min(args.val_samples, len(val_ds))
        idx = np.random.RandomState(args.seed + 1234).choice(len(val_ds), n_val, replace=False)
        val_x = torch.from_numpy(val_ds.X[idx].astype(np.float32))
        val_y_curve = torch.from_numpy(val_ds.Y[idx].astype(np.float32))
        val_asset = torch.from_numpy(val_ds.asset_type[idx].astype(np.int64))
        val_rv = torch.from_numpy(val_ds.realized_vol[idx].astype(np.float32))
        print(f"Loaded v4 validation set: {n_val} samples from {real_data_dir / 'val.npz'}")
    elif is_v3:
        from src.real_data import RealAssetDataset
        real_data_dir = Path(args.real_data_dir)
        val_ds = RealAssetDataset(str(real_data_dir / 'val.npz'))
        n_val = min(args.val_samples, len(val_ds))
        idx = np.random.RandomState(args.seed + 1234).choice(len(val_ds), n_val, replace=False)
        val_x = torch.from_numpy(val_ds.X[idx].astype(np.float32))
        val_h = torch.from_numpy(val_ds.H[idx].astype(np.int64))
        val_y = torch.from_numpy(val_ds.Y[idx].astype(np.float32))
        val_asset = torch.from_numpy(val_ds.asset_type[idx].astype(np.int64))
        val_rv = torch.from_numpy(val_ds.realized_vol[idx].astype(np.float32))
        print(f"Loaded validation set: {n_val} samples from {real_data_dir / 'val.npz'}")
    else:
        print(f"Generating validation set ({args.val_samples} samples, {args.n_branches} branches)...")
        val_x, val_h, val_yb, val_sde, val_rv = make_validation_batch(
            n_samples=args.val_samples, context_len=cfg.context_len,
            n_branches=args.n_branches, seed=args.seed + 1234,
            n_input_channels=args.n_input_channels,
            sde_version=args.sde_version,
        )

    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    model.train()
    global_step = start_step
    # Running accumulators for detailed logging
    run = {
        'loss': 0., 'loss_main': 0., 'loss_aux': 0.,
        'ed': 0., 'nll': 0., 'loss_sde': 0., 'vol_mse': 0.,
        'sde_acc': 0.,
        'mean_mu': 0., 'mean_sigma': 0., 'eff_k': 0.,
        'grad_norm': 0.,
    }
    if args.use_student_t or is_real:
        run['mean_nu'] = 0.
    if is_v4 or is_v5:
        run['mean_mse'] = 0.
        run['pred_mean_std'] = 0.
    if args.quantile_weight > 0:
        run['quantile_loss'] = 0.
    run_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print("Starting pre-training (cross-attn + aux tasks + cond dropout)")
    print(f"{'='*60}\n")

    _synth_iter = iter(_synth_loader) if (is_v4 and _synth_loader is not None) else None

    for epoch in range(start_epoch, args.epochs):
        if not is_real:
            dataset.seed = args.seed + epoch * 1000

        for batch_idx, batch in enumerate(loader):
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            if is_v4 or is_v5:
                # ── v4 multi-horizon curve path ──
                x, y_curve, asset_labels, rv_labels = batch

                # Merge synthetic batch if available
                if _synth_loader is not None:
                    try:
                        sx, sy, sa, sr = next(_synth_iter)
                    except (StopIteration, NameError):
                        _synth_iter = iter(_synth_loader)
                        sx, sy, sa, sr = next(_synth_iter)
                    x = torch.cat([x, sx], dim=0)
                    y_curve = torch.cat([y_curve, sy], dim=0)
                    asset_labels = torch.cat([asset_labels, sa], dim=0)
                    rv_labels = torch.cat([rv_labels, sr], dim=0)

                x = x.to(device, non_blocking=True)
                y_curve = y_curve.to(device, non_blocking=True)
                asset_labels = asset_labels.to(device, non_blocking=True)
                rv_labels = rv_labels.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    enc_out = model.encode(x)

                    if model.training and model.cfg.cond_drop_prob > 0:
                        mask = torch.rand(x.size(0), 1, 1, device=enc_out.device) > model.cfg.cond_drop_prob
                        enc_out_masked = enc_out * mask
                    else:
                        enc_out_masked = enc_out

                    log_pi, mu, sigma, nu = model.decode_curve(enc_out_masked)

                    from src.losses import combined_loss_v4
                    loss_main, nll, crps_val, mean_mse = combined_loss_v4(
                        log_pi, mu, sigma, y_curve, nu,
                        nll_weight=args.nll_weight_v3,
                        crps_weight=args.crps_weight_v3,
                        mean_mse_weight=args.mean_mse_weight,
                        horizon_weighting=args.horizon_weighting,
                        min_mse_horizon=args.min_mse_horizon,
                    )

                    # Auxiliary: asset-type classifier + vol regressor
                    _, vol_pred, asset_logits, _ = model.forward_auxiliary(x)
                    loss_asset = F.cross_entropy(asset_logits, asset_labels)
                    loss_vol = F.mse_loss(vol_pred, rv_labels)
                    loss_aux = loss_asset + loss_vol
                    loss = loss_main + args.asset_cls_weight * loss_aux

                    if args.enc_var_weight > 0:
                        from src.losses import encoder_variance_penalty
                        loss = loss + args.enc_var_weight * encoder_variance_penalty(enc_out)

                    loss = loss / args.grad_accum

                # Track metrics for v4
                primary = crps_val
                sde_labels = asset_labels
                sde_logits = asset_logits
                yb = None
                # Track pred_mean_std (key v4 metric)
                with torch.no_grad():
                    pi = log_pi.exp()
                    pred_mean = (pi * mu).sum(dim=-1)  # (B, H)
                    _pred_mean_std = pred_mean.std().item()

            elif is_v3:
                # ── v3 real data path ──
                x, h, target, asset_labels, rv_labels = batch
                x = x.to(device, non_blocking=True)
                h = h.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                asset_labels = asset_labels.to(device, non_blocking=True)
                rv_labels = rv_labels.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    enc_out = model.encode(x)

                    # Condition dropout
                    if model.training and model.cfg.cond_drop_prob > 0:
                        mask = torch.rand(x.size(0), 1, 1, device=enc_out.device) > model.cfg.cond_drop_prob
                        enc_out_masked = enc_out * mask
                    else:
                        enc_out_masked = enc_out

                    log_pi, mu, sigma, nu = model.decode(enc_out_masked, h)

                    # Main loss: NLL + CRPS on single target
                    from src.losses import combined_loss_v3
                    loss_main, nll, crps_val = combined_loss_v3(
                        log_pi, mu, sigma, target, nu,
                        nll_weight=args.nll_weight_v3,
                        crps_weight=args.crps_weight_v3,
                    )

                    # Auxiliary: asset-type classifier + vol regressor + sign classifier
                    _, vol_pred, asset_logits, sign_logits = model.forward_auxiliary(x)
                    loss_asset = F.cross_entropy(asset_logits, asset_labels)
                    loss_vol = F.mse_loss(vol_pred, rv_labels)
                    sign_labels = (target > 0).long()
                    loss_sign = F.cross_entropy(sign_logits, sign_labels)

                    loss_aux = loss_asset + loss_vol + args.sign_cls_weight * loss_sign
                    loss = loss_main + args.asset_cls_weight * loss_aux

                    # Encoder variance penalty
                    if args.enc_var_weight > 0:
                        from src.losses import encoder_variance_penalty
                        loss_evar = encoder_variance_penalty(enc_out)
                        loss = loss + args.enc_var_weight * loss_evar

                    loss = loss / args.grad_accum

                # Track metrics for v3
                primary = crps_val  # Use CRPS as primary metric
                sde_labels = asset_labels
                sde_logits = asset_logits
                yb = None

            else:
                # ── Synthetic data path (v1/v2) ──
                x, h, yb, sde_labels, rv_labels = batch
                x = x.to(device, non_blocking=True)
                h = h.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                sde_labels = sde_labels.to(device, non_blocking=True)
                rv_labels = rv_labels.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    enc_out = model.encode(x)

                    # Condition dropout
                    if model.training and model.cfg.cond_drop_prob > 0:
                        mask = torch.rand(x.size(0), 1, 1, device=enc_out.device) > model.cfg.cond_drop_prob
                        enc_out_masked = enc_out * mask
                    else:
                        enc_out_masked = enc_out

                    log_pi, mu, sigma, nu = model.decode(enc_out_masked, h)

                    # Main loss: ED or CRPS-avg + NLL (+ optional quantile)
                    loss_main, primary, nll = combined_loss(
                        log_pi, mu, sigma, yb,
                        n_model_samples=args.n_model_samples,
                        nll_weight=args.nll_weight,
                        nu=nu,
                        use_gumbel_softmax=args.use_gumbel_softmax,
                        quantile_weight=args.quantile_weight,
                        use_crps_avg=args.use_crps_avg,
                    )

                    # Auxiliary losses: SDE classification + vol regression
                    sde_logits, vol_pred, _, _ = model.forward_auxiliary(x)
                    loss_sde = F.cross_entropy(sde_logits, sde_labels)
                    loss_vol = F.mse_loss(vol_pred, rv_labels)
                    loss_aux = loss_sde + loss_vol

                    loss = loss_main + args.aux_weight * loss_aux

                    # Contrastive loss (InfoNCE)
                    if args.contrastive_weight > 0:
                        from src.losses import contrastive_loss
                        loss_contr = contrastive_loss(enc_out, yb)
                        loss = loss + args.contrastive_weight * loss_contr

                    # Encoder variance penalty
                    if args.enc_var_weight > 0:
                        from src.losses import encoder_variance_penalty
                        loss_evar = encoder_variance_penalty(enc_out)
                        loss = loss + args.enc_var_weight * loss_evar

                    # Moment matching losses
                    if args.mean_match_weight > 0 or args.var_match_weight > 0:
                        from src.losses import moment_matching_loss
                        mean_loss, var_loss = moment_matching_loss(log_pi, mu, sigma, yb, nu)
                        if args.mean_match_weight > 0:
                            loss = loss + args.mean_match_weight * mean_loss
                        if args.var_match_weight > 0:
                            loss = loss + args.var_match_weight * var_loss

                    loss = loss / args.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                    optimizer.step()
                optimizer.zero_grad()
            else:
                grad_norm = 0.0

            # ── Track detailed metrics ──
            with torch.no_grad():
                pi = log_pi.exp()
                eff_k = torch.exp(-torch.sum(pi * log_pi, dim=-1)).mean().item()

            sde_acc = (sde_logits.detach().argmax(-1) == sde_labels).float().mean().item()
            run['loss'] += loss.item() * args.grad_accum
            run['loss_main'] += loss_main.item()
            run['loss_aux'] += loss_aux.item()
            run['ed'] += primary.item()
            run['nll'] += nll.item()
            run['loss_sde'] += (loss_asset.item() if is_real else loss_sde.item())
            run['vol_mse'] += loss_vol.item()
            run['sde_acc'] += sde_acc
            run['mean_mu'] += mu.detach().abs().mean().item()
            run['mean_sigma'] += sigma.detach().mean().item()
            run['eff_k'] += eff_k
            run['grad_norm'] += grad_norm
            if nu is not None and (args.use_student_t or is_real):
                run['mean_nu'] = run.get('mean_nu', 0.) + nu.detach().mean().item()
            if is_v4 or is_v5:
                run['mean_mse'] = run.get('mean_mse', 0.) + mean_mse.item()
                run['pred_mean_std'] = run.get('pred_mean_std', 0.) + _pred_mean_std
            run_count += 1
            global_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.time() - t_start
                sps = global_step / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / sps if sps > 0 else 0

                metrics = {
                    'epoch': epoch, 'lr': lr,
                    'loss': run['loss'] / run_count,
                    'loss_main': run['loss_main'] / run_count,
                    'loss_aux': run['loss_aux'] / run_count,
                    'ed': run['ed'] / run_count,
                    'nll': run['nll'] / run_count,
                    'loss_sde': run['loss_sde'] / run_count,
                    'vol_mse': run['vol_mse'] / run_count,
                    'sde_acc': run['sde_acc'] / run_count,
                    'mean_mu': run['mean_mu'] / run_count,
                    'mean_sigma': run['mean_sigma'] / run_count,
                    'eff_k': run['eff_k'] / run_count,
                    'grad_norm': run['grad_norm'] / run_count,
                    'steps/s': sps, 'eta_min': eta / 60,
                }
                if 'mean_nu' in run and run['mean_nu'] != 0.:
                    metrics['mean_nu'] = run['mean_nu'] / run_count
                if (is_v4 or is_v5) and 'mean_mse' in run:
                    metrics['mean_mse'] = run['mean_mse'] / run_count
                    metrics['pred_mean_std'] = run['pred_mean_std'] / run_count
                logger.log(metrics, step=global_step)
                # Reset accumulators
                for k in run:
                    run[k] = 0.
                run_count = 0

            if global_step % args.val_every == 0:
                if is_v4 or is_v5:
                    val_m = validate_v4(model, val_x, val_y_curve, val_asset, val_rv,
                                        device, args.nll_weight_v3, args.crps_weight_v3,
                                        args.mean_mse_weight, args.horizon_weighting)
                elif is_v3:
                    val_m = validate_v3(model, val_x, val_h, val_y, val_asset, val_rv,
                                        device, args.nll_weight_v3, args.crps_weight_v3)
                else:
                    val_m = validate(model, val_x, val_h, val_yb, val_sde, val_rv,
                                     device, args.n_model_samples, args.nll_weight,
                                     use_gumbel_softmax=args.use_gumbel_softmax,
                                     quantile_weight=args.quantile_weight)
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
