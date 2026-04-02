#!/usr/bin/env python
"""
Phase 2: Fine-tune pre-trained Phantom on real BTC data.

Mixed batches: 70% synthetic (energy distance) + 30% real (CRPS).
Gradual unfreezing with layer-wise learning rate decay.

Usage:
  python train_finetune.py --pretrained checkpoints/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.model import PhantomConfig, PhantomModel
from src.losses import combined_loss, crps_loss, nll_loss
from src.data import OnlineDataset
from src.btc_data import fetch_btc_daily, temporal_split


# ── Real BTC Dataset ────────────────────────────────────────────────

class RealBTCDataset(Dataset):
    """Dataset of real BTC rolling windows."""

    def __init__(self, X, H, Y):
        self.X = torch.from_numpy(X)
        self.H = torch.from_numpy(H.astype(np.int64))
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.H[idx], self.Y[idx]


# ── Mixed DataLoader (synthetic + real) ─────────────────────────────

class MixedBatchSampler:
    """Yields batches with ~70% synthetic + ~30% real samples."""

    def __init__(self, real_dataset, synthetic_dataset, batch_size=256,
                 real_fraction=0.3, steps_per_epoch=1000):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        self.batch_size = batch_size
        self.n_real = max(1, int(batch_size * real_fraction))
        self.n_synth = batch_size - self.n_real
        self.steps_per_epoch = steps_per_epoch

    def get_batch(self, epoch, step, device):
        """Get a mixed batch of real + synthetic data."""
        # Real samples (random with replacement)
        real_idxs = torch.randint(0, len(self.real), (self.n_real,))
        rx, rh, ry = [], [], []
        for idx in real_idxs:
            x, h, y = self.real[idx]
            rx.append(x); rh.append(h); ry.append(y)
        real_x = torch.stack(rx).to(device)
        real_h = torch.stack(rh).to(device)
        real_y = torch.stack(ry).to(device)

        # Synthetic samples (from iterator)
        synth_x, synth_h, synth_yb = [], [], []
        for _ in range(self.n_synth):
            x, h, yb = next(self._synth_iter)
            synth_x.append(x); synth_h.append(h); synth_yb.append(yb)
        synth_x = torch.stack(synth_x).to(device)
        synth_h = torch.stack(synth_h).to(device)
        synth_yb = torch.stack(synth_yb).to(device)

        return real_x, real_h, real_y, synth_x, synth_h, synth_yb

    def init_synthetic_iter(self, seed):
        """Initialize the synthetic data iterator."""
        loader = DataLoader(
            self.synthetic, batch_size=1, num_workers=0, drop_last=True)
        # Flatten: DataLoader wraps each sample in an extra dim
        def flatten_iter():
            for x, h, yb in loader:
                yield x.squeeze(0), h.squeeze(0), yb.squeeze(0)
        self._synth_iter = flatten_iter()


# ── LR schedule ─────────────────────────────────────────────────────

def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ── Logger ──────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "finetune_log.csv"
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
            parts = [f"step {step:>6d}"]
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts))


# ── Validation ──────────────────────────────────────────────────────

@torch.no_grad()
def validate_real(model, val_X, val_H, val_Y, device, batch_size=512):
    """Validate on real BTC data using CRPS."""
    model.eval()
    total_crps, total_nll, n = 0.0, 0.0, 0
    for i in range(0, len(val_X), batch_size):
        x = val_X[i:i+batch_size].to(device)
        h = val_H[i:i+batch_size].to(device)
        y = val_Y[i:i+batch_size].to(device)
        log_pi, mu, sigma = model(x, h)
        total_crps += crps_loss(log_pi, mu, sigma, y).item()
        total_nll += nll_loss(log_pi, mu, sigma, y).item()
        n += 1
    model.train()
    return {
        'val_crps': total_crps / n,
        'val_nll': total_nll / n,
    }


# ── Checkpoint ──────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, step, best_val, config):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_crps': best_val,
        'config': vars(config),
    }, path)


# ── Main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phantom Phase 2: fine-tune on real BTC")

    # Pre-trained model
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to pre-trained checkpoint (best.pt)')

    # Real data
    p.add_argument('--btc_cache', type=str, default='data/btc_daily.npz')
    p.add_argument('--val_start', type=str, default='2022-01-01')
    p.add_argument('--test_start', type=str, default='2023-07-01')

    # Mixed training
    p.add_argument('--real_fraction', type=float, default=0.3,
                   help='Fraction of real data in each batch')
    p.add_argument('--n_branches', type=int, default=128)

    # Training
    p.add_argument('--steps', type=int, default=10000,
                   help='Total fine-tuning steps')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr_head', type=float, default=1e-4,
                   help='Peak LR for head + horizon embed')
    p.add_argument('--lr_encoder', type=float, default=3e-5,
                   help='Peak LR for encoder (top layer)')
    p.add_argument('--llrd', type=float, default=0.7,
                   help='Layer-wise LR decay factor')
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup_steps', type=int, default=300)
    p.add_argument('--weight_decay', type=float, default=0.02)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--freeze_encoder_steps', type=int, default=2000,
                   help='Steps to freeze encoder (head-only training)')
    p.add_argument('--l2sp_lambda', type=float, default=0.01,
                   help='L2-SP regularization toward pre-trained weights')

    # Loss
    p.add_argument('--nll_weight_real', type=float, default=0.05)
    p.add_argument('--n_model_samples', type=int, default=256)
    p.add_argument('--nll_weight_synth', type=float, default=0.1)

    # Logging
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--val_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_ft/')
    p.add_argument('--log_dir', type=str, default='logs_ft/')

    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()
    if args.no_amp:
        args.amp = False

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load pre-trained model ──
    print(f"Loading pre-trained model from {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = PhantomConfig(**{k: v for k, v in ckpt['config'].items()
                          if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    pretrained_state = {k: v.clone().to(device) for k, v in model.state_dict().items()}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | K={cfg.n_components} | pre-train step={ckpt['step']:,}")

    # ── Fetch real BTC data ──
    print("\nFetching BTC data...")
    btc = fetch_btc_daily(cache_path=args.btc_cache)
    splits = temporal_split(
        btc['dates'], btc['log_returns'],
        context_len=cfg.context_len,
        val_start=args.val_start,
        test_start=args.test_start,
    )

    train_real = RealBTCDataset(*splits['train'])
    val_real = RealBTCDataset(*splits['val'])
    print(f"Real data: train={len(train_real)}, val={len(val_real)}")

    # ── Synthetic data (online, same as pre-training) ──
    synth_dataset = OnlineDataset(
        context_len=cfg.context_len,
        n_branches=args.n_branches,
        samples_per_epoch=100_000_000,  # effectively infinite
        seed=args.seed,
    )

    # ── Mixed batch helper ──
    mixer = MixedBatchSampler(
        train_real, synth_dataset,
        batch_size=args.batch_size,
        real_fraction=args.real_fraction,
    )
    mixer.init_synthetic_iter(args.seed)

    # ── Optimizer with layer-wise LR decay ──
    param_groups = []

    # Head + horizon embed: highest LR
    head_params = list(model.head.parameters()) + list(model.horizon_embed.parameters())
    param_groups.append({'params': head_params, 'lr': args.lr_head, 'name': 'head'})

    # Encoder layers: decaying LR from top to bottom
    n_layers = cfg.n_layers
    for layer_idx in range(n_layers - 1, -1, -1):
        layer = model.encoder.layers[layer_idx]
        depth_from_top = n_layers - 1 - layer_idx
        layer_lr = args.lr_encoder * (args.llrd ** depth_from_top)
        param_groups.append({
            'params': list(layer.parameters()),
            'lr': layer_lr,
            'name': f'encoder.layer.{layer_idx}',
        })

    # Patch embed + positional: lowest LR
    embed_lr = args.lr_encoder * (args.llrd ** n_layers)
    embed_params = (list(model.patch_embed.parameters()) +
                    [model.cls_token, model.pos_enc])
    if hasattr(model.encoder, 'norm') and model.encoder.norm is not None:
        embed_params += list(model.encoder.norm.parameters())
    param_groups.append({'params': embed_params, 'lr': embed_lr, 'name': 'embeddings'})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Print LR schedule
    print("\nLayer-wise learning rates:")
    for pg in param_groups:
        print(f"  {pg['name']}: {pg['lr']:.2e}")

    # ── AMP ──
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
        print("AMP: disabled")

    # ── Logging ──
    logger = Logger(Path(args.log_dir))
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "config_ft.json", 'w') as f:
        json.dump({**vars(cfg), **vars(args)}, f, indent=2, default=str)

    # ── Training loop ──
    model.train()
    best_val_crps = float('inf')
    running_loss, running_real_crps, running_synth_ed = 0.0, 0.0, 0.0
    running_count = 0
    t_start = time.time()

    # Initially freeze encoder
    encoder_frozen = True
    for name, param in model.named_parameters():
        if 'head' not in name and 'horizon' not in name:
            param.requires_grad = False
    print(f"\n{'='*60}")
    print(f"Phase 2a: head-only for {args.freeze_encoder_steps} steps")
    print(f"{'='*60}\n")

    for step in range(1, args.steps + 1):
        # Unfreeze encoder after warmup phase
        if encoder_frozen and step > args.freeze_encoder_steps:
            encoder_frozen = False
            for param in model.parameters():
                param.requires_grad = True
            print(f"\n{'='*60}")
            print(f"Phase 2b: full model unfrozen at step {step}")
            print(f"{'='*60}\n")

        # LR schedule (cosine decay)
        frac = get_lr(step, args.warmup_steps, args.steps, 1.0, args.min_lr / args.lr_head)
        for pg in param_groups:
            base_lr = pg.get('lr', args.lr_head)
            pg['lr'] = base_lr * frac if not encoder_frozen or 'head' in pg['name'] else 0.0

        # Get mixed batch
        real_x, real_h, real_y, synth_x, synth_h, synth_yb = mixer.get_batch(0, step, device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # ── Real data: CRPS + aux NLL ──
            lp_r, mu_r, sig_r = model(real_x, real_h)
            loss_real_crps = crps_loss(lp_r, mu_r, sig_r, real_y)
            loss_real_nll = nll_loss(lp_r, mu_r, sig_r, real_y)
            loss_real = loss_real_crps + args.nll_weight_real * loss_real_nll

            # ── Synthetic data: energy distance + aux NLL ──
            lp_s, mu_s, sig_s = model(synth_x, synth_h)
            loss_synth, ed_val, nll_s_val = combined_loss(
                lp_s, mu_s, sig_s, synth_yb,
                n_model_samples=args.n_model_samples,
                nll_weight=args.nll_weight_synth,
            )

            # ── Combined ──
            n_r = real_x.size(0)
            n_s = synth_x.size(0)
            loss = (loss_real * n_r + loss_synth * n_s) / (n_r + n_s)

            # ── L2-SP regularization ──
            if args.l2sp_lambda > 0 and not encoder_frozen:
                l2sp = 0.0
                for name, param in model.named_parameters():
                    if name in pretrained_state and param.requires_grad:
                        l2sp += torch.sum((param - pretrained_state[name]) ** 2)
                loss = loss + args.l2sp_lambda * l2sp

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        # Tracking
        running_loss += loss.item()
        running_real_crps += loss_real_crps.item()
        running_synth_ed += ed_val.item()
        running_count += 1

        # Logging
        if step % args.log_every == 0:
            avg_loss = running_loss / running_count
            avg_crps = running_real_crps / running_count
            avg_ed = running_synth_ed / running_count
            elapsed = time.time() - t_start
            sps = step / elapsed

            logger.log({
                'loss': avg_loss,
                'real_crps': avg_crps,
                'synth_ed': avg_ed,
                'steps/s': sps,
            }, step=step)
            running_loss, running_real_crps, running_synth_ed, running_count = 0, 0, 0, 0

        # Validation
        if step % args.val_every == 0:
            val_metrics = validate_real(model, val_real.X, val_real.H, val_real.Y, device)
            logger.log(val_metrics, step=step)

            if val_metrics['val_crps'] < best_val_crps:
                best_val_crps = val_metrics['val_crps']
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, step, best_val_crps, cfg)
                print(f"  >> New best val_crps: {best_val_crps:.4f}")

        # Periodic save
        if step % args.save_every == 0:
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, best_val_crps, cfg)

    # Final
    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, args.steps, best_val_crps, cfg)
    elapsed = time.time() - t_start
    print(f"\nFine-tuning complete. {args.steps} steps in {elapsed/60:.1f} min.")
    print(f"Best val CRPS: {best_val_crps:.4f}")


if __name__ == "__main__":
    main()
