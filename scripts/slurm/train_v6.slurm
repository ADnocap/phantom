#!/bin/bash
#SBATCH --job-name=ph-v6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v6_%j.out
#SBATCH --error=logs/v6_%j.err

# ─── Phantom v6: Crypto-Focused + 3 New Features ──
# 8-channel input (6 OHLCV + taker buy ratio + funding rate)
# Crypto-only, init from v5 checkpoint
# Phase A: 5K steps patch embed warmup
# Phase B: up to 45K steps full fine-tuning with LLRD + early stopping
# Early stop: patience=10 (10 val checks = 5K steps without improvement)

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_v6 logs/v6

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

# ─── Step 0: Validate dataset before training ──
echo "=== Validating v6 dataset ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python -c \"
import numpy as np
for split in ['train', 'val', 'test']:
    d = np.load(f'data/processed_v6/{split}.npz', allow_pickle=True)
    X = d['X']
    n = len(X)
    print(f'{split}: {n:,} samples, X shape: {X.shape}')
    assert X.shape[2] == 8, f'Expected 8 channels, got {X.shape[2]}'
    for ch, name in [(6,'taker_buy'), (7,'funding')]:
        nz = (X[:,:,ch] != 0).any(axis=1).mean()
        print(f'  ch{ch} ({name}): {nz*100:.1f}% nonzero')
    bad = ~np.isfinite(X).all(axis=(1,2))
    if bad.any():
        print(f'  WARNING: {bad.sum()} samples have NaN/Inf')
print('Dataset validation passed!')
\""

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset validation failed. Aborting."
    exit 1
fi

# ─── Step 1: Train ──
echo ""
echo "=== Starting v6 training ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_v6.py \
    --v5_checkpoint checkpoints_v5/best.pt \
    --data_dir data/processed_v6 \
    --context_len 120 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 8 \
    --d_ff 2048 \
    --n_decoder_layers 2 \
    --dropout 0.3 \
    --max_horizon 30 \
    --phase_a_steps 5000 \
    --max_steps 50000 \
    --epochs 100 \
    --lr_warmup_a 3e-4 \
    --lr_head 1e-4 \
    --lr_encoder 5e-5 \
    --llrd 0.8 \
    --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --nll_weight 1.0 \
    --crps_weight 0.5 \
    --mean_mse_weight 1.0 \
    --horizon_weighting sqrt \
    --min_mse_horizon 5 \
    --enc_var_weight 0.1 \
    --cond_drop_prob 0.10 \
    --feature_mask_prob 0.15 \
    --patience 10 \
    --batch_size 256 \
    --n_workers 8 \
    --log_every 50 \
    --val_every 500 \
    --save_every 2500 \
    --checkpoint_dir checkpoints_v6 \
    --log_dir logs/v6 \
    --seed 42"

# ─── Step 2: Evaluate ──
echo ""
echo "=== Evaluating best checkpoint ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/eval/eval_v6.py \
    --checkpoint checkpoints_v6/best.pt \
    --test_data data/processed_v6/test.npz \
    --ablation"

echo ""
echo "=== v6 pipeline complete ==="
