#!/bin/bash
#SBATCH --job-name=phantom-ft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=4:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# ─── Phantom Phase 2: Fine-tune on real BTC ──────────────────────────
# Mixed batches: 70% synthetic (energy distance) + 30% real (CRPS)
# Gradual unfreezing: head-only for 2K steps, then full model with LLRD
# L2-SP regularization toward pre-trained weights

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p logs checkpoints_ft

module load apptainer/1.4.4/gcc-15.1.0

# Install ccxt inside container (needed for BTC data fetch)
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  python train_finetune.py \
    --pretrained checkpoints/best.pt \
    --btc_cache data/btc_daily.npz \
    --val_start 2022-01-01 \
    --test_start 2023-07-01 \
    --real_fraction 0.3 \
    --n_branches 128 \
    --steps 10000 \
    --batch_size 256 \
    --lr_head 1e-4 \
    --lr_encoder 3e-5 \
    --llrd 0.7 \
    --min_lr 1e-6 \
    --warmup_steps 300 \
    --weight_decay 0.02 \
    --freeze_encoder_steps 2000 \
    --l2sp_lambda 0.01 \
    --n_model_samples 256 \
    --nll_weight_synth 0.1 \
    --nll_weight_real 0.05 \
    --val_every 500 \
    --save_every 1000 \
    --log_every 50 \
    --seed 42
