#!/bin/bash
#SBATCH --job-name=ph-ftEF
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ftEF_%j.out
#SBATCH --error=logs/ftEF_%j.err

# ─── FT on ExpF: Non-Markovian pretrained → BTC ─────────────────
# Aggressive encoder (what worked in FT-F):
#   - No freeze, full LR on encoder, no L2-SP
#   - 50K steps, 30% real
# The model learned vol conditioning (corr 0.80) from GARCH pretraining
# This should transfer to BTC's volatility clustering

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_ftEF logs/ftEF

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_expF/best.pt \
    --steps 50000 \
    --batch_size 256 \
    --lr_head 1e-4 \
    --lr_encoder 1e-4 \
    --llrd 1.0 \
    --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 0 \
    --l2sp_lambda 0.0 \
    --nll_weight_real 0.5 \
    --nll_weight_synth 0.1 \
    --real_fraction 0.3 \
    --n_branches 128 \
    --log_every 50 --val_every 500 --save_every 2000 \
    --checkpoint_dir checkpoints_ftEF \
    --log_dir logs/ftEF \
    --sde_version v3 \
    --seed 42"
