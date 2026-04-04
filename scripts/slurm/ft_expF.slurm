#!/bin/bash
#SBATCH --job-name=ph-ftF
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ftF_%j.out
#SBATCH --error=logs/ftF_%j.err

# ─── FT-F: Aggressive encoder unfreezing ─────────────────────────
# Hypothesis: encoder is too frozen, preventing BTC-specific adaptation
# Fix: No freeze period, full LR on encoder, no L2-SP, 50K steps
# Uses ExpE (Student-t) pretrained checkpoint

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_ftF logs/ftF

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_expE/best.pt \
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
    --checkpoint_dir checkpoints_ftF \
    --log_dir logs/ftF \
    --seed 42"
