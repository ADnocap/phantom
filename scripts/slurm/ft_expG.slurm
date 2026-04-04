#!/bin/bash
#SBATCH --job-name=ph-ftG
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ftG_%j.out
#SBATCH --error=logs/ftG_%j.err

# ─── FT-G: 100% real data, no synthetic ──────────────────────────
# Hypothesis: synthetic data during fine-tuning is diluting BTC signal
# Fix: real_fraction=1.0 (pure BTC), longer training, moderate LR
# Uses ExpE pretrained checkpoint

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_ftG logs/ftG

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_expE/best.pt \
    --steps 50000 \
    --batch_size 128 \
    --lr_head 5e-5 \
    --lr_encoder 5e-5 \
    --llrd 0.85 \
    --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 500 \
    --l2sp_lambda 0.001 \
    --nll_weight_real 1.0 \
    --nll_weight_synth 0.1 \
    --real_fraction 1.0 \
    --n_branches 128 \
    --log_every 50 --val_every 500 --save_every 2000 \
    --checkpoint_dir checkpoints_ftG \
    --log_dir logs/ftG \
    --seed 42"
