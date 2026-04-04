#!/bin/bash
#SBATCH --job-name=ph-ftI
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ftI_%j.out
#SBATCH --error=logs/ftI_%j.err

# ─── FT-I: Multiple cycles of freeze/unfreeze ────────────────────
# Hypothesis: single unfreeze at step 2K isn't enough
# Fix: 3 cycles — head-only → full → head-only → full → head-only → full
#      with decreasing LR each cycle, 100% real data
# Uses ExpA (CRPS+contrastive) pretrained checkpoint (best SDE acc)

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_ftI logs/ftI

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

# Cycle 1: head-only, high LR, 100% real
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_expA/best.pt \
    --steps 5000 \
    --batch_size 128 \
    --lr_head 3e-4 \
    --lr_encoder 0 \
    --llrd 0.0 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 999999 \
    --l2sp_lambda 0.0 \
    --nll_weight_real 1.0 \
    --nll_weight_synth 0.1 \
    --real_fraction 1.0 \
    --n_branches 128 \
    --log_every 50 --val_every 500 --save_every 5000 \
    --checkpoint_dir checkpoints_ftI \
    --log_dir logs/ftI \
    --seed 42"

# Cycle 2: full model, lower LR, 100% real
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_ftI/latest.pt \
    --steps 10000 \
    --batch_size 128 \
    --lr_head 1e-4 \
    --lr_encoder 3e-5 \
    --llrd 0.8 \
    --min_lr 1e-6 \
    --warmup_steps 300 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 0 \
    --l2sp_lambda 0.001 \
    --nll_weight_real 1.0 \
    --nll_weight_synth 0.1 \
    --real_fraction 0.7 \
    --n_branches 128 \
    --log_every 50 --val_every 500 --save_every 10000 \
    --checkpoint_dir checkpoints_ftI \
    --log_dir logs/ftI \
    --seed 43"

# Cycle 3: head-only refinement, very low LR, 100% real
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_ftI/latest.pt \
    --steps 5000 \
    --batch_size 128 \
    --lr_head 3e-5 \
    --lr_encoder 0 \
    --llrd 0.0 \
    --min_lr 1e-6 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 999999 \
    --l2sp_lambda 0.0 \
    --nll_weight_real 1.0 \
    --nll_weight_synth 0.1 \
    --real_fraction 1.0 \
    --n_branches 128 \
    --log_every 50 --val_every 500 --save_every 5000 \
    --checkpoint_dir checkpoints_ftI \
    --log_dir logs/ftI \
    --seed 44"
