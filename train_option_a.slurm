#!/bin/bash
#SBATCH --job-name=phantom-mog
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_mog_%j.out
#SBATCH --error=logs/train_mog_%j.err

# ─── Option A: MeanAbsScaling + MoG (K=5) ────────────────────────────
# No RevIN — preserves sign/shape/trends in the input
# CRPS-dominant loss + entropy reg + sharpness penalty

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p logs checkpoints_mog

module load apptainer/1.4.4/gcc-15.1.0

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  python train_pretrain.py \
    --data_mode shards \
    --data_dir data/ \
    --context_len 75 \
    --patch_len 5 \
    --patch_stride 5 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 8 \
    --d_ff 2048 \
    --n_components 5 \
    --head_type mog \
    --dropout 0.2 \
    --epochs 9 \
    --batch_size 512 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --alpha 2.0 \
    --entropy_coeff 0.01 \
    --sharpness_coeff 0.1 \
    --n_workers 8 \
    --val_every 5000 \
    --save_every 5000 \
    --log_every 100 \
    --val_samples 4096 \
    --checkpoint_dir checkpoints_mog/ \
    --log_dir logs_mog/ \
    --seed 42
