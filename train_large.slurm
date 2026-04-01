#!/bin/bash
#SBATCH --job-name=phantom-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ─── Phantom: Phase 1 pre-training — Large (75d, d=512, 8L) ─────────
# 150M samples from shards, A100 GPU, 24h max
# Saves only best.pt and latest.pt
# Resumed from ~step 400K with corrected LR schedule (9 epochs)

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom

cd $PROJECT
mkdir -p logs checkpoints

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
    --n_components 8 \
    --dropout 0.2 \
    --resume checkpoints/latest.pt \
    --epochs 9 \
    --batch_size 512 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 0 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --alpha 0.3 \
    --n_workers 8 \
    --val_every 5000 \
    --save_every 5000 \
    --log_every 100 \
    --val_samples 4096 \
    --seed 42
