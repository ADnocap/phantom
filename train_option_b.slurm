#!/bin/bash
#SBATCH --job-name=phantom-qtl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_qtl_%j.out
#SBATCH --error=logs/train_qtl_%j.err

# ─── Option B: MeanAbsScaling + Quantile Head ────────────────────────
# No RevIN — preserves sign/shape/trends in the input
# Pinball loss at 9 quantile levels (Moirai-2 approach)

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p logs checkpoints_qtl

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
    --head_type quantile \
    --dropout 0.2 \
    --epochs 9 \
    --batch_size 512 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --n_workers 8 \
    --val_every 5000 \
    --save_every 5000 \
    --log_every 100 \
    --val_samples 4096 \
    --checkpoint_dir checkpoints_qtl/ \
    --log_dir logs_qtl/ \
    --seed 42
