#!/bin/bash
#SBATCH --job-name=ph-v3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3_%j.out
#SBATCH --error=logs/v3_%j.err

# ─── Phantom v3: Real Multi-Asset Pretraining ──────────────────
# 6-channel OHLCV features, ~1.8M real samples
# NLL + CRPS loss on single targets (no branches)
# Student-t head, asset-type + sign classifiers

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_v3 logs/v3

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_pretrain.py \
    --data_mode real_assets \
    --real_data_dir data/processed/ \
    --n_input_channels 6 \
    --head_type student_t \
    --context_len 75 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 8 \
    --d_ff 2048 \
    --n_decoder_layers 2 \
    --dropout 0.2 \
    --cond_drop_prob 0.15 \
    --nll_weight_v3 1.0 \
    --crps_weight_v3 0.5 \
    --asset_cls_weight 0.3 \
    --sign_cls_weight 0.1 \
    --enc_var_weight 0.1 \
    --epochs 10 \
    --batch_size 256 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --n_workers 4 \
    --log_every 100 --val_every 2000 --save_every 5000 \
    --checkpoint_dir checkpoints_v3 \
    --log_dir logs/v3 \
    --seed 42"
