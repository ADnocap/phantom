#!/bin/bash
#SBATCH --job-name=phantom-v3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v3_%j.out
#SBATCH --error=logs/train_v3_%j.err

# ─── Phantom v3: Cross-attn + aux tasks + condition dropout ──────────
# Three anti-collapse mechanisms:
#   1. Cross-attention decoder (horizon query attends to all patches)
#   2. Auxiliary heads (SDE type classifier + vol regressor)
#   3. Condition dropout (15%) for classifier-free guidance

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
    --data_mode online \
    --samples_per_epoch 1000000 \
    --n_branches 128 \
    --context_len 75 \
    --patch_len 5 \
    --patch_stride 5 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 8 \
    --d_ff 2048 \
    --n_components 5 \
    --n_decoder_layers 2 \
    --cond_drop_prob 0.15 \
    --dropout 0.2 \
    --epochs 30 \
    --batch_size 256 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --n_model_samples 256 \
    --nll_weight 0.1 \
    --aux_weight 0.5 \
    --n_workers 8 \
    --val_every 2000 \
    --save_every 5000 \
    --log_every 100 \
    --val_samples 2048 \
    --seed 42
