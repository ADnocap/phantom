#!/bin/bash
#SBATCH --job-name=ph-expG
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/expG_%j.out
#SBATCH --error=logs/expG_%j.err

# ─── ExpG: Same as ExpF but MoG K=5 head (compare head types) ────
# Tests if the non-Markovian signal works with MoG too

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_expG logs/expG

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_pretrain.py \
    --data_mode online \
    --samples_per_epoch 1000000 \
    --n_branches 128 \
    --context_len 75 \
    --d_model 512 --n_heads 8 --n_layers 8 --d_ff 2048 \
    --n_components 5 --n_decoder_layers 2 \
    --cond_drop_prob 0.15 --dropout 0.2 \
    --epochs 30 --batch_size 256 \
    --lr 3e-4 --min_lr 1e-6 --warmup_steps 2000 \
    --weight_decay 0.01 --grad_clip 1.0 \
    --n_model_samples 256 \
    --nll_weight 1.0 \
    --aux_weight 0.3 \
    --n_workers 0 \
    --val_every 2000 --save_every 5000 --log_every 100 \
    --val_samples 2048 --seed 42 \
    --checkpoint_dir checkpoints_expG \
    --log_dir logs/expG \
    --use_crps_avg \
    --contrastive_weight 0.3 \
    --enc_var_weight 0.1 \
    --mean_match_weight 1.0 \
    --var_match_weight 0.5 \
    --sde_version v3"
