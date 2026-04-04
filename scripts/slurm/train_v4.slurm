#!/bin/bash
#SBATCH --job-name=ph-v4b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err

# ─── Phantom v4b: Multi-Horizon + Synthetic Mix + Regularization ──
# Improvements over v4a:
#   - 15% synthetic GARCH/Momentum data (genuine conditional mean signal)
#   - mean_mse_weight reduced 1.0 → 0.3 (less overfitting pressure)
#   - min_mse_horizon=10 (don't force direction at 1-9 day horizons)
#   - dropout 0.3 (was 0.2)
#   - weight_decay 0.05 (was 0.01)

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_v4 logs/v4

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_pretrain.py \
    --data_mode v4_real_assets \
    --real_data_dir data/processed_v4/ \
    --n_input_channels 6 \
    --head_type student_t \
    --context_len 120 \
    --max_horizon 30 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 8 \
    --d_ff 2048 \
    --n_decoder_layers 2 \
    --dropout 0.3 \
    --cond_drop_prob 0.15 \
    --nll_weight_v3 1.0 \
    --crps_weight_v3 0.5 \
    --mean_mse_weight 0.3 \
    --horizon_weighting sqrt \
    --min_mse_horizon 10 \
    --synth_fraction 0.15 \
    --asset_cls_weight 0.3 \
    --enc_var_weight 0.1 \
    --epochs 10 \
    --batch_size 256 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --n_workers 4 \
    --log_every 100 --val_every 2000 --save_every 5000 \
    --checkpoint_dir checkpoints_v4 \
    --log_dir logs/v4 \
    --init_from checkpoints_v3/best.pt \
    --seed 42"
