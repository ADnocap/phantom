#!/bin/bash
#SBATCH --job-name=ph-ft-D
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ft_expD_%j.out
#SBATCH --error=logs/ft_expD_%j.err

# ─── Fine-tune ExpD on real BTC ──────────────────────────────────
# v2 improvements:
#   - Domain adaptation: anneal real fraction 10% → 70%
#   - Higher NLL weight (0.1) for conditional signal
#   - Quantile loss for calibration pressure
#   - Use Gumbel-Softmax for synthetic ED batches

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p logs checkpoints_ft_expD logs/ft_expD

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_expD/best.pt \
    --btc_cache data/btc_daily.npz \
    --val_start 2022-01-01 \
    --test_start 2023-07-01 \
    --steps 10000 \
    --batch_size 256 \
    --lr_head 1e-4 \
    --lr_encoder 3e-5 \
    --llrd 0.7 \
    --min_lr 1e-6 \
    --warmup_steps 300 \
    --weight_decay 0.02 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 2000 \
    --l2sp_lambda 0.01 \
    --nll_weight_real 0.1 \
    --nll_weight_synth 0.1 \
    --n_model_samples 256 \
    --n_branches 128 \
    --quantile_weight 0.2 \
    --use_gumbel_softmax \
    --anneal_real_fraction \
    --start_real_fraction 0.1 \
    --end_real_fraction 0.7 \
    --sde_version v1 \
    --log_every 50 \
    --val_every 500 \
    --save_every 1000 \
    --checkpoint_dir checkpoints_ft_expD \
    --log_dir logs/ft_expD \
    --seed 42"
