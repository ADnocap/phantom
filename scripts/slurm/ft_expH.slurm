#!/bin/bash
#SBATCH --job-name=ph-ftH
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ftH_%j.out
#SBATCH --error=logs/ftH_%j.err

# ─── FT-H: Heavy NLL + moment matching on real data ──────────────
# Hypothesis: CRPS on single targets has weak conditional gradient
# Fix: NLL weight 2.0 (strong conditional gradient) + moment matching
#      on mini-batches (match per-horizon statistics)
# Mixed training with annealing
# Uses ExpD (MoG + moment matching) pretrained checkpoint

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_ftH logs/ftH

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_finetune.py \
    --pretrained checkpoints_expD/best.pt \
    --steps 50000 \
    --batch_size 256 \
    --lr_head 1e-4 \
    --lr_encoder 5e-5 \
    --llrd 0.85 \
    --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --freeze_encoder_steps 500 \
    --l2sp_lambda 0.001 \
    --nll_weight_real 2.0 \
    --nll_weight_synth 0.1 \
    --anneal_real_fraction \
    --start_real_fraction 0.2 \
    --end_real_fraction 0.8 \
    --n_branches 128 \
    --log_every 50 --val_every 500 --save_every 2000 \
    --checkpoint_dir checkpoints_ftH \
    --log_dir logs/ftH \
    --seed 42"
