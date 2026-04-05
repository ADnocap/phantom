#!/bin/bash
#SBATCH --job-name=ph-v7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v7_%j.out
#SBATCH --error=logs/v7_%j.err

# ─── Phantom v7: 4h bars, crypto-only, 6 OHLCV channels ──
# 720-bar context (120 days), 90 horizons (15 days at 4h)
# patch_len=10 -> 72 patches
# Init encoder/decoder/head from v5, re-init patch_embed + pos_enc
# Early stopping patience=10

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_v7 logs/v7 plots

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

# ─── Validate dataset ──
echo "=== Validating v7 dataset ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c 'cd '"$PROJECT"' && python -c "
import numpy as np
for split in [\"train\", \"val\", \"test\"]:
    d = np.load(f\"data/processed_v7/{split}.npz\", allow_pickle=True)
    X = d[\"X\"]; Y = d[\"Y_relative\"]
    print(f\"{split}: {len(X):,} samples, X={X.shape}, Y={Y.shape}\")
    assert X.shape[1] == 720 and X.shape[2] == 6 and Y.shape[1] == 90
print(\"Dataset validation passed!\")
"'

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset validation failed."
    exit 1
fi

# ─── Train ──
echo ""
echo "=== Starting v7 training ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_v7.py \
    --v5_checkpoint checkpoints_v5/best.pt \
    --data_dir data/processed_v7 \
    --context_len 720 \
    --patch_len 10 \
    --d_model 512 \
    --n_heads 8 \
    --n_layers 8 \
    --d_ff 2048 \
    --n_decoder_layers 2 \
    --dropout 0.3 \
    --max_horizon 90 \
    --max_steps 50000 \
    --epochs 100 \
    --batch_size 128 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --nll_weight 1.0 \
    --crps_weight 0.5 \
    --mean_mse_weight 0.3 \
    --horizon_weighting sqrt \
    --min_mse_horizon 30 \
    --enc_var_weight 0.1 \
    --cond_drop_prob 0.15 \
    --patience 10 \
    --n_workers 8 \
    --log_every 50 \
    --val_every 1000 \
    --save_every 5000 \
    --checkpoint_dir checkpoints_v7 \
    --log_dir logs/v7 \
    --seed 42"

# ─── Evaluate ──
echo ""
echo "=== Evaluating best checkpoint ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/eval/eval_v7.py \
    --checkpoint checkpoints_v7/best.pt \
    --test_data data/processed_v7/test.npz \
    --output plots/eval_v7.png"

echo ""
echo "=== v7 pipeline complete ==="
