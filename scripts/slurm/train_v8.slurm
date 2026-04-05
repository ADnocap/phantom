#!/bin/bash
#SBATCH --job-name=ph-v8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v8_%j.out
#SBATCH --error=logs/v8_%j.err

# ─── Phantom v8: More crypto assets, daily, 6 OHLCV channels ──
# Same architecture as v5 (6ch, 120 context, 30 horizons)
# Init from v5, crypto-only, early stopping patience=15
# Wider cross-section (150+ assets vs 60 in v6)

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT
mkdir -p checkpoints_v8 logs/v8 plots

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

# ─── Validate ──
echo "=== Validating v8 dataset ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c 'cd '"$PROJECT"' && python -c "
import numpy as np, json
meta = json.load(open(\"data/processed_v8/asset_meta.json\"))
print(f\"Assets: {meta[\\\"n_assets\\\"]}\")
for s in [\"train\", \"val\", \"test\"]:
    d = np.load(f\"data/processed_v8/{s}.npz\", allow_pickle=True)
    X = d[\"X\"]; Y = d[\"Y_relative\"]
    print(f\"{s}: {len(X):,} samples, X={X.shape}\")
    assert X.shape[1] == 120 and X.shape[2] == 6 and Y.shape[1] == 30
print(\"Validation passed!\")
"'

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset validation failed."
    exit 1
fi

# ─── Train ──
echo ""
echo "=== Starting v8 training ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/train/train_pretrain.py \
    --data_mode v5_real_assets \
    --real_data_dir data/processed_v8/ \
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
    --min_mse_horizon 5 \
    --enc_var_weight 0.1 \
    --epochs 50 \
    --batch_size 512 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --n_workers 8 \
    --log_every 50 --val_every 1000 --save_every 5000 \
    --checkpoint_dir checkpoints_v8 \
    --log_dir logs/v8 \
    --init_from checkpoints_v5/best.pt \
    --seed 42"

# ─── Evaluate ──
echo ""
echo "=== Evaluating ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/eval/eval_v5.py \
    --checkpoint checkpoints_v8/best.pt \
    --test_data data/processed_v8/test.npz \
    --output plots/eval_v8.png"

echo ""
echo "=== v8 complete ==="
