#!/bin/bash
#SBATCH --job-name=ev-v8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_v8_%j.out
#SBATCH --error=logs/eval_v8_%j.err

# ─── Evaluate BOTH best.pt and latest.pt to compare ──

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

echo "=== Evaluating best.pt (step 24000, best val loss) ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/eval/eval_v5.py \
    --checkpoint checkpoints_v8/best.pt \
    --test_data data/processed_v8/test.npz \
    --output plots/eval_v8_best.png"

echo ""
echo "=== Evaluating latest.pt (final step, highest mu spread) ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/eval/eval_v5.py \
    --checkpoint checkpoints_v8/latest.pt \
    --test_data data/processed_v8/test.npz \
    --output plots/eval_v8_latest.png"

echo ""
echo "=== Both evals complete ==="
