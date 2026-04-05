#!/bin/bash
#SBATCH --job-name=ph-pub
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19,ruche-gpu12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pub_%j.out
#SBATCH --error=logs/pub_%j.err

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom
cd $PROJECT

module load apptainer/1.4.4/gcc-15.1.0
export PYTHONPATH=$PROJECT:$PYTHONPATH

echo "=== Publication Analysis ==="
apptainer exec --nv --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  bash -c "cd $PROJECT && python scripts/eval/publication_analysis.py \
    --checkpoint checkpoints_v8/best.pt \
    --test_data data/processed_v8/test.npz \
    --output_dir plots"

echo "=== Done ==="
