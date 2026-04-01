#!/bin/bash
#SBATCH --job-name=phantom-datagen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=cpu_long
#SBATCH --time=24:00:00
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err

# ─── Phantom: Generate 150M synthetic SDE trajectories ───────────────
# 150 shards x 1M samples, context_len=75 (for Large model)
# CPU-only job — no GPU needed

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/phantom

cd $PROJECT
mkdir -p logs data

module load apptainer/1.4.4/gcc-15.1.0

apptainer exec --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp:/tmp \
  $WORKDIR/unsloth.sif \
  python generate.py \
    --n_shards 150 \
    --n_samples 1000000 \
    --context_len 75 \
    --n_workers 32 \
    --chunk_size 5000 \
    --seed 42 \
    --output_dir data/
