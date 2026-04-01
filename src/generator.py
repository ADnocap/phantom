"""
Sharded synthetic dataset generator with multiprocessing.

Generates .npz files containing:
  X : (n_samples, context_len) float32 — daily log-return context windows
  H : (n_samples,)             int8    — forecast horizon (3, 5, or 7)
  Y : (n_samples,)             float32 — cumulative forward log-return
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .sde import sample_params, simulate_daily_returns

SDE_TYPES = ['gbm', 'merton', 'kou', 'bates', 'regime_switching']
SDE_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.20]
HORIZONS = np.array([3, 5, 7])


def _generate_chunk(args):
    """Generate a chunk of samples (called by worker processes)."""
    chunk_size, context_len, seed = args
    rng = np.random.default_rng(seed)
    # Seed legacy numpy RNG too (Numba uses it)
    np.random.seed(rng.integers(0, 2**31))

    X = np.zeros((chunk_size, context_len), dtype=np.float32)
    H = np.zeros(chunk_size, dtype=np.int8)
    Y = np.zeros(chunk_size, dtype=np.float32)

    for i in range(chunk_size):
        sde_type = rng.choice(SDE_TYPES, p=SDE_WEIGHTS)
        params = sample_params(sde_type, rng=rng)
        h = int(rng.choice(HORIZONS))
        total_days = context_len + h

        returns = simulate_daily_returns(sde_type, params, total_days)

        X[i] = returns[:context_len]
        H[i] = h
        Y[i] = returns[context_len:].sum()

    return X, H, Y


def generate_shard(
    n_samples,
    output_path,
    context_len=60,
    n_workers=None,
    chunk_size=5000,
    base_seed=None,
):
    """Generate a single .npz shard.

    Parameters
    ----------
    n_samples : int
        Number of samples in this shard.
    output_path : str or Path
        Where to save the .npz file.
    context_len : int
        Length of the context window in days.
    n_workers : int or None
        Number of parallel workers. None = cpu_count.
    chunk_size : int
        Samples per worker task. Smaller = more granular progress bar.
    base_seed : int or None
        Base seed for reproducibility.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split work into chunks
    rng = np.random.default_rng(base_seed)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    sizes = [chunk_size] * (n_chunks - 1)
    sizes.append(n_samples - chunk_size * (n_chunks - 1))
    seeds = [int(rng.integers(0, 2**63)) for _ in range(n_chunks)]
    tasks = list(zip(sizes, [context_len] * n_chunks, seeds))

    # Pre-allocate output arrays
    X = np.zeros((n_samples, context_len), dtype=np.float32)
    H = np.zeros(n_samples, dtype=np.int8)
    Y = np.zeros(n_samples, dtype=np.float32)

    offset = 0
    with mp.Pool(n_workers) as pool:
        results = pool.imap_unordered(_generate_chunk, tasks)
        with tqdm(total=n_samples, desc=output_path.name, unit="samples") as pbar:
            for x_chunk, h_chunk, y_chunk in results:
                n = len(x_chunk)
                X[offset:offset + n] = x_chunk
                H[offset:offset + n] = h_chunk
                Y[offset:offset + n] = y_chunk
                offset += n
                pbar.update(n)

    np.savez(output_path, X=X, H=H, Y=Y)
    size_gb = output_path.stat().st_size / 1e9
    print(f"  Saved {output_path} ({size_gb:.2f} GB, {n_samples:,} samples)")


def generate_dataset(
    n_shards,
    n_samples_per_shard,
    output_dir="data",
    context_len=60,
    n_workers=None,
    chunk_size=5000,
    seed=None,
):
    """Generate the full sharded dataset.

    Parameters
    ----------
    n_shards : int
        Number of shard files to create.
    n_samples_per_shard : int
        Samples per shard.
    output_dir : str
        Directory for output .npz files.
    context_len : int
        Context window length in days.
    n_workers : int or None
        Parallel workers per shard.
    chunk_size : int
        Samples per worker task.
    seed : int or None
        Master seed for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    total = n_shards * n_samples_per_shard

    print(f"Generating {total:,} samples across {n_shards} shard(s)")
    print(f"  context_len={context_len}, workers={n_workers or mp.cpu_count()}")
    print()

    for shard_idx in range(n_shards):
        path = output_dir / f"synthetic_shard_{shard_idx}.npz"
        shard_seed = int(rng.integers(0, 2**63))
        generate_shard(
            n_samples=n_samples_per_shard,
            output_path=path,
            context_len=context_len,
            n_workers=n_workers,
            chunk_size=chunk_size,
            base_seed=shard_seed,
        )

    print(f"\nDone. {n_shards} shard(s) in {output_dir}/")
