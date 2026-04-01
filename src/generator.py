"""
Sharded synthetic dataset generator with multiprocessing.

Generates .npz files containing:
  X          : (n_samples, context_len) float32 — daily log-return context windows
  H          : (n_samples,)             int8    — forecast horizon (3, 5, or 7)
  Y_branches : (n_samples, n_branches)  float32 — branched future cumulative returns
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .sde import sample_params, simulate_context_and_branches

SDE_TYPES = ['gbm', 'merton', 'kou', 'bates', 'regime_switching']
SDE_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.20]
HORIZONS = np.array([3, 5, 7])


def _generate_chunk(args):
    """Generate a chunk of samples with branched futures."""
    chunk_size, context_len, n_branches, seed = args
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(0, 2**31))

    X = np.zeros((chunk_size, context_len), dtype=np.float32)
    H = np.zeros(chunk_size, dtype=np.int8)
    Y_branches = np.zeros((chunk_size, n_branches), dtype=np.float32)

    for i in range(chunk_size):
        sde_type = rng.choice(SDE_TYPES, p=SDE_WEIGHTS)
        params = sample_params(sde_type, rng=rng)
        h = int(rng.choice(HORIZONS))

        ctx, branches = simulate_context_and_branches(
            sde_type, params, context_len, h, n_branches
        )

        X[i] = ctx
        H[i] = h
        Y_branches[i] = branches

    return X, H, Y_branches


def generate_shard(
    n_samples,
    output_path,
    context_len=60,
    n_branches=128,
    n_workers=None,
    chunk_size=1000,
    base_seed=None,
):
    """Generate a single .npz shard with branched futures."""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(base_seed)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    sizes = [chunk_size] * (n_chunks - 1)
    sizes.append(n_samples - chunk_size * (n_chunks - 1))
    seeds = [int(rng.integers(0, 2**63)) for _ in range(n_chunks)]
    tasks = list(zip(sizes, [context_len] * n_chunks,
                     [n_branches] * n_chunks, seeds))

    X = np.zeros((n_samples, context_len), dtype=np.float32)
    H = np.zeros(n_samples, dtype=np.int8)
    Y_branches = np.zeros((n_samples, n_branches), dtype=np.float32)

    offset = 0
    with mp.Pool(n_workers) as pool:
        results = pool.imap_unordered(_generate_chunk, tasks)
        with tqdm(total=n_samples, desc=output_path.name, unit="samples") as pbar:
            for x_chunk, h_chunk, yb_chunk in results:
                n = len(x_chunk)
                X[offset:offset + n] = x_chunk
                H[offset:offset + n] = h_chunk
                Y_branches[offset:offset + n] = yb_chunk
                offset += n
                pbar.update(n)

    np.savez(output_path, X=X, H=H, Y_branches=Y_branches)
    size_gb = output_path.stat().st_size / 1e9
    print(f"  Saved {output_path} ({size_gb:.2f} GB, {n_samples:,} samples)")


def generate_dataset(
    n_shards,
    n_samples_per_shard,
    output_dir="data",
    context_len=60,
    n_branches=128,
    n_workers=None,
    chunk_size=1000,
    seed=None,
):
    """Generate the full sharded dataset with branched futures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    total = n_shards * n_samples_per_shard

    print(f"Generating {total:,} samples across {n_shards} shard(s)")
    print(f"  context_len={context_len}, n_branches={n_branches}, workers={n_workers or mp.cpu_count()}")
    print()

    for shard_idx in range(n_shards):
        path = output_dir / f"synthetic_shard_{shard_idx}.npz"
        shard_seed = int(rng.integers(0, 2**63))
        generate_shard(
            n_samples=n_samples_per_shard,
            output_path=path,
            context_len=context_len,
            n_branches=n_branches,
            n_workers=n_workers,
            chunk_size=chunk_size,
            base_seed=shard_seed,
        )

    print(f"\nDone. {n_shards} shard(s) in {output_dir}/")
