"""
Dataset classes for Phantom training.

Two modes:
  - ShardDataset:  loads pre-generated .npz shards from disk (fast, reproducible)
  - OnlineDataset: generates fresh SDE samples on-the-fly (infinite data, JointFM-style)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .sde import sample_params, simulate_daily_returns

SDE_TYPES = ['gbm', 'merton', 'kou', 'bates', 'regime_switching']
SDE_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.20]
HORIZONS = np.array([3, 5, 7])


# ── Shard-based dataset (pre-generated) ────────────────────────────

class ShardDataset(Dataset):
    """Load pre-generated .npz shards into memory.

    Each shard contains:
        X: (N, context_len) float32 — daily log-return context windows
        H: (N,)             int8    — forecast horizon (3, 5, or 7)
        Y: (N,)             float32 — cumulative forward log-return
    """

    def __init__(self, data_dir: str, context_len: int = 60):
        data_dir = Path(data_dir)
        shard_files = sorted(data_dir.glob("*.npz"))
        if not shard_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        X_parts, H_parts, Y_parts = [], [], []
        for f in shard_files:
            d = np.load(f)
            X_parts.append(d['X'][:, :context_len])
            H_parts.append(d['H'])
            Y_parts.append(d['Y'])

        self.X = np.concatenate(X_parts, axis=0)
        self.H = np.concatenate(H_parts, axis=0)
        self.Y = np.concatenate(Y_parts, axis=0)

        print(f"ShardDataset: loaded {len(self.X):,} samples from {len(shard_files)} shard(s)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.H[idx], dtype=torch.long),
            torch.tensor(self.Y[idx], dtype=torch.float32),
        )


# ── Online dataset (generate on-the-fly) ───────────────────────────

class OnlineDataset(IterableDataset):
    """Generate synthetic SDE samples on-the-fly in each DataLoader worker.

    Follows JointFM's "infinite stream" paradigm: each sample uses freshly
    sampled SDE parameters, so the model effectively never sees the same
    trajectory twice.

    Args:
        context_len:      Length of context window in days.
        samples_per_epoch: How many samples to yield before the epoch ends.
                           Set to a large number for effectively infinite data.
        seed:             Base seed (each worker offsets by worker_id).
    """

    def __init__(
        self,
        context_len: int = 60,
        samples_per_epoch: int = 1_000_000,
        seed: int | None = None,
    ):
        self.context_len = context_len
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers
            # Each worker handles a slice of the epoch
            per_worker = self.samples_per_epoch // n_workers
            seed = (self.seed or 0) + worker_id
        else:
            per_worker = self.samples_per_epoch
            seed = self.seed or 0

        rng = np.random.default_rng(seed)
        # Seed legacy numpy RNG for numba compatibility
        np.random.seed(rng.integers(0, 2**31))

        for _ in range(per_worker):
            sde_type = rng.choice(SDE_TYPES, p=SDE_WEIGHTS)
            params = sample_params(sde_type, rng=rng)
            h = int(rng.choice(HORIZONS))
            total_days = self.context_len + h

            returns = simulate_daily_returns(sde_type, params, total_days)

            x = returns[:self.context_len].astype(np.float32)
            y = returns[self.context_len:].sum().astype(np.float32)

            yield (
                torch.from_numpy(x),
                torch.tensor(h, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32),
            )

    def __len__(self):
        return self.samples_per_epoch


# ── Validation set (small, fresh, for tracking metrics) ────────────

def make_validation_batch(
    n_samples: int = 2048,
    context_len: int = 60,
    seed: int = 999,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a fixed validation batch of synthetic data.

    Returns:
        X: (n_samples, context_len) float32
        H: (n_samples,) long
        Y: (n_samples,) float32
    """
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(0, 2**31))

    X = np.zeros((n_samples, context_len), dtype=np.float32)
    H = np.zeros(n_samples, dtype=np.int64)
    Y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        sde_type = rng.choice(SDE_TYPES, p=SDE_WEIGHTS)
        params = sample_params(sde_type, rng=rng)
        h = int(rng.choice(HORIZONS))
        total_days = context_len + h

        returns = simulate_daily_returns(sde_type, params, total_days)
        X[i] = returns[:context_len]
        H[i] = h
        Y[i] = returns[context_len:].sum()

    return torch.from_numpy(X), torch.from_numpy(H), torch.from_numpy(Y)
