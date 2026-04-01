"""
Dataset classes for Phantom training.

Two modes:
  - ShardDataset:  loads pre-generated .npz shards from disk (fast, reproducible)
  - OnlineDataset: generates fresh SDE samples on-the-fly (infinite data, JointFM-style)

Both yield (x, h, y_branches) where y_branches contains N branched future returns.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .sde import sample_params, simulate_context_and_branches

SDE_TYPES = ['gbm', 'merton', 'kou', 'bates', 'regime_switching']
SDE_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.20]
HORIZONS = np.array([3, 5, 7])


# ── Shard-based dataset (pre-generated) ────────────────────────────

class ShardDataset(Dataset):
    """Load pre-generated .npz shards into memory.

    Each shard contains:
        X          : (N, context_len) float32
        H          : (N,)             int8
        Y_branches : (N, n_branches)  float32
    """

    def __init__(self, data_dir: str, context_len: int = 60):
        data_dir = Path(data_dir)
        shard_files = sorted(data_dir.glob("*.npz"))
        if not shard_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        X_parts, H_parts, Yb_parts = [], [], []
        for f in shard_files:
            d = np.load(f)
            X_parts.append(d['X'][:, :context_len])
            H_parts.append(d['H'])
            if 'Y_branches' in d:
                Yb_parts.append(d['Y_branches'])
            else:
                # Backward compat: old shards with scalar Y
                Yb_parts.append(d['Y'][:, None])

        self.X = np.concatenate(X_parts, axis=0)
        self.H = np.concatenate(H_parts, axis=0)
        self.Y_branches = np.concatenate(Yb_parts, axis=0)

        print(f"ShardDataset: loaded {len(self.X):,} samples from {len(shard_files)} shard(s), "
              f"branches={self.Y_branches.shape[1]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.H[idx], dtype=torch.long),
            torch.from_numpy(self.Y_branches[idx]),
        )


# ── Online dataset (generate on-the-fly) ───────────────────────────

class OnlineDataset(IterableDataset):
    """Generate synthetic SDE samples on-the-fly with branched futures.

    Each sample uses freshly sampled SDE parameters and branches N
    future paths from the context terminal state (JointFM-style).
    """

    def __init__(
        self,
        context_len: int = 60,
        n_branches: int = 128,
        samples_per_epoch: int = 1_000_000,
        seed: int | None = None,
    ):
        self.context_len = context_len
        self.n_branches = n_branches
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers
            per_worker = self.samples_per_epoch // n_workers
            seed = (self.seed or 0) + worker_id
        else:
            per_worker = self.samples_per_epoch
            seed = self.seed or 0

        rng = np.random.default_rng(seed)
        np.random.seed(rng.integers(0, 2**31))

        for _ in range(per_worker):
            sde_type = rng.choice(SDE_TYPES, p=SDE_WEIGHTS)
            params = sample_params(sde_type, rng=rng)
            h = int(rng.choice(HORIZONS))

            ctx, branches = simulate_context_and_branches(
                sde_type, params, self.context_len, h, self.n_branches
            )

            yield (
                torch.from_numpy(ctx.astype(np.float32)),
                torch.tensor(h, dtype=torch.long),
                torch.from_numpy(branches),
            )

    def __len__(self):
        return self.samples_per_epoch


# ── Validation set ─────────────────────────────────────────────────

def make_validation_batch(
    n_samples: int = 2048,
    context_len: int = 60,
    n_branches: int = 128,
    seed: int = 999,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a fixed validation batch with branched futures.

    Returns:
        X          : (n_samples, context_len) float32
        H          : (n_samples,) long
        Y_branches : (n_samples, n_branches) float32
    """
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(0, 2**31))

    X = np.zeros((n_samples, context_len), dtype=np.float32)
    H = np.zeros(n_samples, dtype=np.int64)
    Y_branches = np.zeros((n_samples, n_branches), dtype=np.float32)

    for i in range(n_samples):
        sde_type = rng.choice(SDE_TYPES, p=SDE_WEIGHTS)
        params = sample_params(sde_type, rng=rng)
        h = int(rng.choice(HORIZONS))

        ctx, branches = simulate_context_and_branches(
            sde_type, params, context_len, h, n_branches
        )

        X[i] = ctx
        H[i] = h
        Y_branches[i] = branches

    return torch.from_numpy(X), torch.from_numpy(H), torch.from_numpy(Y_branches)
