"""
Dataset classes for Phantom training.

Two modes:
  - ShardDataset:  loads pre-generated .npz shards from disk (fast, reproducible)
  - OnlineDataset: generates fresh SDE samples on-the-fly (infinite data, JointFM-style)

Both yield (x, h, y_branches) where y_branches contains N branched future returns.

v2 additions:
  - SDE v2 families (MRW + Fractional OU)
  - Multi-channel input features (trailing realized vol)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .sde import sample_params, simulate_context_and_branches

# ── SDE family configurations ────────────────────────────────────

SDE_TYPES = ['gbm', 'merton', 'kou', 'bates', 'regime_switching']
SDE_TYPE_TO_IDX = {t: i for i, t in enumerate(SDE_TYPES)}
SDE_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.20]

# v2: adds Multifractal Random Walk + Fractional OU
SDE_TYPES_V2 = SDE_TYPES + ['mrw', 'frac_ou']
SDE_TYPE_TO_IDX_V2 = {t: i for i, t in enumerate(SDE_TYPES_V2)}
SDE_WEIGHTS_V2 = [0.04, 0.12, 0.22, 0.22, 0.15, 0.15, 0.10]

# v3: heavy weight on non-Markovian SDEs (context carries predictive signal)
SDE_TYPES_V3 = SDE_TYPES + ['garch', 'momentum']
SDE_TYPE_TO_IDX_V3 = {t: i for i, t in enumerate(SDE_TYPES_V3)}
# 50% non-Markovian (GARCH + momentum), 30% regime-switching, 20% stationary
SDE_WEIGHTS_V3 = [0.02, 0.05, 0.05, 0.08, 0.30, 0.30, 0.20]

HORIZONS = np.array([3, 5, 7])


# ── Feature computation ──────────────────────────────────────────

def compute_vol_features(returns: np.ndarray, windows: tuple = (7, 14, 30)) -> np.ndarray:
    """Compute trailing realized vol features from log-returns.

    Args:
        returns: (L,) daily log-returns.
        windows: Trailing window sizes in days.

    Returns:
        features: (L, len(windows)) trailing annualized vol at each window.
    """
    L = len(returns)
    features = np.zeros((L, len(windows)), dtype=np.float32)
    for j, w in enumerate(windows):
        # Vectorized rolling std
        for t in range(L):
            start = max(0, t - w + 1)
            window = returns[start:t + 1]
            if len(window) > 1:
                features[t, j] = np.std(window) * np.sqrt(365)
    return features


def _build_input(ctx: np.ndarray, n_input_channels: int) -> np.ndarray:
    """Build model input from raw context returns.

    Args:
        ctx: (L,) daily log-returns.
        n_input_channels: 1 = returns only, 4 = returns + 3 vol features.

    Returns:
        (L,) if n_input_channels == 1, else (L, C) float32.
    """
    if n_input_channels == 1:
        return ctx.astype(np.float32)
    else:
        vol_feats = compute_vol_features(ctx)  # (L, 3)
        returns_col = ctx.astype(np.float32).reshape(-1, 1)  # (L, 1)
        return np.concatenate([returns_col, vol_feats], axis=1)  # (L, 4)


# ── Helpers ───────────────────────────────────────────────────────

def _get_sde_config(sde_version: str):
    """Return (types, type_to_idx, weights) for the given SDE version."""
    if sde_version == 'v3':
        return SDE_TYPES_V3, SDE_TYPE_TO_IDX_V3, SDE_WEIGHTS_V3
    if sde_version == 'v2':
        return SDE_TYPES_V2, SDE_TYPE_TO_IDX_V2, SDE_WEIGHTS_V2
    return SDE_TYPES, SDE_TYPE_TO_IDX, SDE_WEIGHTS


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
        n_input_channels: int = 1,
        sde_version: str = 'v1',
    ):
        self.context_len = context_len
        self.n_branches = n_branches
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.n_input_channels = n_input_channels
        self.sde_version = sde_version

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

        sde_types, type_to_idx, weights = _get_sde_config(self.sde_version)

        for _ in range(per_worker):
            sde_type = rng.choice(sde_types, p=weights)
            params = sample_params(sde_type, rng=rng)
            h = int(rng.choice(HORIZONS))

            ctx, branches = simulate_context_and_branches(
                sde_type, params, self.context_len, h, self.n_branches
            )

            sde_idx = type_to_idx[sde_type]
            realized_vol = np.std(ctx) * np.sqrt(365)  # annualized

            x = _build_input(ctx, self.n_input_channels)

            yield (
                torch.from_numpy(x),
                torch.tensor(h, dtype=torch.long),
                torch.from_numpy(branches),
                torch.tensor(sde_idx, dtype=torch.long),
                torch.tensor(realized_vol, dtype=torch.float32),
            )

    def __len__(self):
        return self.samples_per_epoch


# ── Validation set ─────────────────────────────────────────────────

def make_validation_batch(
    n_samples: int = 2048,
    context_len: int = 60,
    n_branches: int = 128,
    seed: int = 999,
    n_input_channels: int = 1,
    sde_version: str = 'v1',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a fixed validation batch with branched futures.

    Returns:
        X          : (n_samples, context_len) or (n_samples, context_len, C) float32
        H          : (n_samples,) long
        Y_branches : (n_samples, n_branches) float32
        SDE_idx    : (n_samples,) long
        RV         : (n_samples,) float32
    """
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(0, 2**31))

    sde_types, type_to_idx, weights = _get_sde_config(sde_version)

    # Determine X shape
    if n_input_channels > 1:
        X = np.zeros((n_samples, context_len, n_input_channels), dtype=np.float32)
    else:
        X = np.zeros((n_samples, context_len), dtype=np.float32)

    H = np.zeros(n_samples, dtype=np.int64)
    Y_branches = np.zeros((n_samples, n_branches), dtype=np.float32)
    SDE_idx = np.zeros(n_samples, dtype=np.int64)
    RV = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        sde_type = rng.choice(sde_types, p=weights)
        params = sample_params(sde_type, rng=rng)
        h = int(rng.choice(HORIZONS))

        ctx, branches = simulate_context_and_branches(
            sde_type, params, context_len, h, n_branches
        )

        X[i] = _build_input(ctx, n_input_channels)
        H[i] = h
        Y_branches[i] = branches
        SDE_idx[i] = type_to_idx[sde_type]
        RV[i] = np.std(ctx) * np.sqrt(365)

    return (torch.from_numpy(X), torch.from_numpy(H), torch.from_numpy(Y_branches),
            torch.from_numpy(SDE_idx), torch.from_numpy(RV))


# ── Synthetic curve dataset (v4 — multi-horizon) ─────────────────

class SyntheticCurveDataset(IterableDataset):
    """Generate synthetic SDE samples with multi-horizon curve targets.

    Each sample: (x, y_curve, asset_type=0, realized_vol)
    - x: (context_len, 6) — 6-channel features from synthetic OHLCV
    - y_curve: (max_horizon,) — cumulative returns at horizons 1..max_horizon
    - asset_type: always 0 (synthetic)
    - realized_vol: trailing vol from context

    Uses GARCH + Momentum SDEs (non-Markovian, have genuine conditional signal).
    """

    def __init__(
        self,
        context_len: int = 120,
        max_horizon: int = 30,
        samples_per_epoch: int = 200_000,
        seed: int | None = None,
    ):
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

        # Only use GARCH + Momentum (non-Markovian, have conditional mean signal)
        self.sde_types = ['garch', 'momentum']
        self.sde_weights = [0.5, 0.5]

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        from .sde import sample_params, simulate_daily_returns

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = self.samples_per_epoch // worker_info.num_workers
            worker_seed = (self.seed or 0) + worker_info.id * 10000
        else:
            per_worker = self.samples_per_epoch
            worker_seed = self.seed or 0

        rng = np.random.default_rng(worker_seed)
        total_days = self.context_len + self.max_horizon

        for _ in range(per_worker):
            sde_type = rng.choice(self.sde_types, p=self.sde_weights)
            params = sample_params(sde_type, rng=rng)

            # Generate full path: context + forward
            returns = simulate_daily_returns(sde_type, params, total_days)

            ctx = returns[:self.context_len]      # (context_len,)
            fwd = returns[self.context_len:]       # (max_horizon,)
            y_curve = np.cumsum(fwd).astype(np.float32)  # cumulative returns

            # Build 6-channel features from synthetic returns.
            # Only ch0 (returns) and ch4 (trailing vol) are accurate from SDE.
            # Ch1-3 are set to zero (like forex with no OHLCV detail).
            # Ch5 (momentum) is computable from returns.
            # This avoids the model learning artifacts from fake OHLCV.
            x = np.zeros((self.context_len, 6), dtype=np.float32)
            x[:, 0] = ctx                                    # log returns (exact)
            x[:, 1] = np.abs(ctx)                             # rough intraday range proxy
            x[:, 2] = 0.0                                     # no candle body info
            x[:, 3] = 0.0                                     # no volume
            # Ch4: trailing realized vol (30d)
            import pandas as pd
            ret_series = pd.Series(ctx)
            x[:, 4] = ret_series.rolling(30, min_periods=2).std().fillna(0).values * np.sqrt(252)
            # Ch5: trailing momentum (10d)
            x[:, 5] = ret_series.rolling(10, min_periods=1).sum().values

            rv = float(np.std(ctx) * np.sqrt(365))

            yield (
                torch.from_numpy(x),
                torch.from_numpy(y_curve),
                torch.tensor(0, dtype=torch.long),   # asset_type = 0 (synthetic)
                torch.tensor(rv, dtype=torch.float32),
            )
