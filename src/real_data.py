"""
PyTorch Dataset for real multi-asset data (Phantom v3).

Loads pre-processed .npz files created by scripts/data/build_dataset.py.
Each sample yields (features, horizon, target, asset_type, realized_vol).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RealAssetDataset(Dataset):
    """Multi-asset dataset from real OHLCV data.

    Each sample is a 5-tuple:
      - x:           (context_len, 6) float32 — 6-channel OHLCV features
      - h:           int64 — horizon (3, 5, or 7)
      - target:      float32 — realized forward log-return (scalar)
      - asset_type:  int64 — 0=crypto, 1=equity, 2=forex, 3=commodity
      - realized_vol: float32 — trailing realized vol at context end

    This 5-tuple mirrors the synthetic OnlineDataset's
    (x, h, branches, sde_idx, realized_vol) for training loop compatibility.
    """

    def __init__(self, data_path: str):
        d = np.load(data_path, allow_pickle=True)
        self.X = d['X']                                    # (N, context_len, 6)
        self.H = d['H'].astype(np.int64)                   # (N,)
        self.Y = d['Y'].astype(np.float32)                  # (N,)
        self.asset_type = d['asset_type'].astype(np.int64)  # (N,)
        self.realized_vol = d['realized_vol'].astype(np.float32)  # (N,)

        print(f"Loaded {len(self)} samples from {data_path}")
        print(f"  X shape: {self.X.shape}, asset types: {np.bincount(self.asset_type)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].astype(np.float32)),
            torch.tensor(self.H[idx], dtype=torch.long),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            torch.tensor(self.asset_type[idx], dtype=torch.long),
            torch.tensor(self.realized_vol[idx], dtype=torch.float32),
        )


class RealAssetDatasetV5(Dataset):
    """Multi-asset dataset with relative return targets for Phantom v5.

    Each sample is a 4-tuple:
      - x:           (context_len, 6) float32 — 6-channel OHLCV features
      - y_curve:     (max_horizon,) float32 — RELATIVE cumulative returns
      - asset_type:  int64 — 0=crypto, 1=equity, 2=forex, 3=commodity
      - realized_vol: float32 — trailing realized vol at context end
    """

    def __init__(self, data_path: str, target_key: str = 'Y_relative'):
        d = np.load(data_path, allow_pickle=True)
        self.X = d['X']
        self.Y_relative = d['Y_relative'].astype(np.float32) if 'Y_relative' in d else d['Y'].astype(np.float32)
        self.Y_absolute = d['Y'].astype(np.float32)
        self.asset_type = d['asset_type'].astype(np.int64)
        self.realized_vol = d['realized_vol'].astype(np.float32)
        self.dates_end = d['dates_end'] if 'dates_end' in d else None
        self.asset_id = d['asset_id'] if 'asset_id' in d else None
        self.target_key = target_key

        y = self.Y_relative if target_key == 'Y_relative' else self.Y_absolute
        print(f"Loaded {len(self)} v5 samples from {data_path} (target={target_key})")
        print(f"  X shape: {self.X.shape}, Y shape: {y.shape}")
        print(f"  Y stats: mean={y.mean():.6f}, std={y.std():.4f}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = self.Y_relative[idx] if self.target_key == 'Y_relative' else self.Y_absolute[idx]
        return (
            torch.from_numpy(self.X[idx].astype(np.float32)),
            torch.from_numpy(y),
            torch.tensor(self.asset_type[idx], dtype=torch.long),
            torch.tensor(self.realized_vol[idx], dtype=torch.float32),
        )


class RealAssetDatasetV4(Dataset):
    """Multi-asset dataset with curve targets for Phantom v4.

    Each sample is a 4-tuple:
      - x:           (context_len, 6) float32 — 6-channel OHLCV features
      - y_curve:     (max_horizon,) float32 — cumulative returns at horizons 1..30
      - asset_type:  int64 — 0=crypto, 1=equity, 2=forex, 3=commodity
      - realized_vol: float32 — trailing realized vol at context end
    """

    def __init__(self, data_path: str):
        d = np.load(data_path, allow_pickle=True)
        self.X = d['X']                                    # (N, context_len, 6)
        self.Y = d['Y'].astype(np.float32)                  # (N, 30)
        self.asset_type = d['asset_type'].astype(np.int64)
        self.realized_vol = d['realized_vol'].astype(np.float32)

        print(f"Loaded {len(self)} v4 samples from {data_path}")
        print(f"  X shape: {self.X.shape}, Y shape: {self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].astype(np.float32)),
            torch.from_numpy(self.Y[idx]),                   # (30,)
            torch.tensor(self.asset_type[idx], dtype=torch.long),
            torch.tensor(self.realized_vol[idx], dtype=torch.float32),
        )
