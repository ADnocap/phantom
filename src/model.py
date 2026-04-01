"""
Phantom: Transformer encoder for synthetic-pretrained BTC distributional prediction.

Architecture informed by PatchTST, TimesFM, JointFM-0.1, and Chronos:
  - Encoder-only transformer with patched log-return input
  - Mean absolute scaling (Chronos-style) — preserves shape, removes only magnitude
  - Learnable positional encoding
  - [CLS] token for sequence aggregation
  - Horizon conditioning via learned embedding
  - Two head options: Mixture of Gaussians (MoG) or Quantile regression
"""

from dataclasses import dataclass, field
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ───────────────────────────────────────────────────

@dataclass
class PhantomConfig:
    """All model hyperparameters in one place."""

    # --- Input ---
    context_len: int = 60          # days of history the model sees
    patch_len: int = 5             # days per patch (~1 trading week)
    patch_stride: int = 5          # stride between patches (= patch_len for non-overlapping)

    # --- Transformer ---
    d_model: int = 256             # embedding / hidden dimension
    n_heads: int = 8               # attention heads
    n_layers: int = 6              # encoder layers
    d_ff: int = 1024               # feedforward inner dimension
    dropout: float = 0.2           # dropout rate
    activation: str = "gelu"       # feedforward activation

    # --- Forecast head ---
    head_type: str = "mog"         # "mog" or "quantile"
    n_components: int = 5          # K in Mixture of Gaussians (mog only)
    quantiles: list = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    horizons: list = field(default_factory=lambda: [3, 5, 7])
    head_hidden: int = 256         # hidden dim in the MLP head

    # --- Numerical stability (mog only) ---
    log_sigma_min: float = -7.0    # floor for log-std (~e^-7 ≈ 0.0009)
    log_sigma_max: float = 0.0     # ceiling for log-std (~e^0 = 1.0)

    @property
    def n_patches(self) -> int:
        return (self.context_len - self.patch_len) // self.patch_stride + 1


# ── Mean Absolute Scaling (Chronos-style) ─────────────────────────

class MeanAbsScaling(nn.Module):
    """Per-sample scaling by mean absolute value.

    Unlike RevIN, this preserves sign, trends, and distributional shape.
    Only removes magnitude, so the transformer sees the actual pattern
    of returns — not a standardized version.

    Reference: Chronos (Amazon, 2024) uses this for probabilistic forecasting.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """Scale input by mean absolute value.

        Args:
            x: (B, L) raw log-returns.
        Returns:
            x_scaled: (B, L) scaled.
        """
        self.scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=self.eps)
        return x / self.scale

    def inverse_mog(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Denormalize MoG parameters back to original scale."""
        return mu * self.scale, sigma * self.scale

    def inverse_quantiles(self, q: torch.Tensor):
        """Denormalize quantile predictions back to original scale."""
        return q * self.scale


# ── Patch Embedding ─────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Unfold time series into patches and project to d_model."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.proj = nn.Linear(cfg.patch_len, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return self.proj(patches)


# ── Mixture of Gaussians Head ───────────────────────────────────────

class MoGHead(nn.Module):
    """Predicts parameters of a K-component Mixture of Gaussians."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.K = cfg.n_components
        self.log_sigma_min = cfg.log_sigma_min
        self.log_sigma_max = cfg.log_sigma_max

        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 3 * cfg.n_components),
        )

    def forward(self, z: torch.Tensor):
        out = self.net(z)
        log_pi_raw = out[:, :self.K]
        mu = out[:, self.K:2*self.K]
        log_sigma = out[:, 2*self.K:]

        log_pi = F.log_softmax(log_pi_raw, dim=-1)
        log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()

        return log_pi, mu, sigma


# ── Quantile Head ──────────────────────────────────────────────────

class QuantileHead(nn.Module):
    """Predicts quantiles at fixed levels via pinball loss."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.n_quantiles = len(cfg.quantiles)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, self.n_quantiles),
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)  # (B, n_quantiles)


# ── Full Model ──────────────────────────────────────────────────────

class PhantomModel(nn.Module):
    """Encoder-only transformer for distributional BTC prediction.

    Pipeline:
        log-returns → MeanAbsScaling → Patch+Project → [CLS] + Pos Enc
        → Transformer Encoder → CLS output + Horizon Embedding
        → Head (MoG or Quantile) → output
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.cfg = cfg

        # Input processing — mean absolute scaling (NOT RevIN)
        self.scaler = MeanAbsScaling()
        self.patch_embed = PatchEmbedding(cfg)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Learnable positional encoding: N patches + 1 CLS
        n_positions = cfg.n_patches + 1
        self.pos_enc = nn.Parameter(torch.empty(1, n_positions, cfg.d_model))
        nn.init.uniform_(self.pos_enc, -0.02, 0.02)

        # Horizon conditioning
        self.horizon_embed = nn.Embedding(max(cfg.horizons) + 1, cfg.d_model)

        # Transformer encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        # Forecast head
        if cfg.head_type == "quantile":
            self.head = QuantileHead(cfg)
        else:
            self.head = MoGHead(cfg)

        # Input dropout
        self.input_dropout = nn.Dropout(cfg.dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, horizon: torch.Tensor):
        """
        Args:
            x:       (B, L) daily log-returns context window.
            horizon: (B,)   forecast horizon in days (3, 5, or 7).

        Returns:
            MoG mode:     (log_pi, mu, sigma) in original scale.
            Quantile mode: quantile_preds (B, Q) in original scale.
        """
        B = x.size(0)

        # 1. Scale (preserves shape — NOT RevIN)
        x_scaled = self.scaler(x)

        # 2. Patch embedding
        patches = self.patch_embed(x_scaled)

        # 3. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)

        # 4. Positional encoding + dropout
        tokens = tokens + self.pos_enc
        tokens = self.input_dropout(tokens)

        # 5. Transformer encoder
        z = self.encoder(tokens)

        # 6. Extract [CLS]
        cls_out = z[:, 0, :]

        # 7. Horizon conditioning
        h_emb = self.horizon_embed(horizon)
        cls_out = cls_out + h_emb

        # 8. Head + denormalize
        if self.cfg.head_type == "quantile":
            q_norm = self.head(cls_out)
            return self.scaler.inverse_quantiles(q_norm)
        else:
            log_pi, mu_norm, sigma_norm = self.head(cls_out)
            mu, sigma = self.scaler.inverse_mog(mu_norm, sigma_norm)
            return log_pi, mu, sigma
