"""
Phantom: Transformer encoder for synthetic-pretrained BTC distributional prediction.

Architecture informed by PatchTST, TimesFM, JointFM-0.1, and Chronos:
  - Encoder-only transformer with patched log-return input
  - Reversible Instance Normalization (RevIN)
  - Learnable positional encoding
  - [CLS] token for sequence aggregation
  - Horizon conditioning via learned embedding
  - Mixture of Gaussians forecast head
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
    n_components: int = 5          # K in Mixture of Gaussians
    horizons: list = field(default_factory=lambda: [3, 5, 7])
    head_hidden: int = 256         # hidden dim in the MoG MLP head

    # --- Numerical stability ---
    log_sigma_min: float = -7.0    # floor for log-std (~e^-7 ≈ 0.0009)
    log_sigma_max: float = 0.0     # ceiling for log-std (~e^0 = 1.0)

    @property
    def n_patches(self) -> int:
        return (self.context_len - self.patch_len) // self.patch_stride + 1


# ── Reversible Instance Normalization ───────────────────────────────

class RevIN(nn.Module):
    """Per-sample normalization: zero mean, unit std.

    Applied before patching; reversed after the forecast head to keep
    output distribution parameters in the original log-return scale.

    Reference: Kim et al. (2022), used by PatchTST and TimesFM.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """Normalize and store statistics.

        Args:
            x: (B, L) raw log-returns.

        Returns:
            x_norm: (B, L) normalized.
        """
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - self.mean) / self.std

    def inverse(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Denormalize MoG parameters back to original scale.

        Args:
            mu:    (B, K) mixture means in normalized space.
            sigma: (B, K) mixture stds in normalized space.

        Returns:
            mu, sigma in original log-return scale.
        """
        # mean and std have shape (B, 1) — broadcast over K
        return mu * self.std + self.mean, sigma * self.std


# ── Patch Embedding ─────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Unfold time series into patches and project to d_model."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.proj = nn.Linear(cfg.patch_len, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) normalized log-returns.

        Returns:
            patches: (B, N, d_model) projected patch embeddings.
        """
        # (B, L) → (B, N, P)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        # (B, N, P) → (B, N, d_model)
        return self.proj(patches)


# ── Mixture of Gaussians Head ───────────────────────────────────────

class MoGHead(nn.Module):
    """Predicts parameters of a K-component Mixture of Gaussians.

    Outputs per sample: K mixture weights (π), K means (μ), K stds (σ).
    """

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
        """
        Args:
            z: (B, d_model) pooled representation.

        Returns:
            log_pi: (B, K) log mixture weights (log-softmax).
            mu:     (B, K) component means.
            sigma:  (B, K) component standard deviations (positive).
        """
        out = self.net(z)  # (B, 3K)

        log_pi_raw = out[:, :self.K]
        mu = out[:, self.K:2*self.K]
        log_sigma = out[:, 2*self.K:]

        log_pi = F.log_softmax(log_pi_raw, dim=-1)
        log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()

        return log_pi, mu, sigma


# ── Full Model ──────────────────────────────────────────────────────

class PhantomModel(nn.Module):
    """Encoder-only transformer for distributional BTC prediction.

    Pipeline:
        log-returns → RevIN → Patch+Project → [CLS] + Pos Enc
        → Transformer Encoder → CLS output + Horizon Embedding
        → MoG Head → (π, μ, σ)
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.cfg = cfg

        # Input processing
        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(cfg)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Learnable positional encoding: N patches + 1 CLS
        n_positions = cfg.n_patches + 1
        self.pos_enc = nn.Parameter(torch.empty(1, n_positions, cfg.d_model))
        nn.init.uniform_(self.pos_enc, -0.02, 0.02)

        # Horizon conditioning: one embedding per horizon value
        self.horizon_embed = nn.Embedding(max(cfg.horizons) + 1, cfg.d_model)

        # Stat injection: project RevIN-removed (mean, std) back as features
        # so the MoG head can condition on input level/scale
        self.stat_proj = nn.Linear(2, cfg.d_model)

        # Transformer encoder (Pre-LN via norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            norm=nn.LayerNorm(cfg.d_model),  # final norm after last layer
        )

        # Forecast head
        self.head = MoGHead(cfg)

        # Input dropout
        self.input_dropout = nn.Dropout(cfg.dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        horizon: torch.Tensor,
    ):
        """
        Args:
            x:       (B, L) daily log-returns context window.
            horizon: (B,)   forecast horizon in days (3, 5, or 7).

        Returns:
            log_pi: (B, K) log mixture weights.
            mu:     (B, K) mixture means (original log-return scale).
            sigma:  (B, K) mixture stds (original log-return scale).
        """
        B = x.size(0)

        # 1. Instance normalization
        x_norm = self.revin(x)

        # 2. Patch embedding: (B, L) → (B, N, d_model)
        patches = self.patch_embed(x_norm)

        # 3. Prepend [CLS] token: (B, N+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)

        # 4. Add positional encoding
        tokens = tokens + self.pos_enc
        tokens = self.input_dropout(tokens)

        # 5. Transformer encoder
        z = self.encoder(tokens)   # (B, N+1, d_model)

        # 6. Extract [CLS] representation
        cls_out = z[:, 0, :]       # (B, d_model)

        # 7. Add horizon conditioning
        h_emb = self.horizon_embed(horizon)  # (B, d_model)

        # 8. Inject RevIN statistics so MoG head can condition on input level/scale
        stats = torch.cat([self.revin.mean, self.revin.std], dim=-1)  # (B, 2)
        stat_emb = self.stat_proj(stats)  # (B, d_model)

        cls_out = cls_out + h_emb + stat_emb

        # 9. MoG head → distribution parameters (normalized space)
        log_pi, mu_norm, sigma_norm = self.head(cls_out)

        # 10. Denormalize to original log-return scale
        mu, sigma = self.revin.inverse(mu_norm, sigma_norm)

        return log_pi, mu, sigma

    def predict_distribution(
        self,
        x: torch.Tensor,
        horizon: torch.Tensor,
        n_samples: int = 10000,
    ) -> torch.Tensor:
        """Sample from the predicted MoG distribution.

        Args:
            x:         (B, L) context log-returns.
            horizon:   (B,) horizon in days.
            n_samples: number of Monte Carlo samples.

        Returns:
            samples: (B, n_samples) draws from the predicted distribution.
        """
        with torch.no_grad():
            log_pi, mu, sigma = self.forward(x, horizon)

        # Sample component indices: (B, n_samples)
        component = torch.distributions.Categorical(logits=log_pi).sample((n_samples,))
        component = component.T  # (B, n_samples)

        # Gather means and stds for sampled components
        mu_sampled = mu.gather(1, component)          # (B, n_samples)
        sigma_sampled = sigma.gather(1, component)    # (B, n_samples)

        # Sample from the selected Gaussians
        eps = torch.randn_like(mu_sampled)
        return mu_sampled + sigma_sampled * eps

    def get_cdf(
        self,
        x: torch.Tensor,
        horizon: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the predicted CDF at given points.

        Args:
            x:      (B, L) context.
            horizon: (B,) horizon.
            points: (B, P) or (P,) points at which to evaluate.

        Returns:
            cdf: (B, P) CDF values.
        """
        with torch.no_grad():
            log_pi, mu, sigma = self.forward(x, horizon)

        pi = log_pi.exp()  # (B, K)

        if points.dim() == 1:
            points = points.unsqueeze(0).expand(mu.size(0), -1)

        # (B, K, 1) and (B, 1, P) → (B, K, P)
        z = (points.unsqueeze(1) - mu.unsqueeze(2)) / sigma.unsqueeze(2)
        component_cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))

        # Mixture CDF: sum over components
        cdf = (pi.unsqueeze(2) * component_cdf).sum(dim=1)  # (B, P)
        return cdf
