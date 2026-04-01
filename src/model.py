"""
Phantom: Transformer encoder for synthetic-pretrained BTC distributional prediction.

Architecture informed by JointFM-0.1, PatchTST, and Chronos:
  - Encoder-only transformer with patched log-return input
  - NO normalization — raw log-returns preserve conditional signal (JointFM-style)
  - Learnable positional encoding
  - [CLS] token for sequence aggregation
  - Horizon conditioning via learned embedding
  - Mixture of Gaussians forecast head
"""

from dataclasses import dataclass, field

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
    log_sigma_max: float = 2.0     # ceiling for log-std (~e^2 ≈ 7.4)

    @property
    def n_patches(self) -> int:
        return (self.context_len - self.patch_len) // self.patch_stride + 1


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


# ── Full Model ──────────────────────────────────────────────────────

class PhantomModel(nn.Module):
    """Encoder-only transformer for distributional BTC prediction.

    Pipeline:
        raw log-returns → Patch+Project → [CLS] + Pos Enc
        → Transformer Encoder → CLS output + Horizon Embedding
        → MoG Head → (π, μ, σ)

    No normalization is applied — the model sees raw return patterns
    and must learn to produce scale-appropriate distribution parameters.
    This follows JointFM-0.1 which uses no normalization.
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.cfg = cfg

        # Input processing — no normalization (JointFM-style)
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
            x:       (B, L) daily log-returns context window (raw, unnormalized).
            horizon: (B,)   forecast horizon in days (3, 5, or 7).

        Returns:
            log_pi: (B, K) log mixture weights.
            mu:     (B, K) mixture means (raw log-return scale).
            sigma:  (B, K) mixture stds (raw log-return scale).
        """
        B = x.size(0)

        # 1. Patch embedding (no normalization — raw returns)
        patches = self.patch_embed(x)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)

        # 3. Positional encoding + dropout
        tokens = tokens + self.pos_enc
        tokens = self.input_dropout(tokens)

        # 4. Transformer encoder
        z = self.encoder(tokens)

        # 5. Extract [CLS]
        cls_out = z[:, 0, :]

        # 6. Horizon conditioning
        h_emb = self.horizon_embed(horizon)
        cls_out = cls_out + h_emb

        # 7. MoG head → distribution parameters (raw scale)
        log_pi, mu, sigma = self.head(cls_out)

        return log_pi, mu, sigma
