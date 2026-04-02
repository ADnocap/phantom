"""
Phantom: Transformer encoder for synthetic-pretrained BTC distributional prediction.

Architecture:
  - Encoder-only transformer with patched log-return input
  - NO normalization — raw log-returns preserve conditional signal
  - Cross-attention decoder (horizon query attends to all encoder patches)
  - Condition dropout for classifier-free guidance
  - Auxiliary heads (SDE type classification + volatility regression)
  - Mixture of Gaussians forecast head
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ───────────────────────────────────────────────────

@dataclass
class PhantomConfig:
    context_len: int = 60
    patch_len: int = 5
    patch_stride: int = 5

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.2
    activation: str = "gelu"

    n_components: int = 5
    horizons: list = field(default_factory=lambda: [3, 5, 7])
    head_hidden: int = 256

    log_sigma_min: float = -7.0
    log_sigma_max: float = 2.0

    # Cross-attention decoder
    n_decoder_layers: int = 2

    # Auxiliary tasks
    n_sde_types: int = 5           # GBM, Merton, Kou, Bates, Regime-Switching

    # Condition dropout (classifier-free guidance)
    cond_drop_prob: float = 0.15   # probability of zeroing encoder output during training

    @property
    def n_patches(self) -> int:
        return (self.context_len - self.patch_len) // self.patch_stride + 1


# ── Patch Embedding ─────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.proj = nn.Linear(cfg.patch_len, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return self.proj(patches)


# ── Cross-Attention Decoder Layer ───────────────────────────────────

class CrossAttentionDecoderLayer(nn.Module):
    """Single decoder layer: self-attn on query, then cross-attn to encoder output."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, query: torch.Tensor, encoder_out: torch.Tensor):
        # Cross-attention: query attends to encoder patches (Pre-LN)
        q_norm = self.norm1(query)
        attn_out, _ = self.cross_attn(q_norm, encoder_out, encoder_out)
        query = query + self.dropout1(attn_out)

        # Feedforward (Pre-LN)
        ff_out = self.ff(self.norm2(query))
        query = query + self.dropout2(ff_out)

        return query


# ── Mixture of Gaussians Head ───────────────────────────────────────

class MoGHead(nn.Module):
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
    """Transformer encoder + cross-attention decoder for distributional prediction.

    Key design choices to prevent "marginal prediction" collapse:
    1. Cross-attention decoder: horizon query attends to ALL encoder patches,
       making input-independent predictions architecturally impossible.
    2. Condition dropout: randomly zeros encoder output during training,
       enabling classifier-free guidance at inference.
    3. Auxiliary heads: SDE type classifier + volatility regressor force
       the encoder to extract input-dependent features.
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.patch_embed = PatchEmbedding(cfg)
        n_positions = cfg.n_patches
        self.pos_enc = nn.Parameter(torch.empty(1, n_positions, cfg.d_model))
        nn.init.uniform_(self.pos_enc, -0.02, 0.02)
        self.input_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
            activation=cfg.activation, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        # Cross-attention decoder (replaces CLS + additive horizon)
        self.horizon_embed = nn.Embedding(max(cfg.horizons) + 1, cfg.d_model)
        self.decoder_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.d_model)

        # MoG forecast head
        self.head = MoGHead(cfg)

        # Auxiliary heads (Fix #1: force encoder to extract features)
        self.aux_pool = nn.AdaptiveAvgPool1d(1)  # pool encoder output for aux tasks
        self.sde_classifier = nn.Linear(cfg.d_model, cfg.n_sde_types)
        self.vol_regressor = nn.Linear(cfg.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, x: torch.Tensor):
        """Encode context into patch representations.

        Args:
            x: (B, L) raw daily log-returns.
        Returns:
            enc_out: (B, N, d_model) patch representations.
        """
        patches = self.patch_embed(x)            # (B, N, d_model)
        patches = patches + self.pos_enc
        patches = self.input_dropout(patches)
        return self.encoder(patches)              # (B, N, d_model)

    def decode(self, enc_out: torch.Tensor, horizon: torch.Tensor):
        """Decode distribution parameters via cross-attention.

        Args:
            enc_out: (B, N, d_model) encoder patch representations.
            horizon: (B,) forecast horizon in days.
        Returns:
            log_pi, mu, sigma: MoG parameters.
        """
        # Horizon query token
        query = self.horizon_embed(horizon).unsqueeze(1)  # (B, 1, d_model)

        # Cross-attention: query attends to encoder patches
        for layer in self.decoder_layers:
            query = layer(query, enc_out)

        query = self.decoder_norm(query)
        decoded = query.squeeze(1)  # (B, d_model)

        return self.head(decoded)

    def forward(self, x: torch.Tensor, horizon: torch.Tensor):
        """
        Returns:
            log_pi: (B, K) log mixture weights.
            mu:     (B, K) component means.
            sigma:  (B, K) component stds.
        """
        enc_out = self.encode(x)  # (B, N, d_model)

        # Fix #2: Condition dropout — zero encoder output with probability p
        if self.training and self.cfg.cond_drop_prob > 0:
            mask = torch.rand(x.size(0), 1, 1, device=x.device) > self.cfg.cond_drop_prob
            enc_out = enc_out * mask  # (B, N, d_model) with some samples zeroed

        log_pi, mu, sigma = self.decode(enc_out, horizon)
        return log_pi, mu, sigma

    def forward_auxiliary(self, x: torch.Tensor):
        """Compute auxiliary predictions (SDE type + volatility).

        Args:
            x: (B, L) raw daily log-returns.
        Returns:
            sde_logits: (B, n_sde_types) classification logits.
            vol_pred:   (B,) predicted realized volatility.
        """
        enc_out = self.encode(x)  # (B, N, d_model)

        # Pool encoder output: (B, N, d) → (B, d)
        pooled = self.aux_pool(enc_out.transpose(1, 2)).squeeze(-1)

        sde_logits = self.sde_classifier(pooled)
        vol_pred = self.vol_regressor(pooled).squeeze(-1)

        return sde_logits, vol_pred

    def forward_cfg(self, x: torch.Tensor, horizon: torch.Tensor,
                    guidance_scale: float = 2.0):
        """Classifier-free guidance inference.

        Computes both conditional and unconditional predictions,
        then extrapolates for sharper conditional output.
        """
        enc_out = self.encode(x)

        # Conditional prediction
        log_pi_c, mu_c, sigma_c = self.decode(enc_out, horizon)

        # Unconditional prediction (zeroed encoder)
        zeros = torch.zeros_like(enc_out)
        log_pi_u, mu_u, sigma_u = self.decode(zeros, horizon)

        # Guided output: extrapolate away from unconditional
        mu = mu_u + guidance_scale * (mu_c - mu_u)
        # For sigma: interpolate in log space
        log_sigma_c = sigma_c.log()
        log_sigma_u = sigma_u.log()
        log_sigma = log_sigma_u + guidance_scale * (log_sigma_c - log_sigma_u)
        sigma = log_sigma.clamp(self.cfg.log_sigma_min, self.cfg.log_sigma_max).exp()
        # Weights: use conditional
        log_pi = log_pi_c

        return log_pi, mu, sigma
