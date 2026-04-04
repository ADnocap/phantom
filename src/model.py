"""
Phantom: Transformer encoder for synthetic-pretrained BTC distributional prediction.

Architecture:
  - Encoder-only transformer with patched log-return input
  - NO normalization — raw log-returns preserve conditional signal
  - Cross-attention decoder (horizon query attends to all encoder patches)
  - Condition dropout for classifier-free guidance
  - Auxiliary heads (SDE type classification + volatility regression)
  - Mixture forecast head (Gaussian or Student-t components)

v2 additions:
  - Student-t mixture head (use_student_t)
  - Multi-scale patching (patch_sizes)
  - Multi-channel input features (n_input_channels)
  - In-model series decomposition (use_decomposition)
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
    n_sde_types: int = 5           # GBM, Merton, Kou, Bates, Regime-Switching (v1)

    # Condition dropout (classifier-free guidance)
    cond_drop_prob: float = 0.15   # probability of zeroing encoder output during training

    # ── v2 additions ──

    # Student-t mixture head
    use_student_t: bool = False
    log_nu_min: float = 0.7        # log(2.01) — nu > 2 for finite variance
    log_nu_max: float = 4.6        # log(100)

    # Multi-scale patching (None = single-scale original)
    patch_sizes: list | None = None

    # Multi-channel input (1 = raw returns only, 4 = returns + 3 vol features)
    n_input_channels: int = 1

    # In-model series decomposition (Autoformer-style)
    use_decomposition: bool = False
    decomp_kernel: int = 5

    # FiLM conditioning (multiplicative, cannot be bypassed via residual)
    use_film: bool = False

    # Head type: 'mog' (mixture of Gaussians), 'mot' (mixture of Student-t), 'student_t' (single)
    head_type: str = 'mog'

    # ── v3 additions (real multi-asset pretraining) ──
    n_asset_types: int = 4             # crypto=0, equity=1, forex=2, commodity=3
    use_asset_classifier: bool = False  # Asset-type auxiliary classifier
    use_sign_classifier: bool = False   # Return-sign auxiliary classifier

    @property
    def n_patches(self) -> int:
        if self.patch_sizes is not None:
            return sum((self.context_len - ps) // ps + 1 for ps in self.patch_sizes)
        return (self.context_len - self.patch_len) // self.patch_stride + 1


# ── Patch Embedding ─────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Single-scale patch embedding with optional multi-channel support."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.n_channels = cfg.n_input_channels
        in_dim = cfg.patch_len * cfg.n_input_channels
        self.proj = nn.Linear(in_dim, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (B, L, C) multi-channel input
            p = x.unfold(1, self.patch_len, self.patch_stride)  # (B, N, C, ps)
            B, N, C, ps = p.shape
            p = p.reshape(B, N, C * ps)  # (B, N, C*ps)
        else:
            # (B, L) single-channel (backward compat)
            p = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return self.proj(p)


class MultiScalePatchEmbedding(nn.Module):
    """Multi-resolution patch embedding with per-scale learned embeddings."""

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.patch_sizes = cfg.patch_sizes
        self.n_channels = cfg.n_input_channels
        self.projections = nn.ModuleList()
        self.scale_embeddings = nn.ParameterList()
        self.patches_per_scale = []

        for ps in cfg.patch_sizes:
            n_patches = (cfg.context_len - ps) // ps + 1
            self.patches_per_scale.append(n_patches)
            in_dim = ps * cfg.n_input_channels
            self.projections.append(nn.Linear(in_dim, cfg.d_model))
            self.scale_embeddings.append(
                nn.Parameter(torch.empty(1, n_patches, cfg.d_model)))

        self._init_embeddings()

    def _init_embeddings(self):
        for emb in self.scale_embeddings:
            nn.init.uniform_(emb, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_patches = []
        for ps, proj, scale_emb in zip(
                self.patch_sizes, self.projections, self.scale_embeddings):
            if x.dim() == 3:
                # (B, L, C) multi-channel
                p = x.unfold(1, ps, ps)        # (B, N, C, ps)
                B, N, C, _ps = p.shape
                p = p.reshape(B, N, C * _ps)   # (B, N, C*ps)
            else:
                # (B, L) single-channel
                p = x.unfold(-1, ps, ps)       # (B, N, ps)
            p = proj(p) + scale_emb            # (B, N, d_model)
            all_patches.append(p)
        return torch.cat(all_patches, dim=1)   # (B, total_patches, d_model)


# ── Series Decomposition ──────────────────────────────────────────

class SeriesDecomposition(nn.Module):
    """Moving-average trend-residual decomposition (Autoformer-style)."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.avg = nn.AvgPool1d(
            kernel_size, stride=1, padding=kernel_size // 2,
            count_include_pad=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, d) sequence of patch representations.
        Returns:
            residual: (B, N, d) high-frequency component.
            trend:    (B, N, d) low-frequency component.
        """
        # AvgPool1d expects (B, C, L), so transpose
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


# ── Cross-Attention Decoder Layer ───────────────────────────────────

class CrossAttentionDecoderLayer(nn.Module):
    """Single decoder layer: cross-attn to encoder output + FFN."""

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


# ── FiLM Decoder Layer ─────────────────────────────────────────────

class FiLMDecoderLayer(nn.Module):
    """FiLM-conditioned decoder: multiplicative conditioning that cannot be bypassed.

    Instead of cross-attention (additive residual, easy to ignore),
    FiLM modulates features via: output = gamma(condition) * input + beta(condition).
    If gamma=0, the feature is killed — the model MUST use the condition.
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        # Pool encoder output to a conditioning vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        # FiLM: encoder → (gamma, beta) for modulating the query
        self.film = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, 2 * cfg.d_model),
        )

        # Also keep cross-attention for information routing
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
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, query: torch.Tensor, encoder_out: torch.Tensor):
        # FiLM conditioning: modulate query multiplicatively
        enc_pooled = self.pool(encoder_out.transpose(1, 2)).squeeze(-1)  # (B, d)
        film_params = self.film(enc_pooled)  # (B, 2*d)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, d)
        gamma = gamma.unsqueeze(1)  # (B, 1, d)
        beta = beta.unsqueeze(1)    # (B, 1, d)

        # Apply FiLM: multiplicative modulation (cannot be bypassed)
        query = self.norm1(query)
        query = gamma * query + beta

        # Cross-attention (additional, but FiLM already injected the signal)
        q_norm = self.norm2(query)
        attn_out, _ = self.cross_attn(q_norm, encoder_out, encoder_out)
        query = query + self.dropout1(attn_out)

        # FFN
        ff_out = self.ff(self.norm3(query))
        query = query + self.dropout2(ff_out)

        return query


# ── Mixture Forecast Head ──────────────────────────────────────────

class MixtureHead(nn.Module):
    """Mixture distribution head supporting Gaussian and Student-t components.

    When use_student_t=False: outputs (log_pi, mu, sigma, None)  — Gaussian
    When use_student_t=True:  outputs (log_pi, mu, sigma, nu)    — Student-t
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.K = cfg.n_components
        self.use_student_t = cfg.use_student_t
        self.log_sigma_min = cfg.log_sigma_min
        self.log_sigma_max = cfg.log_sigma_max
        self.log_nu_min = cfg.log_nu_min
        self.log_nu_max = cfg.log_nu_max

        out_dim = 4 * cfg.n_components if cfg.use_student_t else 3 * cfg.n_components
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, out_dim),
        )

    def forward(self, z: torch.Tensor):
        out = self.net(z)
        log_pi_raw = out[:, :self.K]
        mu = out[:, self.K:2*self.K]
        log_sigma = out[:, 2*self.K:3*self.K]

        log_pi = F.log_softmax(log_pi_raw, dim=-1)
        log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()

        if self.use_student_t:
            log_nu = out[:, 3*self.K:]
            log_nu = log_nu.clamp(self.log_nu_min, self.log_nu_max)
            nu = log_nu.exp()
            return log_pi, mu, sigma, nu

        return log_pi, mu, sigma, None


# Backward-compatible alias
MoGHead = MixtureHead


# ── Single Student-t Head ─────────────────────────────────────────

class StudentTHead(nn.Module):
    """Single Student-t distribution head: 3 parameters (mu, sigma, nu).

    Outputs a single Student-t distribution per sample — no mixture.
    This is what Lag-Llama and GluonTS/DeepAR use.

    Returns (log_pi, mu, sigma, nu) where log_pi = zeros (single component)
    for API compatibility with MixtureHead.
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.log_sigma_min = cfg.log_sigma_min
        self.log_sigma_max = cfg.log_sigma_max
        self.log_nu_min = cfg.log_nu_min
        self.log_nu_max = cfg.log_nu_max

        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 3),  # mu, log_sigma, log_nu
        )

    def forward(self, z: torch.Tensor):
        out = self.net(z)  # (B, 3)
        mu = out[:, 0:1]   # (B, 1)
        log_sigma = out[:, 1:2].clamp(self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()  # (B, 1)
        log_nu = out[:, 2:3].clamp(self.log_nu_min, self.log_nu_max)
        nu = log_nu.exp()        # (B, 1)

        # log_pi = 0 (single component with weight 1)
        log_pi = torch.zeros_like(mu)  # (B, 1)

        return log_pi, mu, sigma, nu


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

    v2 additions (all backward-compatible, toggled via config):
    4. Student-t mixture head for heavy-tailed distributions.
    5. Multi-scale patching for multi-resolution temporal patterns.
    6. Multi-channel input (trailing realized vol features).
    7. In-model series decomposition for trend/residual separation.
    """

    def __init__(self, cfg: PhantomConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder input
        if cfg.patch_sizes is not None:
            self.patch_embed = MultiScalePatchEmbedding(cfg)
            n_positions = cfg.n_patches
            # Scale embeddings are inside MultiScalePatchEmbedding;
            # pos_enc is additive (zero-init so it has no effect initially)
            self.pos_enc = nn.Parameter(torch.zeros(1, n_positions, cfg.d_model))
        else:
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

        # Series decomposition blocks (v2)
        if cfg.use_decomposition:
            self.decomp_blocks = nn.ModuleList([
                SeriesDecomposition(cfg.decomp_kernel)
                for _ in range(cfg.n_layers)
            ])

        # Decoder (cross-attention or FiLM-conditioned)
        self.horizon_embed = nn.Embedding(max(cfg.horizons) + 1, cfg.d_model)
        if cfg.use_film:
            self.decoder_layers = nn.ModuleList([
                FiLMDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                CrossAttentionDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)
            ])
        self.decoder_norm = nn.LayerNorm(cfg.d_model)

        # Forecast head
        if cfg.head_type == 'student_t':
            self.head = StudentTHead(cfg)
        else:
            # 'mog' or 'mot' — handled by MixtureHead via use_student_t flag
            self.head = MixtureHead(cfg)

        # Auxiliary heads
        self.aux_pool = nn.AdaptiveAvgPool1d(1)
        self.sde_classifier = nn.Linear(cfg.d_model, cfg.n_sde_types)
        self.vol_regressor = nn.Linear(cfg.d_model, 1)

        # v3 auxiliary heads (real multi-asset pretraining)
        if cfg.use_asset_classifier:
            self.asset_classifier = nn.Linear(cfg.d_model, cfg.n_asset_types)
        if cfg.use_sign_classifier:
            self.sign_classifier = nn.Linear(cfg.d_model, 2)

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
            x: (B, L) or (B, L, C) raw daily log-returns (optionally multi-channel).
        Returns:
            enc_out: (B, N, d_model) patch representations.
        """
        patches = self.patch_embed(x)            # (B, N, d_model)
        patches = patches + self.pos_enc
        patches = self.input_dropout(patches)

        if self.cfg.use_decomposition:
            # Manual layer iteration with decomposition between layers
            for i, layer in enumerate(self.encoder.layers):
                patches = layer(patches)
                residual, _trend = self.decomp_blocks[i](patches)
                patches = residual
            return self.encoder.norm(patches)
        else:
            return self.encoder(patches)

    def decode(self, enc_out: torch.Tensor, horizon: torch.Tensor):
        """Decode distribution parameters via cross-attention.

        Args:
            enc_out: (B, N, d_model) encoder patch representations.
            horizon: (B,) forecast horizon in days.
        Returns:
            log_pi, mu, sigma: mixture parameters.
            nu: degrees of freedom (None for Gaussian).
        """
        query = self.horizon_embed(horizon).unsqueeze(1)  # (B, 1, d_model)

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
            sigma:  (B, K) component stds/scales.
            nu:     (B, K) degrees of freedom, or None for Gaussian.
        """
        enc_out = self.encode(x)  # (B, N, d_model)

        # Condition dropout — zero encoder output with probability p
        if self.training and self.cfg.cond_drop_prob > 0:
            mask = torch.rand(x.size(0), 1, 1, device=enc_out.device) > self.cfg.cond_drop_prob
            enc_out = enc_out * mask

        return self.decode(enc_out, horizon)

    def forward_auxiliary(self, x: torch.Tensor):
        """Compute auxiliary predictions (SDE type + volatility + v3 heads).

        Args:
            x: (B, L) or (B, L, C) raw daily log-returns.
        Returns:
            sde_logits:   (B, n_sde_types) classification logits.
            vol_pred:     (B,) predicted realized volatility.
            asset_logits: (B, n_asset_types) or None — asset-type classifier.
            sign_logits:  (B, 2) or None — return-sign classifier.
        """
        enc_out = self.encode(x)  # (B, N, d_model)

        # Pool encoder output: (B, N, d) → (B, d)
        pooled = self.aux_pool(enc_out.transpose(1, 2)).squeeze(-1)

        sde_logits = self.sde_classifier(pooled)
        vol_pred = self.vol_regressor(pooled).squeeze(-1)

        asset_logits = self.asset_classifier(pooled) if hasattr(self, 'asset_classifier') else None
        sign_logits = self.sign_classifier(pooled) if hasattr(self, 'sign_classifier') else None

        return sde_logits, vol_pred, asset_logits, sign_logits

    def forward_cfg(self, x: torch.Tensor, horizon: torch.Tensor,
                    guidance_scale: float = 2.0):
        """Classifier-free guidance inference.

        Computes both conditional and unconditional predictions,
        then extrapolates for sharper conditional output.
        """
        enc_out = self.encode(x)

        # Conditional prediction
        log_pi_c, mu_c, sigma_c, nu_c = self.decode(enc_out, horizon)

        # Unconditional prediction (zeroed encoder)
        zeros = torch.zeros_like(enc_out)
        log_pi_u, mu_u, sigma_u, nu_u = self.decode(zeros, horizon)

        # Guided output: extrapolate away from unconditional
        mu = mu_u + guidance_scale * (mu_c - mu_u)

        log_sigma_c = sigma_c.log()
        log_sigma_u = sigma_u.log()
        log_sigma = log_sigma_u + guidance_scale * (log_sigma_c - log_sigma_u)
        sigma = log_sigma.clamp(self.cfg.log_sigma_min, self.cfg.log_sigma_max).exp()

        log_pi = log_pi_c  # Use conditional weights

        # Guide nu in log space if Student-t
        nu = None
        if nu_c is not None and nu_u is not None:
            log_nu_c = nu_c.log()
            log_nu_u = nu_u.log()
            log_nu = log_nu_u + guidance_scale * (log_nu_c - log_nu_u)
            nu = log_nu.clamp(self.cfg.log_nu_min, self.cfg.log_nu_max).exp()

        return log_pi, mu, sigma, nu
