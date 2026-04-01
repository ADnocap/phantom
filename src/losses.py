"""
Loss functions for Phantom distributional prediction.

Implements:
  - Energy Distance: primary loss comparing MoG samples vs branched MC ground truth
  - NLL:  Negative log-likelihood (auxiliary, for gradient flow to mixture weights)
  - CRPS: Closed-form CRPS for MoG (validation metric)
  - Combined loss: Energy Distance + nll_weight * NLL

References:
  - Gneiting & Raftery (2007) for energy distance and CRPS
  - Grimit et al. (2006) for closed-form MoG CRPS
  - JointFM-0.1 for the branched-future training paradigm
"""

import math
import torch
import torch.nn.functional as F


LOG_2PI = math.log(2 * math.pi)
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)


# ── Energy Distance Loss ──────────────────────────────────────────

def energy_distance_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y_branches: torch.Tensor,
    n_model_samples: int = 256,
) -> torch.Tensor:
    """Energy distance between predicted MoG and empirical branch distribution.

    ED(P, Q) = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]

    where X ~ predicted MoG (via reparameterization), Y ~ ground-truth branches.

    Args:
        log_pi:    (B, K) log mixture weights.
        mu:        (B, K) component means.
        sigma:     (B, K) component stds.
        y_branches: (B, N) ground-truth branched future returns.
        n_model_samples: M, samples to draw from predicted MoG.

    Returns:
        Scalar mean energy distance over the batch.
    """
    B, K = mu.shape
    N = y_branches.size(1)
    M = n_model_samples

    # ── Sample from MoG via reparameterization ──
    pi = log_pi.exp()  # (B, K)
    comp_idx = torch.multinomial(pi, M, replacement=True)  # (B, M)
    mu_sel = torch.gather(mu, 1, comp_idx)                 # (B, M)
    sigma_sel = torch.gather(sigma, 1, comp_idx)            # (B, M)
    X = mu_sel + sigma_sel * torch.randn_like(mu_sel)       # (B, M)

    # ── Term 1: E[|X - Y|] (cross term) ──
    # (B, M, 1) - (B, 1, N) → (B, M, N)
    cross = torch.abs(X.unsqueeze(2) - y_branches.unsqueeze(1))
    term_xy = cross.mean(dim=(1, 2))  # (B,)

    # ── Term 2: E[|X - X'|] via sorted trick ──
    X_sorted = X.sort(dim=1).values  # (B, M)
    diffs_xx = X_sorted[:, 1:] - X_sorted[:, :-1]  # (B, M-1)
    idx = torch.arange(1, M, device=X.device, dtype=X.dtype)
    weights_xx = idx * (M - idx)  # i * (M - i)
    term_xx = (diffs_xx * weights_xx.unsqueeze(0)).sum(dim=1) * 2.0 / (M * M)

    # ── Term 3: E[|Y - Y'|] via sorted trick ──
    Y_sorted = y_branches.sort(dim=1).values  # (B, N)
    diffs_yy = Y_sorted[:, 1:] - Y_sorted[:, :-1]  # (B, N-1)
    idx_y = torch.arange(1, N, device=y_branches.device, dtype=y_branches.dtype)
    weights_yy = idx_y * (N - idx_y)
    term_yy = (diffs_yy * weights_yy.unsqueeze(0)).sum(dim=1) * 2.0 / (N * N)

    ed = 2 * term_xy - term_xx - term_yy  # (B,)
    return ed.mean()


# ── NLL Loss ──────────────────────────────────────────────────────

def nll_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of targets under a Mixture of Gaussians.

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds.
        target: (B,)   observed cumulative log-return.

    Returns:
        Scalar mean NLL over the batch.
    """
    target = target.unsqueeze(-1)  # (B, 1)
    log_sigma = sigma.log()
    z = (target - mu) / sigma
    log_component = -0.5 * (LOG_2PI + 2 * log_sigma + z ** 2)
    log_mixture = torch.logsumexp(log_pi + log_component, dim=-1)
    return -log_mixture.mean()


# ── CRPS Loss (for validation) ────────────────────────────────────

def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / SQRT_2))


def _standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _A_function(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    ratio = mu / sigma
    return mu * _standard_normal_cdf(ratio) + sigma * _standard_normal_pdf(ratio)


def crps_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Closed-form CRPS for a Mixture of Gaussians.

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds.
        target: (B,)   observed cumulative log-return.

    Returns:
        Scalar mean CRPS over the batch.
    """
    pi = log_pi.exp()
    target = target.unsqueeze(-1)

    z = (target - mu) / sigma
    term1_per_k = sigma * (z * (2 * _standard_normal_cdf(z) - 1) + 2 * _standard_normal_pdf(z))
    term1 = (pi * term1_per_k).sum(dim=-1)

    mu_i = mu.unsqueeze(2)
    mu_j = mu.unsqueeze(1)
    sigma_i = sigma.unsqueeze(2)
    sigma_j = sigma.unsqueeze(1)
    pi_i = pi.unsqueeze(2)
    pi_j = pi.unsqueeze(1)

    mu_diff = mu_i - mu_j
    sigma_sum = torch.sqrt(sigma_i**2 + sigma_j**2)
    A_vals = _A_function(mu_diff, sigma_sum)
    term2 = (pi_i * pi_j * A_vals).sum(dim=(-1, -2))

    crps = term1 - term2
    return crps.mean()


# ── Combined Loss ─────────────────────────────────────────────────

def combined_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y_branches: torch.Tensor,
    n_model_samples: int = 256,
    nll_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined training loss: Energy Distance + auxiliary NLL.

    Energy distance is the primary loss (compares MoG samples vs MC ground truth).
    NLL on a random branch provides gradient flow to mixture weights (log_pi),
    since torch.multinomial in energy distance is not differentiable w.r.t. pi.

    Args:
        log_pi:     (B, K) log mixture weights.
        mu:         (B, K) component means.
        sigma:      (B, K) component stds.
        y_branches: (B, N) ground-truth branched future returns.
        n_model_samples: M samples from MoG for energy distance.
        nll_weight: weight for auxiliary NLL term.

    Returns:
        (total_loss, energy_dist, nll_term) — all scalars.
    """
    ed = energy_distance_loss(log_pi, mu, sigma, y_branches, n_model_samples)

    # NLL on a random branch per sample (for pi gradient flow)
    B, N = y_branches.shape
    rand_idx = torch.randint(N, (B,), device=y_branches.device)
    y_single = y_branches[torch.arange(B, device=y_branches.device), rand_idx]
    nll = nll_loss(log_pi, mu, sigma, y_single)

    total = ed + nll_weight * nll
    return total, ed, nll
