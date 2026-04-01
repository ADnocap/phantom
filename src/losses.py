"""
Loss functions for Mixture of Gaussians distributional prediction.

Implements:
  - NLL:  Negative log-likelihood of observed return under predicted MoG
  - CRPS: Closed-form Continuous Ranked Probability Score for MoG
  - Combined loss: NLL + alpha * CRPS

References:
  - Gneiting & Raftery (2007) for CRPS as a proper scoring rule
  - Grimit et al. (2006) for closed-form MoG CRPS
"""

import math
import torch
import torch.nn.functional as F


LOG_2PI = math.log(2 * math.pi)
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)


def nll_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of targets under a Mixture of Gaussians.

    NLL = -log Σ_k π_k * N(target | μ_k, σ_k²)

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds.
        target: (B,)   observed cumulative log-return.

    Returns:
        Scalar mean NLL over the batch.
    """
    target = target.unsqueeze(-1)  # (B, 1)

    # Log probability of target under each component: log N(target | μ_k, σ_k²)
    log_sigma = sigma.log()
    z = (target - mu) / sigma
    log_component = -0.5 * (LOG_2PI + 2 * log_sigma + z ** 2)  # (B, K)

    # Log-sum-exp over mixture: log Σ_k π_k * N(...)
    log_mixture = torch.logsumexp(log_pi + log_component, dim=-1)  # (B,)

    return -log_mixture.mean()


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Φ(x) = 0.5 * (1 + erf(x / √2))."""
    return 0.5 * (1 + torch.erf(x / SQRT_2))


def _standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = exp(-x²/2) / √(2π)."""
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _A_function(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Helper: A(μ, σ) = μ * Φ(μ/σ) + σ * φ(μ/σ).

    Used in the closed-form CRPS for a single Gaussian:
        CRPS(N(μ, σ²), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ.

    But for the MoG mixture CRPS we need A(μ, σ) which arises in the
    E[|X - X'|] term between two Gaussian components.
    """
    ratio = mu / sigma
    return mu * _standard_normal_cdf(ratio) + sigma * _standard_normal_pdf(ratio)


def crps_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Closed-form CRPS for a Mixture of Gaussians.

    CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]

    For MoG with weights π_k, means μ_k, stds σ_k:

    E_F[|X - y|] = Σ_k π_k * σ_k * [z_k(2Φ(z_k)-1) + 2φ(z_k)]
      where z_k = (y - μ_k) / σ_k

    E_F[|X - X'|] = Σ_k Σ_j π_k π_j * 2A(μ_k - μ_j, √(σ_k² + σ_j²))

    Reference: Grimit et al. (2006), "The CRPS for Gaussian mixture models".

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds.
        target: (B,)   observed cumulative log-return.

    Returns:
        Scalar mean CRPS over the batch.
    """
    pi = log_pi.exp()  # (B, K)
    K = mu.size(-1)
    target = target.unsqueeze(-1)  # (B, 1)

    # ── Term 1: E[|X - y|] ──
    z = (target - mu) / sigma  # (B, K)
    term1_per_k = sigma * (z * (2 * _standard_normal_cdf(z) - 1) + 2 * _standard_normal_pdf(z))
    term1 = (pi * term1_per_k).sum(dim=-1)  # (B,)

    # ── Term 2: E[|X - X'|] ──
    # We need all (k, j) pairs. Expand to (B, K, K).
    mu_i = mu.unsqueeze(2)       # (B, K, 1)
    mu_j = mu.unsqueeze(1)       # (B, 1, K)
    sigma_i = sigma.unsqueeze(2) # (B, K, 1)
    sigma_j = sigma.unsqueeze(1) # (B, 1, K)
    pi_i = pi.unsqueeze(2)       # (B, K, 1)
    pi_j = pi.unsqueeze(1)       # (B, 1, K)

    mu_diff = mu_i - mu_j                          # (B, K, K)
    sigma_sum = torch.sqrt(sigma_i**2 + sigma_j**2)  # (B, K, K)

    A_vals = _A_function(mu_diff, sigma_sum)        # (B, K, K)
    term2 = (pi_i * pi_j * A_vals).sum(dim=(-1, -2))  # (B,)

    crps = term1 - term2  # (B,)
    return crps.mean()


def combined_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined training loss: NLL + α * CRPS.

    NLL drives sharp density estimation; CRPS regularizes calibration.

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds.
        target: (B,)   observed cumulative log-return.
        alpha:  weight for CRPS term (default 0.3).

    Returns:
        (total_loss, nll_term, crps_term) — all scalars.
    """
    nll = nll_loss(log_pi, mu, sigma, target)
    crps = crps_loss(log_pi, mu, sigma, target)
    total = nll + alpha * crps
    return total, nll, crps
