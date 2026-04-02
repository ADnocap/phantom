"""
Loss functions for Phantom distributional prediction.

Implements:
  - Energy Distance: primary loss comparing mixture samples vs branched MC ground truth
  - NLL:  Negative log-likelihood (supports both Gaussian and Student-t mixtures)
  - CRPS: Closed-form for Gaussian MoG, sample-based for Student-t (validation metric)
  - Quantile Loss: Pinball loss at specified quantile levels (auxiliary calibration)
  - Combined loss: Energy Distance + nll_weight * NLL

References:
  - Gneiting & Raftery (2007) for energy distance and CRPS
  - Grimit et al. (2006) for closed-form MoG CRPS
  - JointFM-0.1 for the branched-future training paradigm
  - Jang et al. (2017) for Gumbel-Softmax reparameterization
"""

import math
import torch
import torch.nn.functional as F


LOG_2PI = math.log(2 * math.pi)
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)


# ── Mixture sampling ─────────────────────────────────────────────

def _sample_mixture(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor | None,
    n_samples: int,
    use_gumbel_softmax: bool = False,
    gumbel_tau: float = 0.5,
) -> torch.Tensor:
    """Sample from a mixture distribution (Gaussian or Student-t).

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component locations.
        sigma:  (B, K) component scales.
        nu:     (B, K) degrees of freedom (None for Gaussian).
        n_samples: M, number of samples to draw.
        use_gumbel_softmax: Use differentiable Gumbel-Softmax for component selection.
        gumbel_tau: Temperature for Gumbel-Softmax.

    Returns:
        X: (B, M) samples from the mixture.
    """
    B, K = mu.shape
    M = n_samples

    if use_gumbel_softmax:
        # Differentiable component selection via Gumbel-Softmax
        logits = log_pi.unsqueeze(1).expand(B, M, K)
        weights = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)  # (B, M, K)
        mu_sel = (weights * mu.unsqueeze(1)).sum(-1)       # (B, M)
        sigma_sel = (weights * sigma.unsqueeze(1)).sum(-1)  # (B, M)
        if nu is not None:
            nu_sel = (weights * nu.unsqueeze(1)).sum(-1)    # (B, M)
        else:
            nu_sel = None
    else:
        # Standard multinomial sampling (non-differentiable w.r.t. log_pi)
        pi = log_pi.exp()
        comp_idx = torch.multinomial(pi, M, replacement=True)  # (B, M)
        mu_sel = torch.gather(mu, 1, comp_idx)
        sigma_sel = torch.gather(sigma, 1, comp_idx)
        if nu is not None:
            nu_sel = torch.gather(nu, 1, comp_idx)
        else:
            nu_sel = None

    # Reparameterized sampling
    z = torch.randn_like(mu_sel)  # (B, M)
    if nu_sel is not None:
        # Student-t: X = mu + sigma * z * sqrt(nu / chi2)
        # where chi2 ~ Chi2(nu)
        chi2 = torch.distributions.Chi2(nu_sel).rsample()
        X = mu_sel + sigma_sel * z * torch.sqrt(nu_sel / chi2)
    else:
        # Gaussian: X = mu + sigma * z
        X = mu_sel + sigma_sel * z

    return X


# ── Energy Distance Loss ──────────────────────────────────────────

def energy_distance_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y_branches: torch.Tensor,
    n_model_samples: int = 256,
    nu: torch.Tensor | None = None,
    use_gumbel_softmax: bool = False,
    gumbel_tau: float = 0.5,
) -> torch.Tensor:
    """Energy distance between predicted mixture and empirical branch distribution.

    ED(P, Q) = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]

    where X ~ predicted mixture (via reparameterization), Y ~ ground-truth branches.

    Args:
        log_pi:    (B, K) log mixture weights.
        mu:        (B, K) component means.
        sigma:     (B, K) component stds.
        y_branches: (B, N) ground-truth branched future returns.
        n_model_samples: M, samples to draw from predicted mixture.
        nu:        (B, K) degrees of freedom (None for Gaussian mixture).
        use_gumbel_softmax: Use differentiable Gumbel-Softmax for component selection.
        gumbel_tau: Temperature for Gumbel-Softmax.

    Returns:
        Scalar mean energy distance over the batch.
    """
    N = y_branches.size(1)
    M = n_model_samples

    # ── Sample from mixture ──
    X = _sample_mixture(log_pi, mu, sigma, nu, M,
                        use_gumbel_softmax, gumbel_tau)  # (B, M)

    # ── Term 1: E[|X - Y|] (cross term) ──
    cross = torch.abs(X.unsqueeze(2) - y_branches.unsqueeze(1))  # (B, M, N)
    term_xy = cross.mean(dim=(1, 2))  # (B,)

    # ── Term 2: E[|X - X'|] via sorted trick ──
    X_sorted = X.sort(dim=1).values  # (B, M)
    diffs_xx = X_sorted[:, 1:] - X_sorted[:, :-1]  # (B, M-1)
    idx = torch.arange(1, M, device=X.device, dtype=X.dtype)
    weights_xx = idx * (M - idx)
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
    nu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood of targets under a mixture distribution.

    Supports both Gaussian (nu=None) and Student-t (nu provided) components.

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds/scales.
        target: (B,)   observed cumulative log-return.
        nu:     (B, K) degrees of freedom (None for Gaussian).

    Returns:
        Scalar mean NLL over the batch.
    """
    target = target.unsqueeze(-1)  # (B, 1)
    z = (target - mu) / sigma

    if nu is not None:
        # Student-t log-pdf
        log_component = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * math.pi)
            - sigma.log()
            - ((nu + 1) / 2) * torch.log(1 + z ** 2 / nu)
        )
    else:
        # Gaussian log-pdf
        log_sigma = sigma.log()
        log_component = -0.5 * (LOG_2PI + 2 * log_sigma + z ** 2)

    log_mixture = torch.logsumexp(log_pi + log_component, dim=-1)
    return -log_mixture.mean()


# ── CRPS Loss ────────────────────────────────────────────────────

def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / SQRT_2))


def _standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _A_function(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    ratio = mu / sigma
    return mu * _standard_normal_cdf(ratio) + sigma * _standard_normal_pdf(ratio)


def _crps_gaussian(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Closed-form CRPS for a Mixture of Gaussians."""
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


def _crps_sample_based(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    nu: torch.Tensor | None = None,
    n_samples: int = 512,
) -> torch.Tensor:
    """Sample-based CRPS for any mixture distribution.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    """
    X = _sample_mixture(log_pi, mu, sigma, nu, n_samples)  # (B, M)
    target_exp = target.unsqueeze(1)  # (B, 1)

    # E[|X - y|]
    term1 = torch.abs(X - target_exp).mean(dim=1)  # (B,)

    # E[|X - X'|] via sorted trick
    M = n_samples
    X_sorted = X.sort(dim=1).values
    diffs = X_sorted[:, 1:] - X_sorted[:, :-1]
    idx = torch.arange(1, M, device=X.device, dtype=X.dtype)
    weights = idx * (M - idx)
    term2 = (diffs * weights.unsqueeze(0)).sum(dim=1) * 2.0 / (M * M)

    crps = term1 - 0.5 * term2
    return crps.mean()


def crps_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    nu: torch.Tensor | None = None,
) -> torch.Tensor:
    """CRPS loss for mixture distributions.

    Uses closed-form for Gaussian mixtures, sample-based for Student-t.

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds/scales.
        target: (B,)   observed cumulative log-return.
        nu:     (B, K) degrees of freedom (None for Gaussian).

    Returns:
        Scalar mean CRPS over the batch.
    """
    if nu is None:
        return _crps_gaussian(log_pi, mu, sigma, target)
    else:
        return _crps_sample_based(log_pi, mu, sigma, target, nu)


# ── Mixture CDF ──────────────────────────────────────────────────

def mixture_cdf(
    y: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor | None = None,
) -> torch.Tensor:
    """CDF of a Gaussian mixture distribution evaluated at y.

    Note: Only supports Gaussian components (nu must be None).
    For Student-t, use sample-based methods instead.

    Args:
        y:     (B,) or (B, 1) evaluation points.
        pi:    (B, K) mixture weights (NOT log).
        mu:    (B, K) component locations.
        sigma: (B, K) component scales.
        nu:    Ignored (kept for API consistency). Must be None.

    Returns:
        (B,) CDF values.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)  # (B, 1)

    z = (y - mu) / sigma  # (B, K)
    component_cdf = 0.5 * (1 + torch.erf(z / SQRT_2))  # (B, K)

    return (pi * component_cdf).sum(dim=-1)  # (B,)


# ── Quantile Loss ────────────────────────────────────────────────

def quantile_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    nu: torch.Tensor | None = None,
    levels: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    n_samples: int = 1000,
) -> torch.Tensor:
    """Pinball (quantile) loss at specified quantile levels.

    Sample-based: draws from the mixture, sorts to get quantiles,
    then applies asymmetric pinball loss. Fully differentiable via
    reparameterization. Works with both Gaussian and Student-t.

    Args:
        log_pi: (B, K) log mixture weights.
        mu:     (B, K) component means.
        sigma:  (B, K) component stds/scales.
        target: (B,)   observed values.
        nu:     (B, K) degrees of freedom (None for Gaussian).
        levels: Quantile levels to evaluate.
        n_samples: Number of samples for quantile estimation.

    Returns:
        Scalar mean pinball loss over batch and quantile levels.
    """
    # Sample from mixture and sort
    X = _sample_mixture(log_pi, mu, sigma, nu, n_samples)  # (B, M)
    X_sorted = X.sort(dim=1).values  # (B, M)

    total = torch.zeros(1, device=target.device, dtype=target.dtype)
    M = n_samples

    for tau in levels:
        # Predicted quantile from sorted samples
        idx = min(int(tau * M), M - 1)
        q = X_sorted[:, idx]  # (B,)

        # Pinball loss: asymmetric L1
        err = target - q
        pinball = torch.where(err >= 0, tau * err, (tau - 1) * err)
        total = total + pinball.mean()

    return total / len(levels)


# ── Combined Loss ─────────────────────────────────────────────────

def combined_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y_branches: torch.Tensor,
    n_model_samples: int = 256,
    nll_weight: float = 0.1,
    nu: torch.Tensor | None = None,
    use_gumbel_softmax: bool = False,
    gumbel_tau: float = 0.5,
    quantile_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined training loss: Energy Distance + auxiliary NLL + optional quantile.

    Args:
        log_pi:     (B, K) log mixture weights.
        mu:         (B, K) component means.
        sigma:      (B, K) component stds.
        y_branches: (B, N) ground-truth branched future returns.
        n_model_samples: M samples from mixture for energy distance.
        nll_weight: weight for auxiliary NLL term.
        nu:         (B, K) degrees of freedom (None for Gaussian).
        use_gumbel_softmax: Use Gumbel-Softmax in energy distance.
        gumbel_tau: Temperature for Gumbel-Softmax.
        quantile_weight: weight for auxiliary quantile loss (0 = disabled).

    Returns:
        (total_loss, energy_dist, nll_term) — all scalars.
    """
    ed = energy_distance_loss(log_pi, mu, sigma, y_branches,
                              n_model_samples, nu,
                              use_gumbel_softmax, gumbel_tau)

    # NLL on a random branch per sample (for pi gradient flow)
    B, N = y_branches.shape
    rand_idx = torch.randint(N, (B,), device=y_branches.device)
    y_single = y_branches[torch.arange(B, device=y_branches.device), rand_idx]
    nll = nll_loss(log_pi, mu, sigma, y_single, nu)

    total = ed + nll_weight * nll

    # Optional quantile loss on the same random branch
    if quantile_weight > 0:
        ql = quantile_loss(log_pi, mu, sigma, y_single, nu)
        total = total + quantile_weight * ql

    return total, ed, nll
