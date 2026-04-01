"""
Numba-JIT-compiled SDE simulators for synthetic data generation.

Each simulator runs at hourly resolution (dt = 1/(365*24)) and returns
daily log-returns. Five SDE families are implemented:
  - GBM (Geometric Brownian Motion)
  - Merton Jump-Diffusion
  - Kou Double-Exponential Jump-Diffusion
  - Bates (Stochastic Volatility + Jumps)
  - Regime-Switching GBM
"""

import numpy as np
from numba import njit


# ── Time discretisation ──────────────────────────────────────────────
DT = 1.0 / (365 * 24)        # hourly step in years
STEPS_PER_DAY = 24


# ── Individual SDE simulators ────────────────────────────────────────

@njit(cache=True)
def _sim_gbm(mu, sigma, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1

    for i in range(1, total_steps + 1):
        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW
        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1

    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns


@njit(cache=True)
def _sim_merton(mu, sigma, lam, mu_j, sigma_j, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1

    for i in range(1, total_steps + 1):
        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW
        # Poisson jump
        if np.random.rand() < lam * DT:
            log_S += np.random.normal(mu_j, sigma_j)
        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1

    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns


@njit(cache=True)
def _sim_kou(mu, sigma, lam, eta1, eta2, p, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1

    for i in range(1, total_steps + 1):
        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW
        # Poisson jump with double-exponential size
        if np.random.rand() < lam * DT:
            if np.random.rand() < p:
                log_S += np.random.exponential(1.0 / eta1)
            else:
                log_S -= np.random.exponential(1.0 / eta2)
        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1

    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns


@njit(cache=True)
def _sim_bates(mu, sigma, kappa, theta, xi, rho, lam, mu_j, sigma_j, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    v = theta  # start variance at long-run mean
    rho_complement = np.sqrt(1.0 - rho**2)
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1

    for i in range(1, total_steps + 1):
        sqrt_v = np.sqrt(max(v, 0.0))
        dW_s = np.sqrt(DT) * np.random.randn()
        dW_v = rho * dW_s + rho_complement * np.sqrt(DT) * np.random.randn()

        log_S += (mu - 0.5 * v) * DT + sqrt_v * dW_s
        v += kappa * (theta - v) * DT + xi * sqrt_v * dW_v
        v = max(v, 0.0)

        # Poisson jump
        if np.random.rand() < lam * DT:
            log_S += np.random.normal(mu_j, sigma_j)

        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1

    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns


@njit(cache=True)
def _sim_regime_switching(mus, sigmas, Q, n_regimes, n_days):
    """Regime-switching GBM with continuous-time Markov chain.

    Q is a (n_regimes, n_regimes) generator matrix where Q[i,j] >= 0
    for i != j is the transition rate from regime i to j, and
    Q[i,i] = -sum_{j!=i} Q[i,j].
    """
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    regime = 0  # start in regime 0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1

    for i in range(1, total_steps + 1):
        mu = mus[regime]
        sigma = sigmas[regime]

        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW

        # Check for regime transition
        total_rate = -Q[regime, regime]
        if total_rate > 0 and np.random.rand() < total_rate * DT:
            # Transition: pick destination proportional to off-diagonal rates
            u = np.random.rand() * total_rate
            cumulative = 0.0
            for j in range(n_regimes):
                if j == regime:
                    continue
                cumulative += Q[regime, j]
                if u <= cumulative:
                    regime = j
                    break

        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1

    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns


# ── Parameter sampling ───────────────────────────────────────────────

def sample_params(sde_type, rng=None):
    """Draw random parameters from priors for the given SDE type.

    Parameters
    ----------
    sde_type : str
        One of 'gbm', 'merton', 'kou', 'bates', 'regime_switching'.
    rng : np.random.Generator or None
        Random number generator. If None, uses the default.

    Returns
    -------
    dict
        Parameter dictionary ready for simulate_daily_returns.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = rng.uniform(-0.5, 1.5)
    sigma = rng.uniform(0.3, 1.5)

    if sde_type == 'gbm':
        return {'mu': mu, 'sigma': sigma}

    elif sde_type == 'merton':
        return {
            'mu': mu, 'sigma': sigma,
            'lam': rng.uniform(0.5, 50),
            'mu_j': rng.uniform(-0.1, 0.05),
            'sigma_j': rng.uniform(0.01, 0.15),
        }

    elif sde_type == 'kou':
        return {
            'mu': mu, 'sigma': sigma,
            'lam': rng.uniform(0.5, 50),
            'eta1': rng.uniform(7, 100),
            'eta2': rng.uniform(4, 50),
            'p': rng.beta(2, 3),
        }

    elif sde_type == 'bates':
        return {
            'mu': mu, 'sigma': sigma,
            'kappa': rng.uniform(0.5, 10),
            'theta': rng.uniform(0.1, 2.0),
            'xi': rng.uniform(0.2, 2.0),
            'rho': rng.uniform(-0.9, -0.1),
            'lam': rng.uniform(0.5, 50),
            'mu_j': rng.uniform(-0.1, 0.05),
            'sigma_j': rng.uniform(0.01, 0.15),
        }

    elif sde_type == 'regime_switching':
        n_regimes = rng.choice([2, 3])
        mus = rng.uniform(-0.5, 1.5, size=n_regimes)
        sigmas = rng.uniform(0.3, 1.5, size=n_regimes)
        # Build generator matrix Q
        # Off-diagonal rates ~ Exponential with mean duration ~30 days
        Q = np.zeros((n_regimes, n_regimes))
        for r in range(n_regimes):
            for c in range(n_regimes):
                if r != c:
                    # rate = 1 / mean_duration_in_years
                    mean_days = rng.exponential(30.0)
                    mean_days = max(mean_days, 5.0)  # floor at 5 days
                    Q[r, c] = 365.0 / mean_days
            Q[r, r] = -Q[r, :].sum()

        return {
            'mus': mus, 'sigmas': sigmas,
            'Q': Q, 'n_regimes': n_regimes,
        }

    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")


# ── Dispatch ─────────────────────────────────────────────────────────

def simulate_daily_returns(sde_type, params, n_days):
    """Simulate at hourly resolution, return daily log-returns.

    Parameters
    ----------
    sde_type : str
        One of 'gbm', 'merton', 'kou', 'bates', 'regime_switching'.
    params : dict
        Parameter dict from sample_params.
    n_days : int
        Number of days to simulate.

    Returns
    -------
    np.ndarray
        Array of shape (n_days,) with daily log-returns.
    """
    if sde_type == 'gbm':
        return _sim_gbm(params['mu'], params['sigma'], n_days)

    elif sde_type == 'merton':
        return _sim_merton(
            params['mu'], params['sigma'],
            params['lam'], params['mu_j'], params['sigma_j'],
            n_days,
        )

    elif sde_type == 'kou':
        return _sim_kou(
            params['mu'], params['sigma'],
            params['lam'], params['eta1'], params['eta2'], params['p'],
            n_days,
        )

    elif sde_type == 'bates':
        return _sim_bates(
            params['mu'], params['sigma'],
            params['kappa'], params['theta'], params['xi'], params['rho'],
            params['lam'], params['mu_j'], params['sigma_j'],
            n_days,
        )

    elif sde_type == 'regime_switching':
        return _sim_regime_switching(
            params['mus'], params['sigmas'],
            params['Q'], params['n_regimes'],
            n_days,
        )

    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")
