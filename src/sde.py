"""
Numba-JIT-compiled SDE simulators for synthetic data generation.

Each simulator runs at hourly resolution (dt = 1/(365*24)) and returns
daily log-returns. Seven SDE families are implemented:

v1 (original):
  - GBM (Geometric Brownian Motion)
  - Merton Jump-Diffusion
  - Kou Double-Exponential Jump-Diffusion
  - Bates (Stochastic Volatility + Jumps)
  - Regime-Switching GBM

v2 (additions):
  - Multifractal Random Walk (MRW) — captures scale-dependent vol clustering
  - Fractional OU with Stochastic Volatility — captures long-memory effects
"""

import numpy as np
from numba import njit


# ── Time discretisation ──────────────────────────────────────────────
DT = 1.0 / (365 * 24)        # hourly step in years
STEPS_PER_DAY = 24


# ═══════════════════════════════════════════════════════════════════
# v1 SDE simulators (original, numba-JIT)
# ═══════════════════════════════════════════════════════════════════

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


# ── With-state variants (return terminal state for branching) ────────

@njit(cache=True)
def _sim_gbm_with_state(mu, sigma, n_days):
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
    return returns, log_S


@njit(cache=True)
def _sim_merton_with_state(mu, sigma, lam, mu_j, sigma_j, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1
    for i in range(1, total_steps + 1):
        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW
        if np.random.rand() < lam * DT:
            log_S += np.random.normal(mu_j, sigma_j)
        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1
    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns, log_S


@njit(cache=True)
def _sim_kou_with_state(mu, sigma, lam, eta1, eta2, p, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1
    for i in range(1, total_steps + 1):
        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW
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
    return returns, log_S


@njit(cache=True)
def _sim_bates_with_state(mu, sigma, kappa, theta, xi, rho, lam, mu_j, sigma_j, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    v = theta
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
        if np.random.rand() < lam * DT:
            log_S += np.random.normal(mu_j, sigma_j)
        if i % STEPS_PER_DAY == 0:
            daily_closes[day_idx] = log_S
            day_idx += 1
    returns = np.empty(n_days)
    for j in range(n_days):
        returns[j] = daily_closes[j + 1] - daily_closes[j]
    return returns, log_S, v


@njit(cache=True)
def _sim_regime_switching_with_state(mus, sigmas, Q, n_regimes, n_days):
    total_steps = n_days * STEPS_PER_DAY
    log_S = 0.0
    regime = 0
    daily_closes = np.empty(n_days + 1)
    daily_closes[0] = 0.0
    day_idx = 1
    for i in range(1, total_steps + 1):
        mu = mus[regime]
        sigma = sigmas[regime]
        dW = np.sqrt(DT) * np.random.randn()
        log_S += (mu - 0.5 * sigma**2) * DT + sigma * dW
        total_rate = -Q[regime, regime]
        if total_rate > 0 and np.random.rand() < total_rate * DT:
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
    return returns, log_S, regime


# ── Forward-batch simulators (branch N paths from terminal state) ────

@njit(cache=True)
def _sim_gbm_forward_batch(mu, sigma, log_S0, n_days, n_branches):
    results = np.empty(n_branches, dtype=np.float32)
    total_steps = n_days * STEPS_PER_DAY
    for b in range(n_branches):
        log_S = log_S0
        for i in range(total_steps):
            log_S += (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * np.random.randn()
        results[b] = log_S - log_S0
    return results


@njit(cache=True)
def _sim_merton_forward_batch(mu, sigma, lam, mu_j, sigma_j, log_S0, n_days, n_branches):
    results = np.empty(n_branches, dtype=np.float32)
    total_steps = n_days * STEPS_PER_DAY
    for b in range(n_branches):
        log_S = log_S0
        for i in range(total_steps):
            log_S += (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * np.random.randn()
            if np.random.rand() < lam * DT:
                log_S += np.random.normal(mu_j, sigma_j)
        results[b] = log_S - log_S0
    return results


@njit(cache=True)
def _sim_kou_forward_batch(mu, sigma, lam, eta1, eta2, p, log_S0, n_days, n_branches):
    results = np.empty(n_branches, dtype=np.float32)
    total_steps = n_days * STEPS_PER_DAY
    for b in range(n_branches):
        log_S = log_S0
        for i in range(total_steps):
            log_S += (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * np.random.randn()
            if np.random.rand() < lam * DT:
                if np.random.rand() < p:
                    log_S += np.random.exponential(1.0 / eta1)
                else:
                    log_S -= np.random.exponential(1.0 / eta2)
        results[b] = log_S - log_S0
    return results


@njit(cache=True)
def _sim_bates_forward_batch(mu, sigma, kappa, theta, xi, rho, lam, mu_j, sigma_j,
                             log_S0, v0, n_days, n_branches):
    results = np.empty(n_branches, dtype=np.float32)
    total_steps = n_days * STEPS_PER_DAY
    rho_complement = np.sqrt(1.0 - rho**2)
    for b in range(n_branches):
        log_S = log_S0
        v = v0
        for i in range(total_steps):
            sqrt_v = np.sqrt(max(v, 0.0))
            dW_s = np.sqrt(DT) * np.random.randn()
            dW_v = rho * dW_s + rho_complement * np.sqrt(DT) * np.random.randn()
            log_S += (mu - 0.5 * v) * DT + sqrt_v * dW_s
            v += kappa * (theta - v) * DT + xi * sqrt_v * dW_v
            v = max(v, 0.0)
            if np.random.rand() < lam * DT:
                log_S += np.random.normal(mu_j, sigma_j)
        results[b] = log_S - log_S0
    return results


@njit(cache=True)
def _sim_regime_switching_forward_batch(mus, sigmas, Q, n_regimes,
                                        log_S0, regime0, n_days, n_branches):
    results = np.empty(n_branches, dtype=np.float32)
    total_steps = n_days * STEPS_PER_DAY
    for b in range(n_branches):
        log_S = log_S0
        regime = regime0
        for i in range(total_steps):
            mu = mus[regime]
            sigma_val = sigmas[regime]
            log_S += (mu - 0.5 * sigma_val**2) * DT + sigma_val * np.sqrt(DT) * np.random.randn()
            total_rate = -Q[regime, regime]
            if total_rate > 0 and np.random.rand() < total_rate * DT:
                u = np.random.rand() * total_rate
                cumulative = 0.0
                for j in range(n_regimes):
                    if j == regime:
                        continue
                    cumulative += Q[regime, j]
                    if u <= cumulative:
                        regime = j
                        break
        results[b] = log_S - log_S0
    return results


# ═══════════════════════════════════════════════════════════════════
# v2 SDE simulators (pure numpy — FFT-based, not numba)
# ═══════════════════════════════════════════════════════════════════

def _generate_fbm_increments(n, H, rng=None):
    """Generate fractional Brownian motion increments via Davies-Harte (circulant embedding).

    Args:
        n: Number of increments to generate.
        H: Hurst exponent in (0, 1).
        rng: numpy random generator.

    Returns:
        (n,) array of fBM increments.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Autocovariance of fGn: gamma(k) = 0.5 * (|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H))
    m = 2 * n
    k = np.arange(m)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    # Make it circulant: gamma_circ = [gamma(0), gamma(1), ..., gamma(n), gamma(n-1), ..., gamma(1)]
    gamma_circ = np.concatenate([gamma[:n + 1], gamma[n - 1:0:-1]])

    # Eigenvalues via FFT (should be non-negative for valid covariance)
    eigenvalues = np.fft.fft(gamma_circ).real
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety

    # Sample in frequency domain
    z = rng.standard_normal(len(eigenvalues)) + 1j * rng.standard_normal(len(eigenvalues))
    w = np.fft.ifft(np.sqrt(eigenvalues) * z).real

    return w[:n]


def _sim_mrw_daily(mu, sigma, lam_param, T_days, n_days, rng=None):
    """Simulate Multifractal Random Walk and return daily log-returns.

    The MRW generates returns with multifractal (scale-dependent) vol clustering:
        r_t = mu*dt + sigma * exp(omega_t - lam^2 * log(T)) * epsilon_t

    where omega_t is a Gaussian process with covariance:
        Cov(omega_i, omega_j) = lam^2 * log(T / max(|i-j|, 1))

    Args:
        mu: Annualized drift.
        sigma: Base annualized volatility.
        lam_param: Intermittency parameter (controls multifractality strength).
        T_days: Integral scale in days.
        n_days: Number of days to simulate.
        rng: numpy random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = n_days
    T = max(T_days, n + 1)  # integral scale must exceed series length

    # Build covariance matrix for omega
    # Cov(i, j) = lam^2 * log(T / max(|i-j|, 1)) for |i-j| < T, else 0
    lags = np.arange(n)
    cov_row = np.zeros(n)
    cov_row[0] = lam_param ** 2 * np.log(T)
    for k in range(1, min(n, T)):
        cov_row[k] = lam_param ** 2 * np.log(T / k)

    # Circulant embedding for fast Gaussian field generation
    cov_circ = np.concatenate([cov_row, cov_row[-2:0:-1]])
    eigenvalues = np.fft.fft(cov_circ).real
    eigenvalues = np.maximum(eigenvalues, 0)

    z = rng.standard_normal(len(eigenvalues)) + 1j * rng.standard_normal(len(eigenvalues))
    omega = np.fft.ifft(np.sqrt(eigenvalues) * z).real[:n]

    # Generate returns
    dt_day = 1.0 / 365
    daily_vol = sigma * np.sqrt(dt_day)
    epsilon = rng.standard_normal(n)

    # Normalize omega so variance matches lam^2 * log(T)
    vol_multiplier = np.exp(omega - 0.5 * lam_param ** 2 * np.log(T))
    returns = mu * dt_day + daily_vol * vol_multiplier * epsilon

    return returns.astype(np.float32)


def _sim_frac_ou_daily(mu, sigma, theta, kappa_vol, xi_vol, H, n_days, rng=None):
    """Simulate Fractional OU with Stochastic Volatility, return daily log-returns.

    Price dynamics (daily):
        r_t = mu*dt + sigma_t * dB^H_t

    where B^H is fractional Brownian motion with Hurst parameter H, and:
        log(sigma_t) follows an OU process:
        d(log sigma_t) = kappa_vol * (log(sigma) - log(sigma_t)) * dt + xi_vol * dW_t

    Args:
        mu: Annualized drift.
        sigma: Long-run annualized volatility.
        theta: Mean-reversion speed for the OU component of returns.
        kappa_vol: Mean-reversion speed for log-vol.
        xi_vol: Volatility of log-volatility.
        H: Hurst exponent for fBM (< 0.5 = mean-reverting, > 0.5 = trending).
        n_days: Number of days to simulate.
        rng: numpy random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt_day = 1.0 / 365

    # Generate fBM increments for the price process
    # fGn from Davies-Harte has unit variance; scale to daily vol = sigma * sqrt(dt)
    fbm_increments = _generate_fbm_increments(n_days, H, rng)
    # Normalize to unit variance (fGn variance depends on H, normalize empirically)
    inc_std = np.std(fbm_increments)
    if inc_std > 1e-12:
        fbm_increments = fbm_increments / inc_std

    # Simulate stochastic log-vol (standard OU with BM, not fBM)
    log_vol = np.zeros(n_days)
    log_sigma_base = np.log(sigma)
    log_vol[0] = log_sigma_base

    for t in range(1, n_days):
        dW = np.sqrt(dt_day) * rng.standard_normal()
        log_vol[t] = log_vol[t - 1] + kappa_vol * (log_sigma_base - log_vol[t - 1]) * dt_day + xi_vol * dW

    sigma_t = np.exp(log_vol)  # (n_days,) instantaneous annualized vol
    daily_vol = sigma_t * np.sqrt(dt_day)  # convert annualized to daily

    # Combine: returns with fractional noise and stochastic vol
    returns = mu * dt_day + daily_vol * fbm_increments

    return returns.astype(np.float32)


# ── v2 with-state and forward-batch variants ───────────────────────

def _sim_mrw_with_state(mu, sigma, lam_param, T_days, n_days, rng=None):
    """MRW with terminal state for branching."""
    returns = _sim_mrw_daily(mu, sigma, lam_param, T_days, n_days, rng)
    log_S = float(np.sum(returns))
    return returns, log_S


def _sim_frac_ou_with_state(mu, sigma, theta, kappa_vol, xi_vol, H, n_days, rng=None):
    """Fractional OU with terminal state for branching."""
    returns = _sim_frac_ou_daily(mu, sigma, theta, kappa_vol, xi_vol, H, n_days, rng)
    log_S = float(np.sum(returns))
    return returns, log_S


def _sim_mrw_forward_batch(mu, sigma, lam_param, T_days, log_S0, n_days, n_branches, rng=None):
    """Branch N MRW paths from terminal state."""
    if rng is None:
        rng = np.random.default_rng()
    results = np.empty(n_branches, dtype=np.float32)
    for b in range(n_branches):
        returns = _sim_mrw_daily(mu, sigma, lam_param, T_days, n_days, rng)
        results[b] = np.sum(returns)
    return results


def _sim_frac_ou_forward_batch(mu, sigma, theta, kappa_vol, xi_vol, H,
                                log_S0, n_days, n_branches, rng=None):
    """Branch N Fractional OU paths from terminal state."""
    if rng is None:
        rng = np.random.default_rng()
    results = np.empty(n_branches, dtype=np.float32)
    for b in range(n_branches):
        returns = _sim_frac_ou_daily(mu, sigma, theta, kappa_vol, xi_vol, H, n_days, rng)
        results[b] = np.sum(returns)
    return results


# ═══════════════════════════════════════════════════════════════════
# Context + branched futures dispatch
# ═══════════════════════════════════════════════════════════════════

def simulate_context_and_branches(sde_type, params, context_days, horizon_days, n_branches):
    """Simulate context path then branch N independent future paths.

    This is the JointFM-style data generation: one history realization,
    many future continuations from the same terminal state.

    Parameters
    ----------
    sde_type : str
        One of 'gbm', 'merton', 'kou', 'bates', 'regime_switching', 'mrw', 'frac_ou'.
    params : dict
        Parameter dict from sample_params.
    context_days : int
        Days of context (model input).
    horizon_days : int
        Days of forward simulation per branch.
    n_branches : int
        Number of independent future paths to branch.

    Returns
    -------
    context_returns : np.ndarray, shape (context_days,)
        Daily log-returns for the context window.
    branches : np.ndarray, shape (n_branches,)
        Cumulative log-return for each branched future path.
    """
    if sde_type == 'gbm':
        ctx, log_S = _sim_gbm_with_state(params['mu'], params['sigma'], context_days)
        branches = _sim_gbm_forward_batch(
            params['mu'], params['sigma'], log_S, horizon_days, n_branches)

    elif sde_type == 'merton':
        ctx, log_S = _sim_merton_with_state(
            params['mu'], params['sigma'],
            params['lam'], params['mu_j'], params['sigma_j'], context_days)
        branches = _sim_merton_forward_batch(
            params['mu'], params['sigma'],
            params['lam'], params['mu_j'], params['sigma_j'],
            log_S, horizon_days, n_branches)

    elif sde_type == 'kou':
        ctx, log_S = _sim_kou_with_state(
            params['mu'], params['sigma'],
            params['lam'], params['eta1'], params['eta2'], params['p'], context_days)
        branches = _sim_kou_forward_batch(
            params['mu'], params['sigma'],
            params['lam'], params['eta1'], params['eta2'], params['p'],
            log_S, horizon_days, n_branches)

    elif sde_type == 'bates':
        ctx, log_S, v_term = _sim_bates_with_state(
            params['mu'], params['sigma'],
            params['kappa'], params['theta'], params['xi'], params['rho'],
            params['lam'], params['mu_j'], params['sigma_j'], context_days)
        branches = _sim_bates_forward_batch(
            params['mu'], params['sigma'],
            params['kappa'], params['theta'], params['xi'], params['rho'],
            params['lam'], params['mu_j'], params['sigma_j'],
            log_S, v_term, horizon_days, n_branches)

    elif sde_type == 'regime_switching':
        ctx, log_S, regime_term = _sim_regime_switching_with_state(
            params['mus'], params['sigmas'],
            params['Q'], params['n_regimes'], context_days)
        branches = _sim_regime_switching_forward_batch(
            params['mus'], params['sigmas'],
            params['Q'], params['n_regimes'],
            log_S, regime_term, horizon_days, n_branches)

    elif sde_type == 'mrw':
        rng = params.get('_rng', None)
        ctx, log_S = _sim_mrw_with_state(
            params['mu'], params['sigma'],
            params['lam_param'], params['T_days'], context_days, rng)
        branches = _sim_mrw_forward_batch(
            params['mu'], params['sigma'],
            params['lam_param'], params['T_days'],
            log_S, horizon_days, n_branches, rng)

    elif sde_type == 'frac_ou':
        rng = params.get('_rng', None)
        ctx, log_S = _sim_frac_ou_with_state(
            params['mu'], params['sigma'],
            params['theta'], params['kappa_vol'], params['xi_vol'],
            params['H'], context_days, rng)
        branches = _sim_frac_ou_forward_batch(
            params['mu'], params['sigma'],
            params['theta'], params['kappa_vol'], params['xi_vol'],
            params['H'], log_S, horizon_days, n_branches, rng)

    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")

    return ctx, branches


# ── Parameter sampling ───────────────────────────────────────────────

def sample_params(sde_type, rng=None):
    """Draw random parameters from priors for the given SDE type.

    Parameters
    ----------
    sde_type : str
        One of 'gbm', 'merton', 'kou', 'bates', 'regime_switching', 'mrw', 'frac_ou'.
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
        Q = np.zeros((n_regimes, n_regimes))
        for r in range(n_regimes):
            for c in range(n_regimes):
                if r != c:
                    mean_days = rng.exponential(30.0)
                    mean_days = max(mean_days, 5.0)
                    Q[r, c] = 365.0 / mean_days
            Q[r, r] = -Q[r, :].sum()

        return {
            'mus': mus, 'sigmas': sigmas,
            'Q': Q, 'n_regimes': n_regimes,
        }

    elif sde_type == 'mrw':
        return {
            'mu': mu,
            'sigma': sigma,
            'lam_param': rng.uniform(0.01, 0.15),   # intermittency
            'T_days': int(rng.uniform(30, 365)),     # integral scale
            '_rng': rng,
        }

    elif sde_type == 'frac_ou':
        return {
            'mu': mu,
            'sigma': sigma,
            'theta': rng.uniform(0.5, 10),           # mean reversion speed
            'kappa_vol': rng.uniform(0.5, 5),        # vol mean reversion
            'xi_vol': rng.uniform(0.1, 0.5),         # vol of vol
            'H': rng.uniform(0.3, 0.7),              # Hurst exponent
            '_rng': rng,
        }

    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")


# ── Dispatch ─────────────────────────────────────────────────────────

def simulate_daily_returns(sde_type, params, n_days):
    """Simulate at hourly resolution, return daily log-returns.

    Parameters
    ----------
    sde_type : str
        One of 'gbm', 'merton', 'kou', 'bates', 'regime_switching', 'mrw', 'frac_ou'.
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

    elif sde_type == 'mrw':
        rng = params.get('_rng', None)
        return _sim_mrw_daily(
            params['mu'], params['sigma'],
            params['lam_param'], params['T_days'],
            n_days, rng,
        )

    elif sde_type == 'frac_ou':
        rng = params.get('_rng', None)
        return _sim_frac_ou_daily(
            params['mu'], params['sigma'],
            params['theta'], params['kappa_vol'], params['xi_vol'],
            params['H'], n_days, rng,
        )

    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")
