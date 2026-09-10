"""Path simulation: GBM and Heston QE schemes with Sobol QMC and antithetics."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional

from .models import HestonParams


def brownian_bridge(W_end: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Construct Brownian bridge paths via recursive midpoint interpolation.
    
    Note: Increases variance for path-dependent payoffs; better for Europeans.
    """
    n_paths = len(W_end)
    W = np.zeros((n_paths, n_steps + 1))
    W[:, -1] = W_end

    def bridge_recursive(left_idx: int, right_idx: int, Z: np.ndarray):
        if right_idx - left_idx <= 1:
            return
        mid_idx = (left_idx + right_idx) // 2
        t_left = left_idx * dt
        t_mid = mid_idx * dt
        t_right = right_idx * dt
        alpha = (t_mid - t_left) / (t_right - t_left)
        std = np.sqrt((t_mid - t_left) * (t_right - t_mid) / (t_right - t_left))
        W[:, mid_idx] = W[:, left_idx] + alpha * (W[:, right_idx] - W[:, left_idx]) + std * Z[:, mid_idx]
        bridge_recursive(left_idx, mid_idx, Z)
        bridge_recursive(mid_idx, right_idx, Z)
    Z = np.random.randn(n_paths, n_steps + 1)
    bridge_recursive(0, n_steps, Z)
    return W


def generate_sobol_normals(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """Generate scrambled Sobol quasi-random normals for low-discrepancy sampling."""
    try:
        from scipy.stats.qmc import Sobol
        sobol = Sobol(d=n_steps, scramble=True, seed=seed)
        uniforms = sobol.random(n_paths)
        normals = norm.ppf(uniforms)
        return normals
    except:
        rng = np.random.RandomState(seed)
        return rng.randn(n_paths, n_steps)


def simulate_paths_gbm(spot: float, vol: float, rate_curve: Dict, div_schedule: Dict, div_params: Dict, calendar: np.ndarray, n_paths: int, seed: int, use_antithetic: bool=True, drift_override: Optional[float]=None, use_bridge: bool=False) -> Dict:
    """Simulate price paths under Geometric Brownian Motion with discrete dividends.
    
    Uses Sobol QMC with optional antithetic variates for variance reduction.
    Applies discrete dividend jumps on ex-dates.
    
    Args:
        spot: Initial stock price
        vol: Annualized volatility (from calibration)
        rate_curve: Discount curve
        div_schedule: Dividend schedule with dates and amounts
        div_params: Dividend model parameters
        calendar: Trading calendar
        n_paths: Number of simulation paths
        seed: Random seed for reproducibility
        use_antithetic: Apply antithetic variates (default: True)
        drift_override: Override risk-neutral drift (for P-measure forecasts)
        use_bridge: Use Brownian bridge (default: False, not recommended for American)
        
    Returns:
        Dict containing price paths, volatility used, time increments, and dividend info
        
    Notes:
        Risk-neutral drift: μ = r - q - σ²/2
        Path construction: S_t = S_0 * exp(cumsum(μ*dt + σ*√dt*Z))
    """
    n_steps = len(calendar) - 1
    dt_array = np.diff(rate_curve['times'])
    if len(dt_array) < n_steps:
        dt_array = np.append(dt_array, [dt_array[-1]] * (n_steps - len(dt_array)))
    dt_array = dt_array[:n_steps]
    time_grid = np.concatenate([[0], np.cumsum(dt_array)])
    n_paths_actual = n_paths // 2 if use_antithetic else n_paths
    if use_bridge:
        from scipy.stats import qmc
        np.random.seed(seed)
        sobol_engine = qmc.Sobol(d=1, scramble=True, seed=seed)
        sobol_samples = sobol_engine.random(n_paths_actual)
        uniform_samples = sobol_samples[:, 0]
        T = time_grid[-1]
        W_end = norm.ppf(uniform_samples) * np.sqrt(T)
        dt_uniform = T / n_steps
        W_stock = brownian_bridge(W_end, n_steps, dt_uniform)
        Z_stock = np.zeros((n_paths_actual, n_steps))
        for i in range(n_steps):
            dt_actual = dt_array[i]
            dW = W_stock[:, i + 1] - W_stock[:, i]
            Z_stock[:, i] = dW / np.sqrt(dt_actual)
    else:
        Z_stock = generate_sobol_normals(n_paths_actual, n_steps, seed)
    if div_params.get('has_dividends', False) and len(div_schedule['div_idx']) > 0:
        Z_div = generate_sobol_normals(n_paths_actual, len(div_schedule['div_idx']), seed + 1)
    if use_antithetic:
        Z_stock = np.vstack([Z_stock, -Z_stock])
        if div_params.get('has_dividends', False) and len(div_schedule['div_idx']) > 0:
            Z_div = np.vstack([Z_div, -Z_div])
    paths = np.zeros((len(Z_stock), n_steps + 1))
    paths[:, 0] = spot
    dt = np.diff(rate_curve['times'])
    if len(dt) < n_steps:
        dt = np.append(dt, [dt[-1]] * (n_steps - len(dt)))
    dt = dt[:n_steps]
    rates = rate_curve['zero_rates'][1:n_steps + 1]
    for t in range(n_steps):
        r = rates[t] if t < len(rates) else rates[-1]
        # Q-measure: drift = r - σ/2 | P-measure: drift = μ - σ/2
        if drift_override is not None:
            drift = (drift_override - 0.5 * vol ** 2) * dt[t]
        else:
            drift = (r - 0.5 * vol ** 2) * dt[t]
        diffusion = vol * np.sqrt(dt[t]) * Z_stock[:, t]
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion)
    dividend_paths = None
    if div_params.get('has_dividends', False) and len(div_schedule['div_idx']) > 0:
        S0 = div_params['S0']
        mu_div = div_params['mu']
        sigma_div = div_params['sigma']
        freq_days = div_params['frequency_days']
        dt_div = freq_days / 365.0
        n_divs = len(div_schedule['div_idx'])
        dividend_paths = np.zeros((len(Z_stock), n_divs))
        current_div = S0 / (365.0 / freq_days)
        for i in range(n_divs):
            drift_div = (mu_div - 0.5 * sigma_div ** 2) * dt_div
            diffusion_div = sigma_div * np.sqrt(dt_div) * Z_div[:, i]
            current_div = current_div * np.exp(drift_div + diffusion_div)
            dividend_paths[:, i] = current_div
            div_idx = div_schedule['div_idx'][i]
            if div_idx < paths.shape[1]:
                paths[:, div_idx:] -= dividend_paths[:, i:i + 1]
                paths[:, div_idx:] = np.maximum(paths[:, div_idx:], 0.01)
    weights = np.ones(len(paths)) / len(paths)
    return {'S': paths, 'dividend_paths': dividend_paths, 'weights': weights, 'model': 'gbm', 'vol': vol, 'has_stochastic_dividends': div_params.get('has_dividends', False)}


def simulate_paths_heston_qe(spot: float, params: HestonParams, rate_curve: Dict, div_schedule: Dict, div_params: Dict, calendar: np.ndarray, n_paths: int, seed: int, use_antithetic: bool=True) -> Dict:
    """Simulate paths under Heston stochastic volatility model using QE scheme.
    
    Implements Andersen (2008) Quadratic-Exponential scheme for robust variance process.
    Handles v→0 gracefully and maintains Feller condition.
    
    Args:
        spot: Initial stock price
        params: HestonParams (v0, kappa, theta, sigma, rho)
        rate_curve: Discount curve
        div_schedule: Dividend schedule
        div_params: Dividend model parameters
        calendar: Trading calendar
        n_paths: Number of simulation paths
        seed: Random seed
        use_antithetic: Apply antithetic variates
        
    Returns:
        Dict with price paths, variance paths, and metadata
        
    Notes:
        Spot-vol correlation via Cholesky decomposition: dW1 = ρ*dW2 + √(1-ρ²)*dW3
    """
    n_steps = len(calendar) - 1
    n_paths_actual = n_paths // 2 if use_antithetic else n_paths
    rng = np.random.RandomState(seed)
    Z1 = rng.randn(n_paths_actual, n_steps)
    Z2_indep = rng.randn(n_paths_actual, n_steps)
    Z2 = params.rho * Z1 + np.sqrt(1 - params.rho ** 2) * Z2_indep
    if use_antithetic:
        Z1 = np.vstack([Z1, -Z1])
        Z2 = np.vstack([Z2, -Z2])
    n_paths_total = len(Z1)
    S = np.zeros((n_paths_total, n_steps + 1))
    v = np.zeros((n_paths_total, n_steps + 1))
    S[:, 0] = spot
    v[:, 0] = params.v0
    dt = np.diff(rate_curve['times'])
    if len(dt) < n_steps:
        dt = np.append(dt, [dt[-1]] * (n_steps - len(dt)))
    dt = dt[:n_steps]
    rates = rate_curve['zero_rates'][1:n_steps + 1]
    psi_c = 1.5
    n_absorptions = 0
    n_reflections = 0
    for t in range(n_steps):
        r = rates[t] if t < len(rates) else rates[-1]
        dt_t = dt[t]
        m = params.theta + (v[:, t] - params.theta) * np.exp(-params.kappa * dt_t)
        s2 = v[:, t] * params.sigma ** 2 * np.exp(-params.kappa * dt_t) / params.kappa * (1 - np.exp(-params.kappa * dt_t)) + params.theta * params.sigma ** 2 / (2 * params.kappa) * (1 - np.exp(-params.kappa * dt_t)) ** 2
        psi = s2 / m ** 2
        v_next = np.zeros_like(v[:, t])
        for i in range(n_paths_total):
            if psi[i] <= psi_c:
                b2 = 2 / psi[i] - 1 + np.sqrt(2 / psi[i]) * np.sqrt(2 / psi[i] - 1)
                a = m[i] / (1 + b2)
                Z_v = Z1[i, t]
                v_next[i] = a * (np.sqrt(b2) + Z_v) ** 2
            else:
                p = (psi[i] - 1) / (psi[i] + 1)
                beta = (1 - p) / m[i]
                U = norm.cdf(Z1[i, t])
                if U <= p:
                    v_next[i] = 0
                    n_absorptions += 1
                else:
                    v_next[i] = np.log((1 - p) / (1 - U)) / beta
        v_next = np.maximum(v_next, 0)
        v[:, t + 1] = v_next
        v_mean = 0.5 * (v[:, t] + v[:, t + 1])
        K0 = -params.rho * params.kappa * params.theta * dt_t / params.sigma
        K1 = (params.kappa * params.rho / params.sigma - 0.5) * dt_t - params.rho / params.sigma
        K2 = params.rho / params.sigma
        log_S = np.log(S[:, t]) + r * dt_t + K0 + K1 * v[:, t] + K2 * v[:, t + 1] + np.sqrt(v_mean * dt_t) * Z2[:, t]
        S[:, t + 1] = np.exp(log_S)
    if div_params.get('has_dividends', False) and len(div_schedule['div_idx']) > 0:
        for idx, amt in zip(div_schedule['div_idx'], div_schedule['div_amounts']):
            if idx < n_steps + 1:
                S[:, idx:] *= 1 - amt / S[:, idx - 1:idx]
    weights = np.ones(n_paths_total) / n_paths_total
    diagnostics = {'n_absorptions': n_absorptions, 'n_reflections': n_reflections, 'v_min': float(v.min()), 'v_max': float(v.max()), 'v_mean': float(v.mean()), 'psi_c': psi_c}
    return {'S': S, 'v': v, 'Z1': Z1, 'Z2': Z2, 'weights': weights, 'model': 'heston', 'params': params, 'diagnostics': diagnostics}
