"""American exercise via Longstaff-Schwartz, with Black-Scholes control variates."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Literal

from .config import BASIS_DEGREE, CONFIDENCE_LEVEL


def build_lsm_features(state: Dict, t_idx: int, model: str, degree: int=3) -> np.ndarray:
    """Build Laguerre polynomial basis functions for LSM regression."""
    S = state['S'][:, t_idx]
    if model == 'gbm':
        x = S / S.mean()
        features = laguerre_basis(x, degree)
    elif model == 'heston':
        v = state['v'][:, t_idx]
        S_norm = S / S.mean()
        v_norm = v / v.mean()
        L_S = laguerre_basis(S_norm, degree)
        v_features = np.column_stack([v_norm ** i for i in range(1, degree + 1)])
        cross_terms = (S_norm * v_norm).reshape(-1, 1)
        features = np.column_stack([L_S, v_features, cross_terms])
    else:
        raise ValueError(f'Unknown model: {model}')
    return features


def payoff_function(S: np.ndarray, strike: float, option_type: Literal['call', 'put']) -> np.ndarray:
    """Calculate option payoff: max(S-K, 0) for calls, max(K-S, 0) for puts."""
    if option_type == 'call':
        return np.maximum(S - strike, 0)
    else:
        return np.maximum(strike - S, 0)


def laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Compute Laguerre polynomial basis functions for LSM regression.
    
    Orthogonal polynomials that provide stable regression conditioning.
    L0(x)=1, L1(x)=1-x, L2(x)=1-2x+x²/2, L3(x)=1-3x+3x²/2-x³/6
    
    Args:
        x: Normalized moneyness (S/K)
        degree: Polynomial degree (default: 3)
        
    Returns:
        Matrix of basis function values (n_paths, degree+1)
    """
    n = len(x)
    basis = np.ones((n, degree + 1))
    if degree >= 1:
        basis[:, 1] = 1 - x
    if degree >= 2:
        basis[:, 2] = 1 - 2 * x + x ** 2 / 2
    if degree >= 3:
        basis[:, 3] = 1 - 3 * x + 3 * x ** 2 / 2 - x ** 3 / 6
    return basis


def price_american_lsm(paths: Dict, contract: Dict, rate_curve: Dict, calendar: np.ndarray, div_schedule: Dict, basis_degree: int=BASIS_DEGREE, model: str=None) -> Dict:
    """Price American option using Longstaff-Schwartz Method (LSM).
    
    Core pricing engine implementing backward induction with regression.
    At each time step, regresses continuation value on Laguerre basis functions
    and compares to immediate exercise value.
    
    Args:
        paths: Simulated price paths from GBM or Heston
        contract: Option specification (type, strike, expiry, style)
        rate_curve: Discount curve
        calendar: Trading calendar
        div_schedule: Dividend schedule
        basis_degree: Laguerre polynomial degree (default: 3)
        model: Model name for feature engineering
        
    Returns:
        Dict containing:
            - price: Option fair value
            - price_stderr: Monte Carlo standard error
            - exercise_matrix: Boolean array of exercise decisions
            - cashflows: Realized cash flows per path
                        - regression_diagnostics: R², condition number, residual variance per time step
            - exercise_boundary: Early exercise boundary over time
            
    Notes:
        Algorithm:
            1. Initialize payoffs at expiry
            2. For t = T-1 down to 0:
                a. Find ITM paths
                b. Regress discounted continuation on basis functions
                c. Compare immediate vs continuation
                d. Update exercise decisions and cash flows
            3. Discount all cash flows to t=0
    """
    S = paths['S']
    n_paths, n_steps = S.shape
    strike = contract['strike']
    option_type = contract['type']
    if model is None:
        model = paths.get('model', 'gbm')
    has_dividends = div_schedule.get('has_dividends', False)
    allow_early_exercise = True
    if option_type == 'call' and (not has_dividends):
        allow_early_exercise = False
    df = rate_curve['discount_factors']
    lsm_coefficients = {}
    cashflows = np.zeros_like(S)
    cashflows[:, -1] = payoff_function(S[:, -1], strike, option_type)
    exercise_matrix = np.zeros_like(S, dtype=bool)
    exercise_matrix[:, -1] = cashflows[:, -1] > 0
    for t in range(n_steps - 2, -1, -1):
        immediate_exercise = payoff_function(S[:, t], strike, option_type)
        if not allow_early_exercise:
            cashflows[:, t] = cashflows[:, t + 1] * (df[t + 1] / df[t])
            continue
        itm = immediate_exercise > 0
        if np.sum(itm) == 0:
            cashflows[:, t] = cashflows[:, t + 1] * (df[t + 1] / df[t])
            continue
        state_itm = {'S': S[itm, :]}
        if 'v' in paths and model == 'heston':
            state_itm['v'] = paths['v'][itm, :]
        X = build_lsm_features(state_itm, t, model, basis_degree)
        continuation_value = cashflows[itm, t + 1] * (df[t + 1] / df[t])
        try:
            coeffs, residuals_lstsq, rank, s = np.linalg.lstsq(X, continuation_value, rcond=None)
            continuation_estimate = X @ coeffs
            
            # Regression diagnostics logging
            try:
                # Compute R² for regression quality
                residuals = continuation_value - continuation_estimate
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((continuation_value - continuation_value.mean())**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                residual_std = np.std(residuals) if len(residuals) > 0 else 0.0
                
                # Condition number from singular values (safer than cond())
                if len(s) > 0 and s[0] > 0 and s[-1] > 0:
                    cond_number = float(s[0] / s[-1])
                else:
                    cond_number = np.inf
                
                regression_diagnostics = {
                    'r_squared': float(r_squared),
                    'condition_number': float(cond_number),
                    'residual_std': float(residual_std),
                    'n_itm': int(np.sum(itm)),
                    'rank': int(rank),
                    'stable': cond_number < 1000 and r_squared > 0.5
                }
            except:
                # If diagnostics fail, use safe defaults
                regression_diagnostics = {
                    'r_squared': 0.0,
                    'condition_number': np.inf,
                    'residual_std': 0.0,
                    'n_itm': int(np.sum(itm)),
                    'rank': 0,
                    'stable': False
                }
            
            lsm_coefficients[t] = (coeffs, model, basis_degree, regression_diagnostics)

        except Exception as e:
            continuation_estimate = np.full(len(S[itm, t]), np.mean(continuation_value))
            regression_diagnostics = {
                'r_squared': 0.0,
                'condition_number': np.inf,
                'residual_std': 0.0,
                'n_itm': int(np.sum(itm)),
                'rank': 0,
                'stable': False,
                'error': str(e)
            }
            lsm_coefficients[t] = (None, model, basis_degree, regression_diagnostics)
        exercise_now = immediate_exercise[itm] > continuation_estimate
        itm_indices = np.where(itm)[0]
        for i, idx in enumerate(itm_indices):
            if exercise_now[i]:
                cashflows[idx, t] = immediate_exercise[idx]
                exercise_matrix[idx, t] = True
                cashflows[idx, t + 1:] = 0
                exercise_matrix[idx, t + 1:] = False
            else:
                cashflows[idx, t] = cashflows[idx, t + 1] * (df[t + 1] / df[t])
    option_values = np.zeros(n_paths)
    if not allow_early_exercise:
        option_values = cashflows[:, 0]
    else:
        for path_idx in range(n_paths):
            exercise_times = np.where(exercise_matrix[path_idx, :])[0]
            if len(exercise_times) > 0:
                t_ex = exercise_times[0]
                option_values[path_idx] = cashflows[path_idx, t_ex] * df[0] / df[t_ex]
            else:
                option_values[path_idx] = 0
    price = np.mean(option_values)
    std_error = np.std(option_values) / np.sqrt(n_paths)
    z_score = norm.ppf(0.5 + CONFIDENCE_LEVEL / 2)
    confidence_interval = (price - z_score * std_error, price + z_score * std_error)
    exercise_boundary = np.zeros(n_steps)
    exercise_boundary_stats = {'S_min': np.zeros(n_steps), 'S_max': np.zeros(n_steps), 'n_exercised': np.zeros(n_steps, dtype=int)}
    early_exercise_count = 0
    for t in range(n_steps):
        exercised_at_t = exercise_matrix[:, t]
        if np.any(exercised_at_t):
            exercise_boundary_stats['S_min'][t] = np.min(S[exercised_at_t, t])
            exercise_boundary_stats['S_max'][t] = np.max(S[exercised_at_t, t])
            exercise_boundary_stats['n_exercised'][t] = np.sum(exercised_at_t)
            if t in lsm_coefficients and lsm_coefficients[t][0] is not None:
                coeffs, model_used, degree, *_ = lsm_coefficients[t]
                try:
                    from scipy.optimize import brentq

                    def objective(S_star):
                        immediate = payoff_function(S_star, strike, option_type)
                        state_star = {'S': np.array([[S_star]])}
                        if model_used == 'heston' and 'v' in paths:
                            v_median = np.median(paths['v'][:, t])
                            state_star['v'] = np.array([[v_median]])
                        X_star = build_lsm_features(state_star, t, model_used, degree)
                        continuation = float(X_star @ coeffs)
                        return immediate - continuation
                    S_min_search = max(exercise_boundary_stats['S_min'][t] * 0.8, strike * 0.5 if option_type == 'put' else strike * 0.8)
                    S_max_search = min(exercise_boundary_stats['S_max'][t] * 1.2, strike * 1.5 if option_type == 'put' else strike * 2.0)
                    if objective(S_min_search) * objective(S_max_search) < 0:
                        exercise_boundary[t] = brentq(objective, S_min_search, S_max_search)
                    else:
                        exercise_boundary[t] = np.median(S[exercised_at_t, t])
                except:
                    exercise_boundary[t] = np.median(S[exercised_at_t, t])
            else:
                exercise_boundary[t] = np.median(S[exercised_at_t, t])
            if t < n_steps - 1:
                early_exercise_count += np.sum(exercised_at_t)
        else:
            exercise_boundary[t] = np.nan
            exercise_boundary_stats['S_min'][t] = np.nan
            exercise_boundary_stats['S_max'][t] = np.nan
    early_exercise_freq = early_exercise_count / (n_paths * (n_steps - 1)) if allow_early_exercise else 0.0
    return {'price': price, 'std_error': std_error, 'confidence_interval': confidence_interval, 'exercise_matrix': exercise_matrix, 'exercise_boundary': exercise_boundary, 'exercise_boundary_stats': exercise_boundary_stats, 'lsm_coefficients': lsm_coefficients,
        'regression_diagnostics': {t: lsm_coefficients[t][3] if len(lsm_coefficients[t]) > 3 else {} for t in lsm_coefficients}, 'cashflows': cashflows, 'option_values': option_values, 'early_exercise_freq': early_exercise_freq, 'early_exercise_allowed': allow_early_exercise}


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal['call', 'put']) -> float:
    """Calculate Black-Scholes European option price (closed-form solution)."""
    if T <= 0:
        return payoff_function(np.array([S]), K, option_type)[0]
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def apply_control_variate(american_price: float, paths: Dict, contract: Dict, rate_curve: Dict, calendar: np.ndarray, american_payoffs: np.ndarray=None) -> float:
    """Apply control variate variance reduction using European option as control.
    
    Estimates optimal beta: β* = Cov(V_am, V_eu) / Var(V_eu)
    Adjusts price: V_adjusted = V_am + β*(BS_analytical - BS_mc)
    
    Args:
        american_price: Raw American option price from LSM
        paths: Simulated price paths
        contract: Option specification
        rate_curve: Discount curve
        calendar: Trading calendar
        american_payoffs: Optional pre-computed American payoffs
        
    Returns:
        Control variate adjusted price
        
    Notes:
        Optimal beta can achieve 90-99% variance reduction when American ≈ European
    """
    S = paths['S']
    spot = S[0, 0]
    vol = paths['vol']
    strike = contract['strike']
    option_type = contract['type']
    T = rate_curve['times'][-1]
    r = rate_curve['zero_rates'][-1]
    df = rate_curve['discount_factors'][-1]
    euro_analytical = black_scholes_price(spot, strike, T, r, vol, option_type)
    terminal_payoffs = payoff_function(S[:, -1], strike, option_type)
    euro_mc_paths = terminal_payoffs * df
    euro_mc = np.mean(euro_mc_paths)
    if american_payoffs is not None and len(american_payoffs) == len(euro_mc_paths):
        cov = np.cov(american_payoffs, euro_mc_paths)[0, 1]
        var_euro = np.var(euro_mc_paths)
        if var_euro > 1e-10:
            beta_optimal = cov / var_euro
            beta = np.clip(beta_optimal, 0.0, 2.0)
            beta_method = 'optimal'
        else:
            beta = 1.0
            beta_method = 'fixed (zero variance)'
    else:
        beta = 1.0
        beta_method = 'fixed (no payoffs)'
    cv_error = euro_analytical - euro_mc
    cv_adjustment = beta * cv_error
    price_adjusted = american_price + cv_adjustment
    if american_payoffs is not None and len(american_payoffs) == len(euro_mc_paths):
        correlation = np.corrcoef(american_payoffs, euro_mc_paths)[0, 1]
        theoretical_var_reduction = correlation ** 2
    else:
        theoretical_var_reduction = None
    return {'price_adjusted': price_adjusted, 'cv_adjustment': cv_adjustment, 'euro_analytical': euro_analytical, 'euro_mc': euro_mc, 'beta': beta, 'beta_method': beta_method, 'cv_error': cv_error, 'correlation': correlation if american_payoffs is not None else None, 'theoretical_var_reduction': theoretical_var_reduction}
