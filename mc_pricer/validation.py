"""Validation checks, convergence sweeps, and Greek stability diagnostics."""

import numpy as np
import pandas as pd
from typing import Dict, List

from .lsm import payoff_function


def greek_stability_report(greeks_by_bump: Dict[str, Dict], threshold_cv: float=0.1) -> Dict:
    """Generate stability report for Greeks across different bump sizes."""
    diagnostics = {}
    greek_names = ['delta', 'gamma', 'vega', 'rho', 'theta']
    for greek in greek_names:
        values = [g.get(greek, 0) for g in greeks_by_bump.values() if greek in g]
        if len(values) > 1:
            mean = np.mean(values)
            std = np.std(values)
            cv = std / abs(mean) if abs(mean) > 1e-06 else np.inf
            stable = cv < threshold_cv
            diagnostics[greek] = {'values': values, 'mean': float(mean), 'std': float(std), 'cv': float(cv), 'stable': stable, 'warning': None if stable else f'⚠ {greek.upper()} CV={cv:.1%} exceeds {threshold_cv:.0%}'}
        else:
            diagnostics[greek] = {'values': values, 'stable': True, 'warning': None}
    return diagnostics


def convergence_sweep(pricing_func: callable, n_paths_list: List[int], seed: int=42) -> pd.DataFrame:
    """Test pricing convergence across different path counts."""
    results = []
    for n_paths in n_paths_list:
        result = pricing_func(n_paths=n_paths, seed=seed)
        results.append({'n_paths': n_paths, 'price': result['price'], 'std_error': result.get('std_error', 0), 'ci_lower': result.get('confidence_interval', [0, 0])[0], 'ci_upper': result.get('confidence_interval', [0, 0])[1]})
    df = pd.DataFrame(results)
    if len(df) > 1:
        df['price_change'] = df['price'].diff().abs()
        df['expected_improvement'] = 1 / np.sqrt(df['n_paths'])
    return df


def run_validation_checks(market: Dict, contract: Dict, pricing_result: Dict, paths: Dict, calendar: np.ndarray) -> Dict:
    """Run comprehensive validation checks on pricing results.
    
    Checks:
        - Intrinsic bound: Price >= max(S-K, 0)
        - Non-negative: Price >= 0
        - Upper bound: Price <= S (call) or K (put)
        - Convergence: Standard error CV < 5%
        - European equivalence: American ≈ European for no-dividend calls
    """
    validations = {}
    spot = market['spot']
    strike = contract['strike']
    option_type = contract['type']
    price = pricing_result['price']
    intrinsic = payoff_function(np.array([spot]), strike, option_type)[0]
    validations['intrinsic_bound'] = {'pass': price >= intrinsic, 'price': price, 'intrinsic': intrinsic, 'message': f'Price ({price:.4f}) >= Intrinsic ({intrinsic:.4f})'}
    validations['non_negative'] = {'pass': price >= 0, 'price': price, 'message': f'Price ({price:.4f}) >= 0'}
    if option_type == 'call':
        upper_bound = spot
        bound_check = price <= upper_bound
    else:
        upper_bound = strike
        bound_check = price <= upper_bound
    validations['upper_bound'] = {'pass': bound_check, 'price': price, 'bound': upper_bound, 'message': f'Price ({price:.4f}) <= Bound ({upper_bound:.4f})'}
    exercise_matrix = pricing_result['exercise_matrix']
    early_exercise_freq = np.mean(exercise_matrix[:, :-1])
    validations['early_exercise'] = {'frequency': early_exercise_freq, 'message': f'Early exercise frequency: {early_exercise_freq:.2%}'}
    std_error = pricing_result['std_error']
    cv = std_error / price if price > 0 else np.inf
    validations['convergence'] = {'std_error': std_error, 'coefficient_of_variation': cv, 'pass': cv < 0.05, 'message': f'CV: {cv:.2%} (target < 5%)'}
    has_divs = len(paths.get('div_schedule', {}).get('div_idx', [])) > 0
    if option_type == 'call' and (not has_divs):
        S_terminal = paths['S'][:, -1]
        terminal_payoffs = payoff_function(S_terminal, strike, option_type)
        rate_curve = paths.get('rate_curve', market.get('rates_raw', {}))
        if isinstance(rate_curve, dict) and 'discount_factors' in rate_curve:
            df_terminal = rate_curve['discount_factors'][-1]
        else:
            T = len(calendar) / 252
            df_terminal = np.exp(-0.045 * T)
        euro_price = np.mean(terminal_payoffs) * df_terminal
        pct_diff = abs(price - euro_price) / euro_price if euro_price > 0 else 0
        validations['european_equivalence'] = {'american_price': price, 'european_price': euro_price, 'pct_difference': pct_diff, 'pass': pct_diff < 0.02, 'message': f'American={price:.4f} vs European={euro_price:.4f} (diff: {pct_diff:.2%})'}
    return validations
