"""Real-world (P-measure) P&L forecasting under a fixed exercise policy."""

import numpy as np
from typing import Dict

from .calendar_curves import build_dividend_schedule, build_rate_curve
from .simulation import simulate_paths_gbm
from .lsm import build_lsm_features, payoff_function, price_american_lsm


def convert_Q_to_P_drift(rate: float, expected_return: float, vol: float) -> float:
    """Convert risk-neutral drift (Q-measure) to real-world drift (P-measure)."""
    return expected_return


def compute_real_world_pnl(market: Dict, contract: Dict, calendar: np.ndarray, vol: float, premium_paid: float, div_params: Dict, n_paths: int, seed: int, expected_return: float=0.1, q_measure_policy: Dict=None) -> Dict:
    """Simulate real-world P&L distribution under P-measure with exercise policy reuse.
    
    V2.5: Now reuses Q-measure exercise policy instead of re-solving under P-measure.
    This reflects realistic behavior: traders exercise based on market prices (Q-measure)
    even when forecasting with different expected returns (P-measure).
    
    Args:
        market: Market snapshot
        contract: Option specification
        calendar: Trading calendar
        vol: Volatility
        premium_paid: Premium paid for P&L calculation
        div_params: Dividend parameters
        n_paths: Number of paths
        seed: Random seed
        expected_return: Expected stock return under P-measure (default: 10%)
        q_measure_policy: Pre-computed Q-measure exercise policy (if None, re-solves)
        
    Returns:
        Dict with P&L distribution, probabilities, percentiles, and paths
        
    Notes:
        If q_measure_policy provided: applies same exercise rule to P-measure paths
        If None: re-solves LSM under P-measure (old behavior, less realistic)
    """
    spot = market['spot']
    strike = contract['strike']
    option_type = contract['type']
    
    rate_curve = build_rate_curve(market['rates_raw'], calendar)
    div_schedule = build_dividend_schedule(market['dividends_raw'], calendar)
    
    # Simulate paths under P-measure (real-world drift)
    paths_P = simulate_paths_gbm(
        spot, vol, rate_curve, div_schedule, div_params, calendar, 
        n_paths, seed, use_antithetic=False, drift_override=expected_return
    )
    
    if q_measure_policy is not None:
        # NEW: Apply Q-measure exercise policy to P-measure paths
        # This is more realistic: traders don't re-optimize exercise in fantasy world
        print("  → Applying Q-measure exercise policy to P-measure paths...")
        
        S_P = paths_P['S']
        n_paths_P, n_steps = S_P.shape
        lsm_coefficients = q_measure_policy.get('lsm_coefficients', {})
        df = rate_curve['discount_factors']
        
        # Apply exercise decisions using Q-measure regression coefficients
        payoffs = np.zeros(n_paths_P)
        for path_idx in range(n_paths_P):
            exercised = False
            for t in range(n_steps - 1):
                if exercised:
                    break
                    
                immediate = payoff_function(S_P[path_idx, t], strike, option_type)
                if immediate <= 0:
                    continue
                
                # Use Q-measure regression to estimate continuation value
                if t in lsm_coefficients and lsm_coefficients[t][0] is not None:
                    coeffs, model_used, degree, *_ = lsm_coefficients[t]
                    
                    # Build features for this path at time t
                    state_single = {'S': S_P[path_idx:path_idx+1, :]}
                    X_single = build_lsm_features(state_single, t, model_used, degree)
                    
                    try:
                        continuation_estimate = float(X_single @ coeffs)
                        
                        # Exercise if immediate > continuation
                        if immediate > continuation_estimate:
                            payoffs[path_idx] = immediate * df[0] / df[t]
                            exercised = True
                    except:
                        # If regression fails, hold to maturity
                        pass
            
            # If never exercised, use terminal payoff
            if not exercised:
                terminal = payoff_function(S_P[path_idx, -1], strike, option_type)
                payoffs[path_idx] = terminal * df[0] / df[-1]
        
        pnl_distribution = payoffs - premium_paid
        terminal_payoffs = payoff_function(paths_P['S'][:, -1], strike, option_type)
        
    else:
        # OLD: Re-solve LSM under P-measure (less realistic)
        print("  → Re-solving LSM under P-measure (no policy provided)...")
        
        pricing_P = price_american_lsm(
            paths_P, contract, rate_curve, calendar, div_schedule
        )
        
        terminal_prices = paths_P['S'][:, -1]
        terminal_payoffs = payoff_function(terminal_prices, strike, option_type)
        pnl_distribution = pricing_P['option_values'] - premium_paid
    
    expected_pnl = np.mean(pnl_distribution)
    prob_profit = np.mean(pnl_distribution > 0)
    prob_worthless = np.mean(terminal_payoffs == 0)
    percentiles = np.percentile(pnl_distribution, [5, 25, 50, 75, 95])
    
    return {
        'pnl_distribution': pnl_distribution,
        'terminal_payoffs': terminal_payoffs,
        'expected_pnl': expected_pnl,
        'prob_profit': prob_profit,
        'prob_worthless': prob_worthless,
        'percentiles': {
            '5%': percentiles[0],
            '25%': percentiles[1],
            '50%': percentiles[2],
            '75%': percentiles[3],
            '95%': percentiles[4]
        },
        'paths': paths_P['S'],
        'policy_reused': q_measure_policy is not None
    }


def extract_exercise_policy(pricing_result: Dict, paths: Dict) -> Dict:
    """Extract optimal early exercise policy from LSM pricing results."""
    exercise_matrix = pricing_result['exercise_matrix']
    S = paths['S']
    n_steps = S.shape[1] - 1
    policy = []
    for t in range(n_steps):
        exercised_at_t = exercise_matrix[:, t]
        if exercised_at_t.sum() > 0:
            S_exercised = S[exercised_at_t, t]
            policy.append({'t': t, 'n_exercised': int(exercised_at_t.sum()), 'S_mean': float(S_exercised.mean()), 'S_std': float(S_exercised.std()), 'S_min': float(S_exercised.min()), 'S_max': float(S_exercised.max())})
    return {'policy_by_time': policy, 'total_exercised': int(exercise_matrix.any(axis=1).sum()), 'exercise_freq': float(exercise_matrix.any(axis=1).mean())}


def apply_exercise_policy(paths_P: Dict, policy: Dict, contract: Dict) -> np.ndarray:
    """Apply pre-computed exercise policy to new simulation paths."""
    S = paths_P['S']
    n_paths, n_steps = S.shape
    strike = contract['strike']
    option_type = contract['type']
    terminal_payoff = payoff_function(S[:, -1], strike, option_type)
    pnl = terminal_payoff.copy()
    for rule in policy['policy_by_time']:
        t = rule['t']
        S_min, S_max = (rule['S_min'], rule['S_max'])
        in_range = (S[:, t] >= S_min) & (S[:, t] <= S_max)
        exercise_value = payoff_function(S[in_range, t], strike, option_type)
        pnl[in_range] = exercise_value
    return pnl
