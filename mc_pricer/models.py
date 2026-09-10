"""Model calibration: GBM volatility and Heston stochastic-volatility parameters."""

import numpy as np
from datetime import datetime
from scipy.optimize import differential_evolution
from typing import Dict, NamedTuple, Optional, Tuple

from .calendar_curves import year_fraction
from .lsm import black_scholes_price


def calibrate_gbm_vol(surface: Dict, strike: float, expiry: datetime, spot: float) -> float:
    """Extract GBM volatility from IV surface for given strike and expiry."""
    if not surface or 'strikes' not in surface or 'vols' not in surface:
        print('  ⚠⚠ CRITICAL: No IV surface provided - cannot calibrate volatility')
        raise ValueError('Cannot calibrate volatility without IV surface data')
    strikes = np.array(surface['strikes'])
    vols = np.array(surface['vols'])
    if len(strikes) == 0:
        print('  ⚠⚠ CRITICAL: IV surface is empty - cannot calibrate volatility')
        raise ValueError('Cannot calibrate volatility with empty IV surface')
    exact_match = np.where(strikes == strike)[0]
    if len(exact_match) > 0:
        return vols[exact_match[0]]
    if len(strikes) > 1:
        sort_idx = np.argsort(strikes)
        strikes_sorted = strikes[sort_idx]
        vols_sorted = vols[sort_idx]
        vol = np.interp(strike, strikes_sorted, vols_sorted)
        return vol
    return vols[0]


class HestonParams(NamedTuple):
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float


def check_feller_condition(params: HestonParams) -> Tuple[bool, str]:
    """Verify Feller condition: 2*kappa*theta > sigma^2 (prevents variance from hitting zero)."""
    lhs = 2 * params.kappa * params.theta
    rhs = params.sigma ** 2
    satisfied = lhs > rhs
    if satisfied:
        msg = f'✓ Feller condition satisfied: 2κθ ({lhs:.4f}) > σ² ({rhs:.4f})'
    else:
        msg = f'⚠ Feller condition violated: 2κθ ({lhs:.4f}) ≤ σ² ({rhs:.4f}) - variance may go negative'
    return (satisfied, msg)


def price_european_heston_characteristic(spot: float, strike: float, T: float, rate: float, params: HestonParams) -> float:
    """Price European option under Heston model using characteristic function approach.
    
    Uses semi-analytical Heston formula (Heston 1993) via characteristic function.
    More accurate than MC for calibration purposes.
    
    Args:
        spot: Current spot price
        strike: Option strike
        T: Time to expiry in years
        rate: Risk-free rate
        params: HestonParams (v0, kappa, theta, sigma, rho)
        
    Returns:
        Call option price (use put-call parity for puts)
        
    Notes:
        Uses trapezoidal integration of characteristic function
    """
    from scipy.integrate import quad
    
    v0, kappa, theta, sigma, rho = params
    
    def characteristic_func(phi, j):
        """Heston characteristic function for P_j integral"""
        if j == 1:
            u, b = 0.5, kappa - rho * sigma
        else:
            u, b = -0.5, kappa
            
        a = kappa * theta
        x = np.log(spot)
        
        d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2*u*phi*1j - phi**2))
        g = (b - rho*sigma*phi*1j + d) / (b - rho*sigma*phi*1j - d)
        
        C = rate * phi * 1j * T + (a/sigma**2) * ((b - rho*sigma*phi*1j + d)*T - 2*np.log((1 - g*np.exp(d*T))/(1-g)))
        D = ((b - rho*sigma*phi*1j + d)/sigma**2) * ((1 - np.exp(d*T))/(1 - g*np.exp(d*T)))
        
        return np.exp(C + D*v0 + 1j*phi*x)
    
    def integrand(phi, j):
        """Integrand for P_j probability"""
        char = characteristic_func(phi, j)
        return np.real(np.exp(-1j*phi*np.log(strike)) * char / (1j*phi))
    
    # Compute P1 and P2 via integration
    P1 = 0.5 + (1/np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    P2 = 0.5 + (1/np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]
    
    # Heston call price
    call_price = spot * P1 - strike * np.exp(-rate*T) * P2
    
    return max(call_price, 0)  # Ensure non-negative


def heston_iv_from_price(price: float, spot: float, strike: float, T: float, rate: float, option_type: str) -> float:
    """Convert Heston price to implied volatility via inversion."""
    from scipy.optimize import brentq
    
    def objective(vol):
        bs_price = black_scholes_price(spot, strike, T, rate, vol, option_type)
        return bs_price - price
    
    try:
        iv = brentq(objective, 0.001, 5.0)
        return iv
    except:
        return np.nan


def calibrate_heston_params(iv_data: Dict, spot: float, rate_curve: Dict, expiry: datetime, as_of: datetime, initial_guess: Optional[HestonParams]=None, use_smile_calibration: bool=True) -> Tuple[HestonParams, Dict]:
    """Calibrate Heston stochastic volatility parameters to market IV surface.
    
    Includes proper smile calibration via pricing across all strikes.
    
    Uses two methods:
        - Moment matching (fast, approximate): Matches ATM vol, skew, and dispersion
        - Smile calibration (slow, precise): Prices at all strikes and minimizes IV error
        - User specifies if they want the slower more accurate model ran ? 
    
    Args:
        iv_data: Implied volatility surface data with strikes, IVs, moneyness
        spot: Current spot price
        rate_curve: Discount curve
        expiry: Option expiration
        as_of: Valuation date
        initial_guess: Optional starting parameters
        use_smile_calibration: If True, perform full smile fit (default: True)
        
    Returns:
        Tuple of (calibrated HestonParams, diagnostics dict with R², RMSE, residuals)
        
    Notes:
        Full smile calibration minimizes: Σ w_i * (σ_model(K_i) - σ_market(K_i))²
        Enforces Feller condition: 2*kappa*theta > sigma²
        Uses differential evolution for robust global optimization
    """
    strikes = iv_data['strikes']
    ivs = iv_data['ivs']
    weights = iv_data.get('weights', np.ones(len(strikes)))
    moneyness = iv_data['moneyness']
    
    if len(ivs) == 0:
        print("  ⚠⚠ CRITICAL: No IV data provided - cannot calibrate Heston parameters")
        raise ValueError("Cannot calibrate Heston model without IV surface data")
    
    # Step 1: Moment matching for initial guess
    v0 = np.average(ivs**2, weights=weights)
    theta = np.mean(ivs**2)
    iv_std = np.std(ivs)
    sigma = max(0.1, min(2.0, iv_std * 2))
    kappa = 2.0
    
    # Estimate rho from skew
    otm_puts = moneyness < 0.98
    otm_calls = moneyness > 1.02
    if otm_puts.sum() > 0 and otm_calls.sum() > 0:
        put_iv_avg = ivs[otm_puts].mean()
        call_iv_avg = ivs[otm_calls].mean()
        skew = put_iv_avg - call_iv_avg
        rho = -np.clip(skew / 0.1, 0.0, 0.9)
    else:
        print("  ⚠ WARNING: Insufficient OTM options for skew - setting rho=0")
        rho = 0.0
    
    if initial_guess is None:
        params = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    else:
        params = initial_guess
    
    # Ensure Feller condition
    feller_ok, feller_msg = check_feller_condition(params)
    if not feller_ok:
        required_lhs = params.sigma**2 * 1.1
        if params.theta < required_lhs / (2 * params.kappa):
            theta_adj = required_lhs / (2 * params.kappa)
            params = HestonParams(v0=params.v0, kappa=params.kappa, theta=theta_adj, 
                                 sigma=params.sigma, rho=params.rho)
            feller_ok, feller_msg = check_feller_condition(params)
    
    diagnostics = {
        'method': 'moment_matching',
        'n_strikes': len(strikes),
        'moneyness_range': (float(moneyness.min()), float(moneyness.max())),
        'iv_range': (float(ivs.min()), float(ivs.max())),
        'feller_satisfied': feller_ok,
        'feller_message': feller_msg,
        'params': {'v0': params.v0, 'kappa': params.kappa, 'theta': params.theta, 
                   'sigma': params.sigma, 'rho': params.rho}
    }
    
    # Step 2: Full smile calibration (if requested)
    if not use_smile_calibration:
        return params, diagnostics
    
    print("  → Running full smile calibration (pricing across strikes)...")
    
    T = year_fraction(as_of, expiry)
    rate = rate_curve['zero_rates'][-1]
    
    # Normalize weights (ATM gets more weight)
    atm_weight = 2.0
    weights_normalized = weights.copy()
    atm_idx = np.argmin(np.abs(moneyness - 1.0))
    weights_normalized[atm_idx] *= atm_weight
    weights_normalized = weights_normalized / weights_normalized.sum()
    
    def calibration_objective(params_array):
        """Objective: minimize sum of squared IV errors"""
        v0, kappa, theta, sigma, rho = params_array
        candidate = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        
        # Check Feller condition
        if 2 * kappa * theta <= sigma**2:
            return 1e10  # Penalty for violating Feller
        
        # Price at each strike and compute model IV
        total_error = 0.0
        for i, (K, market_iv) in enumerate(zip(strikes, ivs)):
            try:
                model_price = price_european_heston_characteristic(spot, K, T, rate, candidate)
                model_iv = heston_iv_from_price(model_price, spot, K, T, rate, 'call')
                
                if np.isnan(model_iv):
                    total_error += 100 * weights_normalized[i]  # Penalty
                else:
                    error = (model_iv - market_iv)**2
                    total_error += error * weights_normalized[i]
            except:
                total_error += 100 * weights_normalized[i]
        
        return total_error
    
    # Optimization bounds
    bounds = [
        (0.001, 1.0),      # v0
        (0.1, 10.0),       # kappa
        (0.001, 1.0),      # theta
        (0.01, 2.0),       # sigma
        (-0.99, -0.01)     # rho (negative for equity)
    ]
    
    # Use differential evolution for robust global optimization
    try:
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            calibration_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            atol=1e-4,
            tol=1e-4,
            disp=False
        )
        
        optimized_params = HestonParams(
            v0=result.x[0],
            kappa=result.x[1],
            theta=result.x[2],
            sigma=result.x[3],
            rho=result.x[4]
        )
        
        # Compute final diagnostics
        model_ivs = []
        residuals = []
        for K in strikes:
            try:
                model_price = price_european_heston_characteristic(spot, K, T, rate, optimized_params)
                model_iv = heston_iv_from_price(model_price, spot, K, T, rate, 'call')
                model_ivs.append(model_iv)
            except:
                model_ivs.append(np.nan)
        
        model_ivs = np.array(model_ivs)
        valid = ~np.isnan(model_ivs)
        
        if valid.sum() > 0:
            residuals = model_ivs[valid] - ivs[valid]
            rmse = np.sqrt(np.mean(residuals**2))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ivs[valid] - ivs[valid].mean())**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            rmse = np.nan
            r_squared = np.nan
            residuals = []
        
        diagnostics.update({
            'method': 'smile_calibration',
            'optimization_success': result.success,
            'optimization_iterations': result.nit,
            'final_objective': float(result.fun),
            'rmse': float(rmse) if not np.isnan(rmse) else None,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else None,
            'residuals': residuals.tolist() if len(residuals) > 0 else [],
            'model_ivs': model_ivs.tolist(),
            'market_ivs': ivs.tolist(),
            'strikes': strikes.tolist(),
            'params': {
                'v0': optimized_params.v0,
                'kappa': optimized_params.kappa,
                'theta': optimized_params.theta,
                'sigma': optimized_params.sigma,
                'rho': optimized_params.rho
            }
        })
        
        feller_ok, feller_msg = check_feller_condition(optimized_params)
        diagnostics['feller_satisfied'] = feller_ok
        diagnostics['feller_message'] = feller_msg
        
        print(f"  ✓ Smile calibration complete: R²={r_squared:.4f}, RMSE={rmse:.4f}")
        
        return optimized_params, diagnostics
        
    except Exception as e:
        print(f"  ⚠ Smile calibration failed ({str(e)}), using moment matching")
        diagnostics['calibration_error'] = str(e)
        return params, diagnostics


def report_heston_calibration(params: HestonParams, diagnostics: Dict) -> None:
    """Print Heston calibration results to console."""
    print(f"\n{'=' * 80}")
    print(f'HESTON MODEL CALIBRATION')
    print(f"{'=' * 80}")
    print(f"  Method:                {diagnostics['method']}")
    print(f"  Strikes used:          {diagnostics.get('n_strikes', 'N/A')}")
    if 'moneyness_range' in diagnostics:
        m_range = diagnostics['moneyness_range']
        print(f'  Moneyness range:       [{m_range[0]:.2f}, {m_range[1]:.2f}]')
    print(f'\n  Parameters:')
    print(f'    v₀ (initial var):    {params.v0:.6f}  ({np.sqrt(params.v0):.2%} IV)')
    print(f'    κ (mean reversion):  {params.kappa:.4f}')
    print(f'    θ (long-term var):   {params.theta:.6f}  ({np.sqrt(params.theta):.2%} IV)')
    print(f'    σ (vol-of-vol):      {params.sigma:.4f}')
    print(f'    ρ (correlation):     {params.rho:.4f}')
    print(f"\n  {diagnostics['feller_message']}")
    if not diagnostics['feller_satisfied']:
        print(f'    → Variance process may become negative')
        print(f'    → QE scheme will apply absorption/reflection')
    print(f"{'=' * 80}\n")
