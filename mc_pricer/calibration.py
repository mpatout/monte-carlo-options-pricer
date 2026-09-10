"""Calibrate a model to the observed market option chain."""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import differential_evolution, minimize
from typing import Dict

from .calendar_curves import build_dividend_schedule, build_rate_curve, build_trading_calendar, create_option_contract
from .models import HestonParams
from .simulation import simulate_paths_gbm, simulate_paths_heston_qe
from .lsm import black_scholes_price, price_american_lsm


def calibrate_model_to_market(ticker: str, spot: float, as_of: datetime, target_expiry: datetime, option_chain: pd.DataFrame, model: str='gbm', n_paths: int=10000, seed: int=42) -> Dict:
    """Calibrate model parameters to match market option prices.
    
    Iteratively adjusts model parameters to minimize pricing errors vs market.
    """
    print(f"\n{'=' * 80}")
    print('MARKET-CONSISTENT CALIBRATION')
    print(f"{'=' * 80}")
    print(f"Model: {model.upper()} | Target Expiry: {target_expiry.strftime('%Y-%m-%d')}")
    chain = option_chain[(option_chain['volume'] > 0) & (option_chain['bid'] > 0) & (option_chain['ask'] > 0)].copy()
    if len(chain) == 0:
        raise ValueError('No liquid options available for calibration')
    chain['mid_price'] = (chain['bid'] + chain['ask']) / 2.0
    calls = chain[chain['optionType'] == 'call'].sort_values('strike')
    puts = chain[chain['optionType'] == 'put'].sort_values('strike')
    print(f'\nLiquid options: {len(calls)} calls, {len(puts)} puts')
    atm_strike = min(chain['strike'], key=lambda x: abs(x - spot))
    strikes_list = sorted(chain['strike'].unique())
    atm_idx = strikes_list.index(atm_strike)
    start_idx = max(0, atm_idx - 2)
    end_idx = min(len(strikes_list), atm_idx + 3)
    calibration_strikes = strikes_list[start_idx:end_idx]
    targets = []
    for strike in calibration_strikes:
        call_row = calls[calls['strike'] == strike]
        put_row = puts[puts['strike'] == strike]
        if not call_row.empty:
            targets.append({'strike': strike, 'option_type': 'call', 'market_price': call_row.iloc[0]['mid_price']})
        if not put_row.empty:
            targets.append({'strike': strike, 'option_type': 'put', 'market_price': put_row.iloc[0]['mid_price']})
    if len(targets) == 0:
        raise ValueError('No calibration targets available')
    print(f'Calibration targets: {len(targets)} options across {len(calibration_strikes)} strikes')
    print(f"Strikes: {[f'${s:.0f}' for s in calibration_strikes]}")
    from datetime import datetime as dt
    calendar = build_trading_calendar(as_of, target_expiry)
    T = (target_expiry - as_of).days / 365.0
    rates_raw = {'tenors': [0.0, T, T + 1], 'rates': [0.045, 0.045, 0.044]}
    rate_curve = build_rate_curve(rates_raw, calendar)
    div_schedule = build_dividend_schedule([], calendar)
    div_params = {'has_dividends': False}
    atm_strike = min(calibration_strikes, key=lambda x: abs(x - spot))
    atm_targets = [t for t in targets if t['strike'] == atm_strike]
    if atm_targets and model == 'gbm':
        atm_market_price = atm_targets[0]['market_price']
        atm_type = atm_targets[0]['option_type']
        try:
            from scipy.optimize import brentq

            def bs_error(sigma):
                return black_scholes_price(spot, atm_strike, T, 0.045, sigma, atm_type) - atm_market_price
            initial_vol = brentq(bs_error, 0.01, 2.0)
        except:
            initial_vol = 0.25
    else:
        initial_vol = 0.25
    path_cache = {}

    def get_cached_paths(vol_key, spot_val, vol_val):
        if vol_key not in path_cache:
            path_cache[vol_key] = simulate_paths_gbm(spot_val, vol_val, rate_curve, div_schedule, div_params, calendar, n_paths, seed, use_antithetic=True)
        return path_cache[vol_key]

    def objective_gbm(params):
        vol = params[0]
        if vol < 0.01 or vol > 3.0:
            return 10000000000.0
        vol_key = f'{vol:.6f}'
        paths = get_cached_paths(vol_key, spot, vol)
        errors = []
        for target in targets:
            contract = create_option_contract(target['option_type'], target['strike'], target_expiry, 'American')
            result = price_american_lsm(paths, contract, rate_curve, calendar, div_schedule, model='gbm')
            model_price = result['price']
            weight = 1.0 / max(target['market_price'], 1.0)
            error = weight * (model_price - target['market_price']) ** 2
            errors.append(error)
        sse = np.sum(errors)
        return sse

    def objective_heston(params):
        v0, kappa, theta, sigma_v, rho = params
        if 2 * kappa * theta < sigma_v ** 2:
            return 10000000000.0
        if v0 < 0.0001 or v0 > 1.0:
            return 10000000000.0
        if kappa < 0.1 or kappa > 10.0:
            return 10000000000.0
        if theta < 0.0001 or theta > 1.0:
            return 10000000000.0
        if sigma_v < 0.01 or sigma_v > 2.0:
            return 10000000000.0
        if rho < -0.99 or rho > 0.99:
            return 10000000000.0
        heston_params = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=sigma_v, rho=rho)
        errors = []
        for target in targets:
            contract = create_option_contract(target['option_type'], target['strike'], target_expiry, 'American')
            paths = simulate_paths_heston_qe(spot, heston_params, rate_curve, div_schedule, div_params, calendar, n_paths, seed, use_antithetic=True)
            result = price_american_lsm(paths, contract, rate_curve, calendar, div_schedule, model='heston')
            model_price = result['price']
            error = (model_price - target['market_price']) ** 2
            errors.append(error)
        sse = np.sum(errors)
        return sse
    print('\nRunning optimization...')
    if model == 'gbm':
        print(f'Initial guess: vol={initial_vol:.2%} (estimated from ATM market price)')
        result = minimize(objective_gbm, x0=[initial_vol], method='L-BFGS-B', bounds=[(0.05, 2.0)], options={'ftol': 1e-06, 'maxiter': 50})
        optimal_vol = result.x[0]
        calibrated_params = {'vol': optimal_vol}
        print(f'Optimal volatility: {optimal_vol:.2%}')
    else:
        initial_v0 = 0.04
        initial_kappa = 2.0
        initial_theta = 0.04
        initial_sigma = 0.3
        initial_rho = -0.5
        print(f'Initial guess: v0={initial_v0:.4f}, kappa={initial_kappa:.2f}, theta={initial_theta:.4f}, sigma={initial_sigma:.2f}, rho={initial_rho:.2f}')
        bounds = [(0.001, 0.5), (0.1, 10.0), (0.001, 0.5), (0.05, 2.0), (-0.95, 0.95)]
        result = differential_evolution(objective_heston, bounds=bounds, seed=seed, maxiter=50, atol=0.01, tol=0.01, workers=1)
        v0_opt, kappa_opt, theta_opt, sigma_opt, rho_opt = result.x
        calibrated_params = {'v0': v0_opt, 'kappa': kappa_opt, 'theta': theta_opt, 'sigma': sigma_opt, 'rho': rho_opt}
        print(f'Optimal: v0={v0_opt:.4f}, kappa={kappa_opt:.2f}, theta={theta_opt:.4f}, sigma={sigma_opt:.2f}, rho={rho_opt:.2f}')
    print('\nComputing residual errors...')
    model_prices = []
    market_prices = []
    residual_errors = []
    relative_errors = []
    strikes_used = []
    for target in targets:
        contract = create_option_contract(target['option_type'], target['strike'], target_expiry, 'American')
        if model == 'gbm':
            paths = simulate_paths_gbm(spot, calibrated_params['vol'], rate_curve, div_schedule, div_params, calendar, n_paths, seed, use_antithetic=True)
            result = price_american_lsm(paths, contract, rate_curve, calendar, div_schedule, model='gbm')
        else:
            heston_params = HestonParams(v0=calibrated_params['v0'], kappa=calibrated_params['kappa'], theta=calibrated_params['theta'], sigma=calibrated_params['sigma'], rho=calibrated_params['rho'])
            paths = simulate_paths_heston_qe(spot, heston_params, rate_curve, div_schedule, div_params, calendar, n_paths, seed, use_antithetic=True)
            result = price_american_lsm(paths, contract, rate_curve, calendar, div_schedule, model='heston')
        model_price = result['price']
        market_price = target['market_price']
        residual = model_price - market_price
        relative = residual / market_price * 100 if market_price > 0 else 0
        model_prices.append(model_price)
        market_prices.append(market_price)
        residual_errors.append(residual)
        relative_errors.append(relative)
        strikes_used.append(target['strike'])
    rmse = np.sqrt(np.mean(np.array(residual_errors) ** 2))
    mae = np.mean(np.abs(residual_errors))
    max_error = np.max(np.abs(residual_errors))
    mape = np.mean(np.abs(relative_errors))
    print(f"\n{'=' * 80}")
    print('CALIBRATION RESULTS')
    print(f"{'=' * 80}")
    print(f'Root Mean Squared Error (RMSE): ${rmse:.4f}')
    print(f'Mean Absolute Error (MAE): ${mae:.4f}')
    print(f'Max Absolute Error: ${max_error:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    print(f'\nPer-Strike Results:')
    print(f"{'Strike':>8} {'Type':>6} {'Market':>10} {'Model':>10} {'Error':>10} {'Error %':>10}")
    print('-' * 66)
    for i, target in enumerate(targets):
        print(f"${strikes_used[i]:>7.0f} {target['option_type']:>6} ${market_prices[i]:>9.2f} ${model_prices[i]:>9.2f} ${residual_errors[i]:>9.2f} {relative_errors[i]:>9.1f}%")
    return {'calibrated_params': calibrated_params, 'residual_errors': residual_errors, 'relative_errors': relative_errors, 'rmse': rmse, 'mae': mae, 'max_error': max_error, 'mape': mape, 'strikes_used': strikes_used, 'market_prices': market_prices, 'model_prices': model_prices, 'targets': targets}
