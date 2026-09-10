"""End-to-end pricing pipeline."""

import numpy as np
from datetime import datetime
from typing import Dict, List, Literal, Optional

from .config import N_PATHS_DEFAULT, RANDOM_SEED
from .calendar_curves import build_dividend_schedule, build_rate_curve, build_trading_calendar, create_market_snapshot, create_option_contract
from .models import calibrate_gbm_vol, calibrate_heston_params, report_heston_calibration
from .simulation import simulate_paths_gbm, simulate_paths_heston_qe
from .lsm import apply_control_variate, price_american_lsm
from .greeks import compute_greeks_t0
from .pnl import compute_real_world_pnl
from .validation import run_validation_checks
from .plots import plot_exercise_boundary, plot_option_value_over_time, plot_path_fan, plot_pnl_distribution, plot_pnl_risk_analysis
from .reports import generate_forecast_report, generate_pricing_report


def run_pricer(ticker: str, spot: float, strike: float, expiry: datetime, option_type: Literal['call', 'put'], as_of: datetime=None, implied_vol_surface: Dict=None, dividends_raw: List[Dict]=None, div_params: Dict=None, rates_raw: Dict=None, n_paths: int=N_PATHS_DEFAULT, premium_paid: Optional[float]=None, expected_return: float=0.1, compute_pnl: bool=True, show_plots: bool=True, save_plots: bool=False, plots_dir: str='.', model: str='gbm', use_market_calibration: bool=True) -> Dict:
    """Main pricing engine orchestrating complete workflow.
    
    Executes full pricing pipeline:
        1. Build calendar and rate curve
        2. Build dividend schedule
        3. Calibrate model parameters
        4. Simulate paths (GBM or Heston)
        5. Price via LSM
        6. Apply control variates
        7. Compute Greeks
        8. Run validation
        9. Compute P-measure forecasts
        10. Generate reports and plots
    
    Args:
        ticker: Stock symbol
        spot: Current spot price
        strike: Option strike
        expiry: Expiration date
        option_type: 'call' or 'put'
        as_of: Valuation date (default: today)
        implied_vol_surface: Pre-computed IV surface (fetches if None)
        dividends_raw: Dividend data (fetches if None)
        div_params: Dividend model parameters
        rates_raw: Rate curve data (fetches if None)
        n_paths: Number of simulation paths
        premium_paid: Market premium for P&L calc
        expected_return: Expected stock return for P-measure (default: 10%)
        compute_pnl: Compute real-world P&L forecasts
        show_plots: Display plots
        save_plots: Save plots to disk
        plots_dir: Directory for saved plots
        model: 'gbm' or 'heston'
        use_market_calibration: Calibrate to market IV surface
        
    Returns:
        Dict containing all pricing results, Greeks, validations, and forecasts
    """
    model_display = 'GBM' if model == 'gbm' else 'HESTON STOCHASTIC VOLATILITY'
    print('\n' + '=' * 80)
    print(f'MONTE CARLO AMERICAN OPTIONS PRICER - VERSION 2.0 ({model_display})')
    print('=' * 80)
    print(f'\nInitializing pricing run for {ticker} {option_type.upper()} option...')
    print(f'Model: {model.upper()} | Seed: {RANDOM_SEED} | Paths: {n_paths:,}')
    if as_of is None:
        as_of = datetime.now()
    if implied_vol_surface is None:
        implied_vol_surface = {'strikes': [strike], 'vols': [0.25]}
    if dividends_raw is None:
        dividends_raw = []
    if rates_raw is None:
        rates_raw = {'tenors': [0.0, 1.0, 2.0], 'rates': [0.045, 0.045, 0.044]}
    if premium_paid is None:
        premium_paid = max(spot - strike, 0) if option_type == 'call' else max(strike - spot, 0)
    print('\n[1/10] Building market snapshot...')
    market = create_market_snapshot(ticker=ticker, spot=spot, as_of=as_of, implied_vol_surface=implied_vol_surface, dividends_raw=dividends_raw, rates_raw=rates_raw, div_params=div_params)
    print('[2/10] Defining option contract...')
    contract = create_option_contract(option_type, strike, expiry, 'American')
    print('[3/10] Building trading calendar...')
    calendar = build_trading_calendar(as_of, expiry)
    print(f'  Trading days: {len(calendar)}')
    print('[4/10] Building rate curve...')
    rate_curve = build_rate_curve(rates_raw, calendar)
    print('[5/10] Building dividend schedule...')
    div_schedule = build_dividend_schedule(dividends_raw, calendar)
    print(f"  Dividends: {len(div_schedule['div_idx'])}")
    if len(div_schedule['div_idx']) > 0:
        for idx, amt in zip(div_schedule['div_idx'], div_schedule['div_amounts']):
            div_date = calendar[idx].astype('M8[D]').astype('O')
            print(f"    {div_date.strftime('%Y-%m-%d')}: ${amt:.2f}")
    print('[6/10] Calibrating model parameters...')
    vol = calibrate_gbm_vol(implied_vol_surface, strike, expiry, spot)
    print(f'  Volatility: {vol:.2%}')
    heston_params = None
    print('[7/10] Simulating price paths (GBM with dividends)...')
    if div_params is not None:
        div_params_for_sim = div_params
    elif isinstance(dividends_raw, dict) and 'has_dividends' in dividends_raw:
        div_params_for_sim = dividends_raw
    else:
        div_params_for_sim = {'has_dividends': len(dividends_raw) > 0, 'S0': dividends_raw[0]['amount'] if dividends_raw else 0.0, 'mu': 0.0, 'sigma': 0.0, 'frequency_days': 90, 'schedule': dividends_raw if isinstance(dividends_raw, list) else []}
    print(f'[7/10] Simulating price paths ({model.upper()})...')
    if model == 'gbm':
        paths = simulate_paths_gbm(spot=spot, vol=vol, rate_curve=rate_curve, div_schedule=div_schedule, div_params=div_params_for_sim, calendar=calendar, n_paths=n_paths, seed=RANDOM_SEED, use_antithetic=True)
    elif model == 'heston':
        print('  Calibrating Heston parameters from market IVs...')
        iv_data_simple = {'strikes': np.array(implied_vol_surface.get('strikes', [strike])), 'ivs': np.array(implied_vol_surface.get('vols', [vol])), 'weights': np.ones(len(implied_vol_surface.get('strikes', [strike]))), 'moneyness': np.array(implied_vol_surface.get('strikes', [strike])) / spot}
        heston_params, heston_diag = calibrate_heston_params(iv_data_simple, spot, rate_curve, expiry, as_of)
        report_heston_calibration(heston_params, heston_diag)
        paths = simulate_paths_heston_qe(spot=spot, params=heston_params, rate_curve=rate_curve, div_schedule=div_schedule, div_params=div_params_for_sim, calendar=calendar, n_paths=n_paths, seed=RANDOM_SEED, use_antithetic=True)
        h_diag = paths['diagnostics']
        print(f"  Variance range: [{h_diag['v_min']:.6f}, {h_diag['v_max']:.6f}]")
        print(f"  Variance absorptions: {h_diag['n_absorptions']}")
    else:
        raise ValueError(f'Unknown model: {model}')
    print(f"  Paths simulated: {paths['S'].shape[0]:,}")
    print('[8/10] Pricing American option (Longstaff-Schwartz)...')
    pricing_result = price_american_lsm(paths, contract, rate_curve, calendar, div_schedule, model=model)
    print('  Applying control variate correction (optimal beta)...')
    cv_result = apply_control_variate(american_price=pricing_result['price'], paths=paths, contract=contract, rate_curve=rate_curve, calendar=calendar, american_payoffs=pricing_result['option_values'])
    pricing_result['price_raw'] = pricing_result['price']
    pricing_result['price'] = cv_result['price_adjusted']
    pricing_result['cv_adjustment'] = cv_result['cv_adjustment']
    pricing_result['cv_beta'] = cv_result['beta']
    pricing_result['cv_beta_method'] = cv_result['beta_method']
    pricing_result['euro_bs'] = cv_result['euro_analytical']
    pricing_result['euro_mc'] = cv_result['euro_mc']
    print(f"  Raw American Price: ${pricing_result['price_raw']:.4f}")
    print(f"  CV Adjustment: ${cv_result['cv_adjustment']:.4f} (β={cv_result['beta']:.4f}, {cv_result['beta_method']})")
    if cv_result.get('correlation') is not None:
        print(f"  Correlation(Am,Eu): {cv_result['correlation']:.4f}")
        if cv_result.get('theoretical_var_reduction') is not None:
            print(f"  Theoretical VarRed: {cv_result['theoretical_var_reduction'] * 100:.1f}%")
    print(f"  CV-Adjusted Price: ${pricing_result['price']:.4f}")
    print(f"  95% CI: [${pricing_result['confidence_interval'][0]:.4f}, ${pricing_result['confidence_interval'][1]:.4f}]")
    if 'early_exercise_freq' in pricing_result:
        print(f"  Early Exercise Frequency: {pricing_result['early_exercise_freq']:.2%}")
    print('[9/10] Computing Greeks...')
    greeks_kwargs = {'market': market, 'contract': contract, 'calendar': calendar, 'rate_curve': rate_curve, 'div_schedule': div_schedule, 'vol': vol, 'n_paths': n_paths // 2, 'seed': RANDOM_SEED, 'div_params': div_params_for_sim, 'model': model}
    if model == 'heston':
        greeks_kwargs['heston_params'] = heston_params
        greeks_kwargs['implied_vol_surface'] = implied_vol_surface
    greeks = compute_greeks_t0(**greeks_kwargs)
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega:  {greeks['vega']:.4f}")
    print('[10/10] Running validation checks...')
    validations = run_validation_checks(market, contract, pricing_result, paths, calendar)
    all_pass = all((v.get('pass', True) for v in validations.values() if 'pass' in v))
    print(f"  Validation: {('PASS' if all_pass else 'FAIL')}")
    forecast_result = None
    if compute_pnl:
        print('\n[Bonus] Computing real-world P/L forecasts...')
        forecast_result = compute_real_world_pnl(market, contract, calendar, vol, premium_paid, n_paths=n_paths // 2, seed=RANDOM_SEED + 1, expected_return=expected_return, div_params=div_params_for_sim)
        print(f"  Expected P/L: ${forecast_result['expected_pnl']:.2f}")
        print(f"  Prob(Profit): {forecast_result['prob_profit']:.2%}")
    print('\n' + '=' * 80)
    report_kwargs = {'market': market, 'contract': contract, 'pricing_result': pricing_result, 'greeks': greeks, 'validations': validations, 'vol': vol, 'model': model}
    if model == 'heston':
        report_kwargs['heston_params'] = heston_params
        report_kwargs['heston_diagnostics'] = paths.get('diagnostics')
    print(generate_pricing_report(**report_kwargs))
    if forecast_result:
        print('\n')
        print(generate_forecast_report(forecast_result, premium_paid))
    if show_plots or save_plots:
        print('\nGenerating visualizations...')
        exp_date_str = expiry.strftime('%Y-%m-%d')
        option_label = option_type.capitalize()
        save_path_fan = f'{plots_dir}/{ticker}_path_fan.png' if save_plots else None
        save_path_boundary = f'{plots_dir}/{ticker}_exercise_boundary.png' if save_plots else None
        save_path_pnl = f'{plots_dir}/{ticker}_pnl_distribution.png' if save_plots else None
        save_path_risk = f'{plots_dir}/{ticker}_risk_analysis.png' if save_plots else None
        save_path_value_decay = f'{plots_dir}/{ticker}_value_decay.png' if save_plots else None
        
        plot_path_fan(calendar, paths['S'], div_schedule=div_schedule, strike=strike, n_paths_display=100, 
                     save_path=save_path_fan, ticker=ticker, expiration_date=exp_date_str, option_type=option_type)
        
        # Show exercise boundary chart for puts (always) or calls with dividends (early exercise can be optimal)
        show_exercise_chart = (option_type.lower() == 'put') or (option_type.lower() == 'call' and len(div_schedule['div_idx']) > 0)
        if show_exercise_chart:
            plot_exercise_boundary(calendar, pricing_result['exercise_boundary'], div_schedule, strike, 
                                  save_path=save_path_boundary, ticker=ticker, expiration_date=exp_date_str, option_type=option_type)
        
        # Plot discounted option value over time (Q-measure theta decay)
        plot_option_value_over_time(calendar, pricing_result['cashflows'], pricing_result['exercise_matrix'],
                                   rate_curve, strike, option_type, save_path=save_path_value_decay,
                                   ticker=ticker, expiration_date=exp_date_str)
        
        if forecast_result:
            plot_pnl_distribution(forecast_result['pnl_distribution'], premium_paid, 
                                 save_path=save_path_pnl, ticker=ticker, expiration_date=exp_date_str, option_type=option_type)
            plot_pnl_risk_analysis(forecast_result['pnl_distribution'], premium_paid, 
                                  save_path=save_path_risk, ticker=ticker, expiration_date=exp_date_str, option_type=option_type)
    print('\n' + '=' * 80)
    print('PRICING RUN COMPLETE')
    print('=' * 80)
    return {'pricing_Q': {'price': pricing_result['price'], 'confidence_interval': pricing_result['confidence_interval'], 'std_error': pricing_result['std_error'], 'exercise_boundary': pricing_result['exercise_boundary']}, 'greeks': greeks, 'forecast_P': forecast_result, 'validations': validations, 'metadata': {'ticker': ticker, 'spot': spot, 'strike': strike, 'expiry': expiry, 'option_type': option_type, 'as_of': as_of, 'vol': vol, 'n_paths': n_paths, 'seed': RANDOM_SEED}, 'paths': paths, 'calendar': calendar}
