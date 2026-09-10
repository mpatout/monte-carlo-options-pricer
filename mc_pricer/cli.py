"""Interactive command-line entry point."""

import numpy as np
from datetime import datetime

from .config import N_PATHS_DEFAULT, RANDOM_SEED
from .market_data import fetch_all_market_data, fetch_option_chain, fetch_ticker_data
from .calendar_curves import build_dividend_schedule, build_rate_curve, create_option_contract
from .calibration import calibrate_model_to_market
from .greeks import test_greek_bump_sensitivity
from .pricer import run_pricer


def price_option(ticker_and_type: str=None):
    """Interactive CLI for pricing options with user prompts.
    
    Guides user through:
        1. Ticker and option type selection
        2. Expiration date selection
        3. Strike price selection
        4. Path count specification
        5. Optional features (calibration, bump sensitivity)
        6. Displays results and visualizations
    """
    print('\n' + '=' * 80)
    print('MONTE CARLO AMERICAN OPTIONS PRICER')
    print('=' * 80)
    if ticker_and_type is None:
        ticker_input = input("\nEnter Ticker and P/C (e.g., 'AAPL C' or 'SPY P'): ").strip().upper()
    else:
        ticker_input = ticker_and_type.strip().upper()
        print(f'\nInput: {ticker_input}')
    parts = ticker_input.split()
    if len(parts) != 2:
        print("Error: Format must be 'TICKER P/C' (e.g., 'AAPL C' or 'SPY P')")
        return
    ticker = parts[0]
    option_code = parts[1]
    if option_code not in ['P', 'C', 'PUT', 'CALL']:
        print("Error: Must specify 'P' for put or 'C' for call")
        return
    option_type = 'put' if option_code in ['P', 'PUT'] else 'call'
    print(f'\nTicker: {ticker}')
    print(f'Option Type: {option_type.upper()}')
    print('\nFetching available expirations...')
    try:
        ticker_data = fetch_ticker_data(ticker)
        spot = ticker_data['spot']
        expirations = ticker_data['expirations']
        print(f"\n{'=' * 80}")
        print(f'SPOT PRICE: ${spot:.2f}')
        print(f"{'=' * 80}")
        print(f'\nAvailable Expirations for {ticker}:')
        print(f"{'#':>3} | {'Expiration':>12} | {'Days Out':>10}")
        print('-' * 35)
        today = datetime.now()
        for i, exp in enumerate(expirations[:20], 1):
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            days_out = (exp_date - today).days
            print(f'{i:>3} | {exp:>12} | {days_out:>10}')
        if len(expirations) > 20:
            print(f'\n... and {len(expirations) - 20} more expirations')
    except Exception as e:
        print(f'Error: {e}')
        return
    print(f"\n{'=' * 80}")
    exp_choice = input('Choose expiration (enter date YYYY-MM-DD or number): ').strip()
    if exp_choice.isdigit():
        exp_idx = int(exp_choice) - 1
        if 0 <= exp_idx < len(expirations):
            expiry_str = expirations[exp_idx]
        else:
            print('Error: Invalid expiration number')
            return
    elif exp_choice in expirations:
        expiry_str = exp_choice
    else:
        print('Error: Invalid expiration date')
        return
    print(f'Selected: {expiry_str}')
    print(f'\nFetching option chain for {expiry_str}...')
    try:
        calls, puts = fetch_option_chain(ticker, expiry_str)
        df = calls if option_type == 'call' else puts
        df_filtered = df[df['volume'] > 0].copy() if 'volume' in df.columns else df.copy()
        if len(df_filtered) == 0:
            df_filtered = df.copy()
        strikes = sorted(df_filtered['strike'].unique())
        print(f"\n{'=' * 80}")
        print(f'Available Strikes (Total: {len(strikes)})')
        print(f"{'=' * 80}")
        print(f"{'#':>4} | {'Strike':>10} | {'Moneyness':>12} | {'Last Price':>12} | {'Volume':>10}")
        print('-' * 70)
        atm_idx = np.argmin(np.abs(np.array(strikes) - spot))
        start_idx = max(0, atm_idx - 10)
        end_idx = min(len(strikes), atm_idx + 11)
        for i in range(start_idx, end_idx):
            strike_val = strikes[i]
            moneyness = strike_val / spot
            strike_data = df_filtered[df_filtered['strike'] == strike_val].iloc[0]
            last_price = strike_data.get('lastPrice', 0.0)
            volume = strike_data.get('volume', 0)
            marker = ' ← ATM' if abs(moneyness - 1.0) < 0.02 else ''
            print(f'{i + 1:>4} | ${strike_val:>9.2f} | {moneyness:>11.1%} | ${last_price:>11.2f} | {volume:>10.0f}{marker}')
        if end_idx < len(strikes):
            print(f'\n... {len(strikes) - end_idx} more strikes available')
    except Exception as e:
        print(f'Error fetching strikes: {e}')
        return
    print(f"\n{'=' * 80}")
    strike_input = input(f'Enter strike price (current spot: ${spot:.2f}): ').strip()
    try:
        strike = float(strike_input)
    except:
        print('Error: Invalid strike price')
        return
    n_paths_input = input(f'Number of paths [default {N_PATHS_DEFAULT:,}]: ').strip()
    n_paths = int(n_paths_input) if n_paths_input else N_PATHS_DEFAULT
    print(f"\n{'=' * 80}")
    print('ADVANCED FEATURES (optional)')
    print(f"{'=' * 80}")
    run_calibration = input('Run market-consistent calibration? (y/N): ').strip().lower() == 'y'
    run_bump_test = input('Run Greeks bump sensitivity test? (y/N): ').strip().lower() == 'y'
    print(f"\n{'=' * 80}")
    print('STARTING PRICING ENGINE...')
    print(f"{'=' * 80}")
    print(f'Ticker: {ticker}')
    print(f'Type: {option_type.upper()}')
    print(f'Strike: ${strike:.2f}')
    print(f'Expiry: {expiry_str}')
    print(f'Paths: {n_paths:,}')
    print(f"{'=' * 80}\n")
    try:
        market_data = fetch_all_market_data(ticker, expiry_str, option_type)
        premium_paid = None
        try:
            df['strike_diff'] = abs(df['strike'] - strike)
            closest = df.loc[df['strike_diff'].idxmin()]
            premium_paid = float(closest['lastPrice'])
            print(f'Market Premium: ${premium_paid:.2f}\n')
        except:
            premium_paid = max(spot - strike, 0) if option_type == 'call' else max(strike - spot, 0)
        results = run_pricer(ticker=ticker, spot=market_data['spot'], strike=strike, expiry=market_data['expiry'], option_type=option_type, as_of=market_data['as_of'], implied_vol_surface=market_data['vol_surface'], dividends_raw=market_data['dividends'], div_params=market_data['div_params'], rates_raw=market_data['rates'], n_paths=n_paths, premium_paid=premium_paid, expected_return=0.1, compute_pnl=True, show_plots=True, save_plots=False)
        if run_calibration:
            print(f"\n{'=' * 80}")
            print('RUNNING MARKET-CONSISTENT CALIBRATION')
            print(f"{'=' * 80}\n")
            try:
                calibration_result = calibrate_model_to_market(ticker=ticker, spot=market_data['spot'], as_of=market_data['as_of'], target_expiry=market_data['expiry'], option_chain=df, model='gbm', n_paths=10000, seed=RANDOM_SEED)
                print(f"\nCalibration RMSE: ${calibration_result['rmse']:.4f}")
                print(f"Calibration MAPE: {calibration_result['mape']:.2f}%")
                results['calibration'] = calibration_result
            except Exception as e:
                print(f'Calibration error: {e}')
        if run_bump_test:
            print(f"\n{'=' * 80}")
            print('RUNNING GREEKS BUMP SENSITIVITY TEST')
            print(f"{'=' * 80}\n")
            try:
                calendar = results['calendar']
                rate_curve = build_rate_curve(market_data['rates'], calendar)
                div_schedule = build_dividend_schedule(market_data['dividends'], calendar)
                bump_result = test_greek_bump_sensitivity(market={'spot': market_data['spot'], 'as_of': market_data['as_of']}, contract=create_option_contract(option_type, strike, market_data['expiry'], 'American'), calendar=calendar, rate_curve=rate_curve, div_schedule=div_schedule, vol=results['metadata']['vol'], n_paths=n_paths // 2, seed=RANDOM_SEED, div_params=market_data['div_params'], model='gbm')
                print(f"\nDelta CV: {bump_result['delta_cv']:.2f}%")
                print(f"Gamma CV: {bump_result['gamma_cv']:.2f}%")
                print(f"Status: {bump_result['status']}")
                results['bump_sensitivity'] = bump_result
            except Exception as e:
                print(f'Bump test error: {e}')
        print('\n' + '=' * 80)
        print('✓ PRICING COMPLETED SUCCESSFULLY!')
        print('=' * 80)
        print(f"\nModel Fair Value: ${results['pricing_Q']['price']:.4f}")
        print(f'Market Price:     ${premium_paid:.2f}')
        print(f"Difference:       ${results['pricing_Q']['price'] - premium_paid:.4f}")
        print(f"\nDelta: {results['greeks']['delta']:.4f}")
        print(f"Gamma: {results['greeks']['gamma']:.6f}")
        print(f"Vega:  {results['greeks']['vega']:.4f}")
        if results['forecast_P']:
            print(f"\nExpected P/L:       ${results['forecast_P']['expected_pnl']:.2f}")
            print(f"Probability Profit: {results['forecast_P']['prob_profit']:.2%}")
        print('=' * 80)
        return results
    except Exception as e:
        print(f'\nError during pricing: {e}')
        import traceback
        traceback.print_exc()
        return None
