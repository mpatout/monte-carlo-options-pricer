"""Live market data: spot, option chains, implied-vol surface, dividends, rates."""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import Dict, List, Literal, Tuple


def fetch_ticker_data(ticker: str) -> Dict:
    """Fetch live market data for a ticker via yfinance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY')
        
    Returns:
        Dict containing spot price, available expirations, and ticker info
    """
    print(f'\nFetching market data for {ticker}...')
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        spot = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if spot is None:
            raise ValueError(f'Could not fetch spot price for {ticker}')
        expirations = stock.options
        if not expirations:
            raise ValueError(f'No options available for {ticker}')
        return {'ticker': ticker, 'spot': float(spot), 'expirations': list(expirations), 'info': info, 'stock_object': stock}
    except Exception as e:
        raise ValueError(f'Error fetching data for {ticker}: {str(e)}')


def get_available_expirations(ticker: str) -> List[str]:
    """Get list of available option expiration dates for a ticker."""
    data = fetch_ticker_data(ticker)
    return data['expirations']


def calculate_iv_from_price(option_price: float, spot: float, strike: float, time_to_expiry: float, rate: float, option_type: Literal['call', 'put'], initial_guess: float=0.25) -> float:
    """Calculate implied volatility from option market price via Newton-Raphson (reverse black scholes).
    
    Args:
        option_price: Market price of the option
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        initial_guess: Starting IV guess (default: 25%)
        
    Returns:
        Implied volatility (annualized) or None if calculation fails
    """
    from scipy.stats import norm
    if time_to_expiry <= 0 or option_price <= 0:
        return 0.0
    if option_type == 'call':
        intrinsic = max(spot - strike, 0)
    else:
        intrinsic = max(strike - spot, 0)
    if option_price <= intrinsic:
        return 0.0
    vol = initial_guess
    max_iterations = 100
    tolerance = 1e-06
    for i in range(max_iterations):
        d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
        d2 = d1 - vol * np.sqrt(time_to_expiry)
        if option_type == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry)
        diff = option_price - price
        if abs(diff) < tolerance:
            return max(vol, 0.0)
        if vega < 1e-10:
            return vol
        vol = vol + diff / vega
        vol = max(0.01, min(vol, 3.0))
    return max(vol, 0.0)


def fetch_option_chain(ticker: str, expiry_str: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch complete option chain (calls and puts) for given expiration."""
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry_str)
    return (chain.calls, chain.puts)


def build_implied_vol_surface(ticker: str, expiry_str: str, option_type: Literal['call', 'put'], spot: float, rate: float=None) -> Dict:
    """Build implied volatility surface from for each strike by running calculate_iv_from_price.
    
    Computes IV for each strike by inverting Black-Scholes formula on market prices.
    Filters out low-volume options and applies quality checks.
    
    Args:
        ticker: Stock symbol
        expiry_str: Expiration date string (YYYY-MM-DD)
        option_type: 'call' or 'put'
        spot: Current spot price
        rate: Risk-free rate (uses Treasury rate if None)
        
    Returns:
        Dict with strikes, IVs, moneyness, volumes, and interpolation function
    """
    calls, puts = fetch_option_chain(ticker, expiry_str)
    df = calls if option_type == 'call' else puts
    df = df[(df['volume'] > 0) & (df['bid'] > 0) & (df['ask'] > 0)].copy()
    if len(df) == 0:
        print('  ⚠ WARNING: No liquid options found with valid bid/ask')
        return {'strikes': [], 'vols': []}
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
    today = datetime.now()
    time_to_expiry = (expiry_date - today).days / 365.0
    if time_to_expiry <= 0:
        print('  ⚠ WARNING: Expiry date is in the past')
        return {'strikes': [], 'vols': []}
    if rate is None:
        rate_data = fetch_risk_free_rate()
        rate = np.interp(time_to_expiry, rate_data['tenors'], rate_data['rates'])
    ivs = []
    valid_strikes = []
    failed_strikes = []
    for _, row in df.iterrows():
        try:
            iv = calculate_iv_from_price(option_price=row['mid_price'], spot=spot, strike=row['strike'], time_to_expiry=time_to_expiry, rate=rate, option_type=option_type)
            if iv > 0:
                ivs.append(iv)
                valid_strikes.append(row['strike'])
            else:
                failed_strikes.append(row['strike'])
        except Exception as e:
            failed_strikes.append(row['strike'])
            continue
    if len(ivs) == 0:
        print('  ⚠⚠ CRITICAL: Could not calculate any valid IVs from option prices')
        print('  ⚠⚠ Check if bid/ask spreads are reasonable and options have liquidity')
        return {'strikes': [], 'vols': []}
    if len(failed_strikes) > 0 and len(failed_strikes) < 10:
        print(f'  ⚠ WARNING: Failed to calculate IV for {len(failed_strikes)} strikes: {failed_strikes[:5]}')
    elif len(failed_strikes) >= 10:
        print(f'  ⚠ WARNING: Failed to calculate IV for {len(failed_strikes)} strikes')
    return {'strikes': valid_strikes, 'vols': ivs}


def fetch_dividends(ticker: str, lookback_years: int=5) -> Dict:
    """Fetch historical dividends and project future dividends using GBM.
    
    Args:
        ticker: Stock symbol
        lookback_years: Years of history to analyze (default: 5)
        
    Returns:
        Dict containing historical dividends and GBM projection parameters
    """
    stock = yf.Ticker(ticker)
    div_history = stock.dividends
    if len(div_history) == 0:
        print(f'  No dividends found for {ticker}')
        return {'has_dividends': False, 'S0': 0.0, 'mu': 0.0, 'sigma': 0.0, 'frequency_days': 90, 'schedule': []}
    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=lookback_years * 365))
    div_index_naive = div_history.index.tz_localize(None) if div_history.index.tz is not None else div_history.index
    recent_divs = div_history[div_index_naive >= cutoff_date]
    if len(recent_divs) == 0:
        return {'has_dividends': False, 'S0': 0.0, 'mu': 0.0, 'sigma': 0.0, 'frequency_days': 90, 'schedule': []}
    if len(recent_divs) >= 2:
        dates_naive = recent_divs.index.tz_localize(None) if recent_divs.index.tz is not None else recent_divs.index
        time_diffs = np.diff(dates_naive.to_pydatetime())
        avg_days_between = np.mean([td.days for td in time_diffs])
    else:
        avg_days_between = 90
    try:
        annual_div_rate = stock.info.get('dividendRate', None)
        if annual_div_rate is None or annual_div_rate == 0:
            annual_div_rate = recent_divs.values[-1] * (365.0 / avg_days_between)
        S0 = float(annual_div_rate)
    except:
        S0 = recent_divs.values[-1] * (365.0 / avg_days_between)
    if len(recent_divs) >= 4:
        div_values = recent_divs.values
        growth_rates = []
        for i in range(1, len(div_values)):
            if div_values[i - 1] > 0:
                growth = (div_values[i] - div_values[i - 1]) / div_values[i - 1]
                growth_rates.append(growth)
        if len(growth_rates) > 0:
            periods_per_year = 365.0 / avg_days_between
            mu = np.mean(growth_rates) * periods_per_year
            sigma = np.std(growth_rates) * np.sqrt(periods_per_year)
        else:
            print('  ⚠ WARNING: No dividend growth history - using 0% growth')
            mu = 0.0
            sigma = 0.0
    else:
        print('  ⚠ WARNING: Insufficient dividend history (<4 payments) - using 0% growth')
        mu = 0.0
        sigma = 0.0
    last_div_idx = recent_divs.index[-1]
    last_div_date = last_div_idx.tz_localize(None).to_pydatetime() if last_div_idx.tz is not None else last_div_idx.to_pydatetime()
    future_dividends = []
    current_date = datetime.now()
    projection_date = last_div_date
    max_date = current_date + timedelta(days=730)
    while projection_date <= max_date:
        projection_date += timedelta(days=int(avg_days_between))
        if projection_date > current_date:
            future_dividends.append({'ex_date': projection_date.strftime('%Y-%m-%d'), 'amount': float(S0 / (365.0 / avg_days_between)), 'flag': 'estimated'})
    print(f'  Dividend GBM: S0=${S0:.2f}/yr, μ={mu:.2%}/yr, σ={sigma:.2%}/yr')
    return {'has_dividends': True, 'S0': S0, 'mu': mu, 'sigma': sigma, 'frequency_days': avg_days_between, 'schedule': future_dividends}


def fetch_risk_free_rate() -> Dict:
    """Fetch live Treasury zero rates for discounting (Q-measure only).
    
    Uses continuously compounded zero rates from:
        - ^IRX: 3-month T-bill (0.25Y)
        - ^FVX: 5-year Treasury (5.0Y)
        - ^TNX: 10-year Treasury (10.0Y)
        - ^TYX: 30-year Treasury (30.0Y)
    
    Returns:
        Dict with tenors (years) and zero rates (decimal)
        Linear interpolation, flat extrapolation beyond endpoints
    """
    try:
        import yfinance as yf
        
        # Fetch Treasury tickers (quotes in %)
        tickers = {"IRX": "^IRX", "FVX": "^FVX", "TNX": "^TNX", "TYX": "^TYX"}
        rates_data = {}
        
        for name, ticker in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if len(hist) > 0:
                    rates_data[name] = float(hist["Close"].iloc[-1]) / 100.0  # Convert % to decimal
            except:
                pass
        
        # Require at least 3M and 10Y
        if "IRX" not in rates_data or "TNX" not in rates_data:
            raise ValueError(f"Missing critical rates (IRX or TNX)")
        
        # Extract rates
        r_3m = rates_data["IRX"]
        r_5y = rates_data.get("FVX")
        r_10y = rates_data["TNX"]
        r_30y = rates_data.get("TYX")
        
        # Build curve (no interpolation - just use actual Treasury points)
        # Overnight = 3M approximation
        tenors = [0.0, 0.25]
        rates = [r_3m, r_3m]
        
        if r_5y:
            tenors.append(5.0)
            rates.append(r_5y)
        
        tenors.append(10.0)
        rates.append(r_10y)
        
        if r_30y:
            tenors.append(30.0)
            rates.append(r_30y)
        
        print(f"   Live Treasury rates: 3M={r_3m:.2%}, 10Y={r_10y:.2%}", end="")
        if r_5y:
            print(f", 5Y={r_5y:.2%}", end="")
        if r_30y:
            print(f", 30Y={r_30y:.2%}", end="")
        print()
        
        return {"tenors": tenors, "rates": rates}
        
    except Exception as e:
        print(f"   WARNING: Could not fetch live Treasury rates ({str(e)})")
        print("   Using approximate rate curve (not live data)")
        return {
            "tenors": [0.0, 0.25, 5.0, 10.0, 30.0],
            "rates": [0.0425, 0.0425, 0.042, 0.0415, 0.043]
        }


def fetch_all_market_data(ticker: str, expiry_str: str, option_type: Literal['call', 'put']) -> Dict:
    """Fetch complete market snapshot: spot, IV surface, dividends, and rates.
    
    Central data aggregation function that calls all market data fetchers.
    
    Args:
        ticker: Stock symbol
        expiry_str: Expiration date string
        option_type: 'call' or 'put'
        
    Returns:
        Comprehensive market data dict ready for pricing
    """
    print(f"\n{'=' * 80}")
    print(f'FETCHING MARKET DATA: {ticker} {option_type.upper()}')
    print(f"{'=' * 80}")
    ticker_data = fetch_ticker_data(ticker)
    spot = ticker_data['spot']
    print(f'  ✓ Spot Price: ${spot:.2f}')
    if expiry_str not in ticker_data['expirations']:
        print(f'\n  Available expirations:')
        for exp in ticker_data['expirations'][:10]:
            print(f'    - {exp}')
        raise ValueError(f'Expiration {expiry_str} not available. Choose from list above.')
    expiry = datetime.strptime(expiry_str, '%Y-%m-%d')
    print(f'  ✓ Expiration: {expiry_str}')
    print(f'  Fetching option chain...')
    vol_surface = build_implied_vol_surface(ticker, expiry_str, option_type, spot)
    print(f"  ✓ IV Surface: {len(vol_surface['strikes'])} strikes")
    print(f'  Fetching dividends...')
    div_params = fetch_dividends(ticker)
    dividends_before_expiry = [d for d in div_params['schedule'] if datetime.strptime(d['ex_date'], '%Y-%m-%d') <= expiry]
    print(f'  ✓ Dividends: {len(dividends_before_expiry)} before expiry')
    print(f'  Fetching risk-free rates...')
    rates = fetch_risk_free_rate()
    print(f"  ✓ Rate Curve: {len(rates['tenors'])} points")
    print(f"{'=' * 80}\n")
    return {'ticker': ticker, 'spot': spot, 'expiry': expiry, 'expiry_str': expiry_str, 'option_type': option_type, 'vol_surface': vol_surface, 'dividends': dividends_before_expiry, 'div_params': div_params, 'rates': rates, 'as_of': datetime.now()}


def prepare_iv_data(option_chain: pd.DataFrame, spot: float, option_type: str, min_volume: int=1, min_open_interest: int=0, max_bid_ask_spread_pct: float=0.5, moneyness_min: float=0.8, moneyness_max: float=1.2) -> Dict:
    """Filter and prepare IV data for calibration with quality checks."""
    df = option_chain.copy()
    n_initial = len(df)
    df = df[(df['volume'] >= min_volume) | (df['openInterest'] >= min_open_interest)]
    n_after_liquidity = len(df)
    df = df[df['impliedVolatility'] > 0]
    df = df[df['impliedVolatility'] < 3.0]
    n_after_iv = len(df)
    if 'bid' in df.columns and 'ask' in df.columns:
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
        df = df[df['spread_pct'] <= max_bid_ask_spread_pct]
    n_after_spread = len(df)
    df['moneyness'] = df['strike'] / spot
    df = df[(df['moneyness'] >= moneyness_min) & (df['moneyness'] <= moneyness_max)]
    n_after_moneyness = len(df)
    df['weight'] = np.exp(-(df['moneyness'] - 1.0) ** 2 / 0.1)
    df['weight'] *= np.log(1 + df['volume'])
    df['weight'] /= df['weight'].sum()
    strikes = df['strike'].values
    ivs = df['impliedVolatility'].values
    weights = df['weight'].values
    moneyness = df['moneyness'].values
    diagnostics = {'n_initial': n_initial, 'n_after_liquidity': n_after_liquidity, 'n_after_iv': n_after_iv, 'n_after_spread': n_after_spread, 'n_final': n_after_moneyness, 'moneyness_range': (float(moneyness.min()), float(moneyness.max())), 'iv_range': (float(ivs.min()), float(ivs.max())), 'iv_mean': float(ivs.mean()), 'iv_weighted_mean': float(np.average(ivs, weights=weights)), 'dropped_pct': 100 * (1 - n_after_moneyness / n_initial) if n_initial > 0 else 0}
    return {'strikes': strikes, 'ivs': ivs, 'weights': weights, 'moneyness': moneyness, 'diagnostics': diagnostics}


def report_calibration_diagnostics(iv_data: Dict, model: str='GBM') -> None:
    """Print calibration diagnostics and data quality report to console."""
    diag = iv_data['diagnostics']
    print(f"\n{'=' * 80}")
    print(f'CALIBRATION DIAGNOSTICS - {model}')
    print(f"{'=' * 80}")
    print(f"  Initial strikes:       {diag['n_initial']}")
    print(f"  After liquidity:       {diag['n_after_liquidity']}")
    print(f"  After IV filter:       {diag['n_after_iv']}")
    print(f"  After spread filter:   {diag['n_after_spread']}")
    print(f"  Final strikes used:    {diag['n_final']}")
    print(f"  Dropped:               {diag['dropped_pct']:.1f}%")
    print(f"  Moneyness range:       [{diag['moneyness_range'][0]:.2f}, {diag['moneyness_range'][1]:.2f}]")
    print(f"  IV range:              [{diag['iv_range'][0]:.2%}, {diag['iv_range'][1]:.2%}]")
    print(f"  IV (equal-weighted):   {diag['iv_mean']:.2%}")
    print(f"  IV (ATM-weighted):     {diag['iv_weighted_mean']:.2%}")
    if diag['n_final'] < 5:
        print(f"  ⚠ WARNING: Only {diag['n_final']} strikes available - calibration may be unreliable")
    if diag['moneyness_range'][0] > 0.95 or diag['moneyness_range'][1] < 1.05:
        print(f'  ⚠ WARNING: Limited moneyness range - extrapolation needed')
    if diag['iv_range'][1] - diag['iv_range'][0] > 0.5:
        print(f'  ⚠ WARNING: Wide IV range suggests volatility skew/smile')
    if diag['iv_mean'] < 0.05:
        print(f'  ⚠⚠ CRITICAL: IVs < 5% are likely STALE DATA from expired options')
        print(f'  ⚠⚠ Consider using a longer-dated expiry (30+ days) or manually override vol')
    print(f"{'=' * 80}\n")


def display_available_strikes(ticker: str, expiry_str: str, option_type: Literal['call', 'put'], spot: float) -> List[float]:
    """Display formatted table of available strikes with moneyness indicators."""
    calls, puts = fetch_option_chain(ticker, expiry_str)
    df = calls if option_type == 'call' else puts
    df = df[df['volume'] > 0].copy()
    strikes = sorted(df['strike'].unique())
    print(f'\nAvailable strikes for {ticker} {expiry_str} {option_type}:')
    print(f"{'Strike':>10} | {'Moneyness':>12} | {'Volume':>10}")
    print('-' * 40)
    for strike in strikes[:20]:
        moneyness = strike / spot
        row = df[df['strike'] == strike].iloc[0]
        volume = row['volume']
        marker = ' ← ATM' if abs(moneyness - 1.0) < 0.05 else ''
        print(f'${strike:>8.2f} | {moneyness:>11.2%} | {volume:>10.0f}{marker}')
    if len(strikes) > 20:
        print(f'... and {len(strikes) - 20} more strikes')
    return strikes
