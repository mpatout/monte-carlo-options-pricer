"""Trading calendar, discount curves, and discrete dividend schedules."""

import numpy as np
import warnings
from datetime import datetime, timedelta
from scipy import interpolate
from typing import Dict, List, Literal, Optional


def create_market_snapshot(ticker: str, spot: float, as_of: datetime, implied_vol_surface: Dict, dividends_raw: List[Dict], rates_raw: Dict, div_params: Optional[Dict]=None, borrow_rate: Optional[float]=None) -> Dict:
    """Create structured market snapshot dict for pricing engine."""
    return {'ticker': ticker, 'spot': spot, 'as_of': as_of, 'implied_vol_surface': implied_vol_surface, 'dividends_raw': dividends_raw, 'div_params': div_params, 'rates_raw': rates_raw, 'borrow_rate': borrow_rate or 0.0}


def create_option_contract(option_type: Literal['call', 'put'], strike: float, expiry: datetime, style: Literal['American', 'European']='American') -> Dict:
    """Create option contract specification dict."""
    return {'style': style.lower(), 'strike': strike, 'expiry': expiry, 'type': option_type}


def build_trading_calendar(as_of: datetime, expiry: datetime) -> np.ndarray:
    """Generate NYSE trading days between valuation date and expiry (excludes weekends and holidays)."""
    nyse_holidays = [datetime(2024, 1, 1), datetime(2024, 1, 15), datetime(2024, 2, 19), datetime(2024, 3, 29), datetime(2024, 5, 27), datetime(2024, 6, 19), datetime(2024, 7, 4), datetime(2024, 9, 2), datetime(2024, 11, 28), datetime(2024, 12, 25), datetime(2025, 1, 1), datetime(2025, 1, 20), datetime(2025, 2, 17), datetime(2025, 4, 18), datetime(2025, 5, 26), datetime(2025, 6, 19), datetime(2025, 7, 4), datetime(2025, 9, 1), datetime(2025, 11, 27), datetime(2025, 12, 25), datetime(2026, 1, 1), datetime(2026, 1, 19), datetime(2026, 2, 16), datetime(2026, 4, 3), datetime(2026, 5, 25), datetime(2026, 6, 19), datetime(2026, 7, 3), datetime(2026, 9, 7), datetime(2026, 11, 26), datetime(2026, 12, 25)]
    current = as_of.replace(hour=0, minute=0, second=0, microsecond=0)
    expiry_date = expiry.replace(hour=0, minute=0, second=0, microsecond=0)
    trading_days = []
    while current <= expiry_date:
        if current.weekday() < 5 and current not in nyse_holidays:
            trading_days.append(current)
        current += timedelta(days=1)
    return np.array(trading_days, dtype='datetime64[D]')


def get_time_indices(dates: np.ndarray, calendar: np.ndarray) -> np.ndarray:
    """Map dates to their indices in the trading calendar."""
    indices = []
    for date in dates:
        idx = np.where(calendar == date)[0]
        if len(idx) > 0:
            indices.append(idx[0])
    return np.array(indices, dtype=int)


def year_fraction(start: datetime, end: datetime) -> float:
    """Calculate year fraction between two dates (ACT/365 convention)."""
    delta = (end - start).days
    return delta / 365.0


def build_rate_curve(rates_raw: Dict, calendar: np.ndarray) -> Dict:
    """Build discount factor curve with log-DF interpolation and sanity checks.
    
    Industry best practice: interpolate in log discount factor space for monotonicity.
    
    Args:
        rates_raw: Dict of tenor -> rate mappings (e.g., {tenors: [0.25, 5, 10], rates: [0.04, 0.04, 0.04]})
                   Must provide 'tenors' and 'rates' keys (no defaults)
        calendar: Trading calendar array
        
    Returns:
        Dict with times, zero_rates, discount_factors, interpolator, diagnostics
        
    Notes:
        - Interpolates ln(DF) linearly, then recovers DF and implied rates
        - Performs sanity checks but does NOT modify data (warnings only)
        - User is responsible for providing clean input data
    """
    # Step 1: Extract input tenors and rates (NO DEFAULTS)
    tenors = np.array(rates_raw['tenors'])
    rates = np.array(rates_raw['rates'])
    
    # Step 2: Compute log discount factors at input tenors
    # ln(DF) = -r * T
    log_dfs_input = -rates * tenors
    log_dfs_input[0] = 0.0  # DF(0) = 1, so ln(DF(0)) = 0
    
    # Step 3: Create interpolator in log DF space
    interp_log_df = interpolate.interp1d(
        tenors, log_dfs_input,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=True
    )
    
    # Step 4: Compute times for calendar
    base_date = calendar[0].astype('M8[D]').astype('O')
    times = np.array([year_fraction(base_date, date.astype('M8[D]').astype('O')) for date in calendar])
    
    # Step 5: Interpolate log DF and recover DF
    log_dfs = interp_log_df(times)
    discount_factors = np.exp(log_dfs)
    
    # Step 6: Recover implied zero rates
    # r(T) = -ln(DF(T)) / T
    zero_rates = np.zeros_like(times)
    zero_rates[0] = rates[0]  # Use input rate for T=0
    zero_rates[1:] = -log_dfs[1:] / times[1:]
    
    # Step 7: SANITY CHECKS (warnings only, no modifications)
    diagnostics = {'warnings': [], 'errors': []}
    
    # Check 1: DFs strictly decreasing
    if not np.all(np.diff(discount_factors) <= 1e-10):
        max_increase = np.max(np.diff(discount_factors))
        diagnostics['warnings'].append(f'Discount factors not strictly decreasing (max increase: {max_increase:.2e})')
    
    # Check 2: No DF > 1 for T > 0
    if np.any(discount_factors[1:] > 1.0):
        max_df = np.max(discount_factors[1:])
        diagnostics['errors'].append(f'DF > 1 detected: max = {max_df:.6f}')
    
    # Check 3: No negative DF
    if np.any(discount_factors < 0):
        min_df = np.min(discount_factors)
        diagnostics['errors'].append(f'Negative DF detected: min = {min_df:.6f}')
    
    # Check 4: Reasonable rate bounds (0% to 20%)
    if np.any(zero_rates < 0) or np.any(zero_rates > 0.20):
        bad_rates = zero_rates[(zero_rates < 0) | (zero_rates > 0.20)]
        diagnostics['warnings'].append(f'Extreme rates detected (outside [0%, 20%]): {bad_rates[:3]}')
    
    # Create interpolator for zero rates
    rate_interpolator = interpolate.interp1d(
        times, zero_rates,
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False
    )
    
    return {
        'times': times,
        'zero_rates': zero_rates,
        'discount_factors': discount_factors,
        'interpolator': rate_interpolator,
        'diagnostics': diagnostics,
        'method': 'log_df_interpolation'
    }


def discount_cashflows(cashflows: np.ndarray, rate_curve: Dict, from_idx: np.ndarray, to_idx: int=0) -> np.ndarray:
    """Discount cashflows from future time indices back to present."""
    discount_factors = rate_curve['discount_factors']
    if np.isscalar(from_idx):
        df_from = discount_factors[from_idx]
        df_to = discount_factors[to_idx]
        return cashflows * (df_to / df_from)
    else:
        df_from = discount_factors[from_idx]
        df_to = discount_factors[to_idx]
        return cashflows * (df_to / df_from)


def build_dividend_schedule(dividends_raw: List[Dict], calendar: np.ndarray) -> Dict:
    """Map dividend ex-dates to simulation time indices.
    
    Snaps ex-dates to nearest prior trading day if falling on holiday/weekend.
    
    Args:
        dividends_raw: List of dividend dicts with 'ex_date' and 'amount'
        calendar: Trading calendar array
        
    Returns:
        Dict with time indices and amounts for each dividend
    """
    if not dividends_raw:
        return {'div_idx': np.array([], dtype=int), 'div_amounts': np.array([], dtype=float), 'flags': [], 'has_dividends': False}
    div_dates = []
    div_amounts = []
    flags = []
    warnings = []
    for div in dividends_raw:
        ex_date = div['ex_date']
        if isinstance(ex_date, str):
            ex_date = datetime.strptime(ex_date, '%Y-%m-%d')
        ex_date_np = np.datetime64(ex_date, 'D')
        idx = np.where(calendar == ex_date_np)[0]
        if len(idx) > 0:
            div_dates.append(idx[0])
            div_amounts.append(div['amount'])
            flags.append(div.get('flag', 'announced'))
        else:
            calendar_dates = calendar.astype('datetime64[D]')
            prior_dates = calendar_dates[calendar_dates < ex_date_np]
            if len(prior_dates) > 0:
                snapped_date = prior_dates[-1]
                snapped_idx = np.where(calendar == snapped_date)[0][0]
                div_dates.append(snapped_idx)
                div_amounts.append(div['amount'])
                flags.append(div.get('flag', 'announced') + '_adjusted')
                warning = f"Dividend ex-date {ex_date.strftime('%Y-%m-%d')} (weekend/holiday) → snapped to prior trading day {snapped_date}"
                warnings.append(warning)
                print(f'  ⚠ {warning}')
            else:
                warning = f"Dividend ex-date {ex_date.strftime('%Y-%m-%d')} before calendar start - skipped"
                warnings.append(warning)
                print(f'  ⚠ {warning}')
    return {'div_idx': np.array(div_dates, dtype=int), 'div_amounts': np.array(div_amounts, dtype=float), 'flags': flags, 'warnings': warnings, 'has_dividends': len(div_dates) > 0}


def apply_dividend_jumps(paths: np.ndarray, div_schedule: Dict) -> np.ndarray:
    """Apply discrete dividend jumps to price paths (ex-dividend adjustments)."""
    paths_adjusted = paths.copy()
    for idx, amount in zip(div_schedule['div_idx'], div_schedule['div_amounts']):
        if idx < paths.shape[1]:
            paths_adjusted[:, idx:] -= amount
            paths_adjusted[:, idx:] = np.maximum(paths_adjusted[:, idx:], 0.01)
    return paths_adjusted
