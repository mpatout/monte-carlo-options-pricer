"""Human-readable pricing and forecast reports."""

import numpy as np
from typing import Any, Dict

from .config import RANDOM_SEED


def generate_pricing_report(market: Dict, contract: Dict, pricing_result: Dict, greeks: Dict, validations: Dict, vol: float, model: str='gbm', heston_params: Any=None, heston_diagnostics: Dict=None) -> str:
    """Generate comprehensive pricing report with all results."""
    report = []
    report.append('=' * 80)
    report.append('AMERICAN OPTION PRICING REPORT')
    report.append('=' * 80)
    report.append('')
    report.append('MARKET SNAPSHOT')
    report.append('-' * 80)
    report.append(f"Ticker:           {market['ticker']}")
    report.append(f"Valuation Date:   {market['as_of'].strftime('%Y-%m-%d')}")
    report.append(f"Spot Price:       ${market['spot']:.2f}")
    report.append('')
    report.append('CONTRACT SPECIFICATION')
    report.append('-' * 80)
    report.append(f"Style:            {contract['style'].upper()}")
    report.append(f"Type:             {contract['type'].upper()}")
    report.append(f"Strike:           ${contract['strike']:.2f}")
    report.append(f"Expiry:           {contract['expiry'].strftime('%Y-%m-%d')}")
    days_to_expiry = (contract['expiry'] - market['as_of']).days
    report.append(f'Days to Expiry:   {days_to_expiry}')
    report.append('')
    report.append('MODEL PARAMETERS')
    report.append('-' * 80)
    if model == 'gbm':
        report.append(f'Model:            GBM with Discrete Dividends')
    else:
        report.append(f'Model:            HESTON STOCHASTIC VOLATILITY')
    if model == 'gbm':
        report.append(f'Volatility:       {vol:.2%}')
    else:
        if heston_params is not None:
            report.append(f'Initial Variance: {heston_params.v0:.6f} (vol={np.sqrt(heston_params.v0):.2%})')
            report.append(f'Mean Reversion:   κ = {heston_params.kappa:.4f}')
            report.append(f'Long-run Var:     θ = {heston_params.theta:.6f} (vol={np.sqrt(heston_params.theta):.2%})')
            report.append(f'Vol of Vol:       σ = {heston_params.sigma:.4f}')
            report.append(f'Correlation:      ρ = {heston_params.rho:.4f}')
            feller = 2 * heston_params.kappa * heston_params.theta
            feller_rhs = heston_params.sigma ** 2
            feller_pass = feller >= feller_rhs
            report.append(f"Feller Condition: {feller:.6f} {('≥' if feller_pass else '<')} {feller_rhs:.6f} ({('PASS' if feller_pass else 'FAIL')})")
        if heston_diagnostics is not None:
            report.append(f"Variance Range:   [{heston_diagnostics.get('v_min', 0):.6f}, {heston_diagnostics.get('v_max', 0):.6f}]")
            report.append(f"Absorptions:      {heston_diagnostics.get('n_absorptions', 0):,}")
    report.append(f'Volatility:       {vol:.2%}')
    report.append(f'Random Seed:      {RANDOM_SEED}')
    report.append('')
    report.append('PRICING RESULTS (Q-MEASURE)')
    report.append('-' * 80)
    if 'price_raw' in pricing_result:
        report.append(f"Raw MC Price:     ${pricing_result['price_raw']:.4f}")
        report.append(f"CV Adjustment:    ${pricing_result['cv_adjustment']:.4f}")
        report.append(f"  European BS:    ${pricing_result['euro_bs']:.4f}")
        report.append(f"  European MC:    ${pricing_result['euro_mc']:.4f}")
    report.append(f"Fair Value:       ${pricing_result['price']:.4f}")
    report.append(f"Std Error:        ${pricing_result['std_error']:.4f}")
    report.append(f"95% CI:           [${pricing_result['confidence_interval'][0]:.4f}, ${pricing_result['confidence_interval'][1]:.4f}]")
    if 'cv_adjustment' in pricing_result:
        cv_improvement = abs(pricing_result['cv_adjustment'])
        report.append(f'CV Variance Reduction: ${cv_improvement:.4f}')
    report.append('')
    report.append('GREEKS (T=0)')
    report.append('-' * 80)
    report.append(f"Delta:            {greeks['delta']:>10.4f}  (∂V/∂S)")
    report.append(f"Gamma:            {greeks['gamma']:>10.6f}  (∂²V/∂S²)")
    report.append(f"Vega:             {greeks['vega']:>10.4f}  (∂V/∂σ)")
    report.append(f"Theta:            {greeks['theta']:>10.4f}  (∂V/∂t, per year)")
    report.append(f"Rho:              {greeks['rho']:>10.4f}  (∂V/∂r)")
    report.append('')
    report.append('VALIDATION')
    report.append('-' * 80)
    for key, val in validations.items():
        if 'pass' in val:
            status = 'PASS' if val['pass'] else 'FAIL'
            report.append(f'{key:.<30} {status:>10}')
            report.append(f"  {val['message']}")
    report.append('')
    report.append('=' * 80)
    return '\n'.join(report)


def generate_forecast_report(forecast_result: Dict, premium_paid: float) -> str:
    """Generate real-world P&L forecast report (P-measure)."""
    report = []
    report.append('=' * 80)
    report.append('REAL-WORLD FORECAST (P-MEASURE)')
    report.append('=' * 80)
    report.append('')
    report.append(f'Premium Paid:      ${premium_paid:.2f}')
    report.append(f"Expected P/L:      ${forecast_result['expected_pnl']:.2f}")
    report.append(f"Probability Profit: {forecast_result['prob_profit']:.2%}")
    report.append('')
    report.append('P/L PERCENTILES')
    report.append('-' * 80)
    for pct, value in forecast_result['percentiles'].items():
        report.append(f'{pct:>5}:  ${value:>10.2f}')
    report.append('')
    report.append('=' * 80)
    return '\n'.join(report)
