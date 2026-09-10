# Monte Carlo Options Pricer

A Monte Carlo pricing engine for American-style US equity options, built to be as close to
industry standard as a self-contained Python project reasonably gets.

The model fetches live market data with `yfinance` — option chains, dividends, quotes, and
Treasury yields — bootstraps a real risk-free **curve** rather than assuming a flat rate, then
prices under either geometric Brownian motion or a Heston stochastic-volatility process
calibrated to the full implied-volatility smile. American exercise is valued with
Longstaff-Schwartz least-squares Monte Carlo.

## Features

**Models**
- Geometric Brownian motion — fast, stable Greeks
- Heston stochastic volatility, simulated via Andersen's QE scheme
- Heston calibration to the full IV smile (not just at-the-money), with a Feller-condition check
- Semi-analytic Heston European prices via the characteristic function, used to validate the simulation

**Pricing**
- American exercise via Longstaff-Schwartz, regressing continuation value on a Laguerre polynomial basis
- European pricing for comparison and control variates
- Discrete dividend modeling with GBM projection of future payments
- A real NYSE trading calendar, holidays excluded — not 252 evenly spaced steps

**Variance reduction**
- Sobol quasi-Monte Carlo sequences
- Antithetic variates
- Black-Scholes control variates with an optimal beta
- Brownian bridge construction (implemented, off by default)

**Greeks**
- Pathwise derivative estimator for delta
- Likelihood-ratio estimator for vega
- Adaptive finite differences with Richardson extrapolation and convergence tolerance
- Stability reporting across bump sizes, so you can see when a Greek is not trustworthy

**Analysis**
- Real-world (P-measure) P&L forecasting under a fixed exercise policy, distinct from the
  risk-neutral (Q-measure) pricing measure
- Validation checks: intrinsic-value bounds, convergence sweeps, European equivalence
- Diagnostic plots: path fans, exercise boundaries, P&L distributions and risk analysis

## Install

```bash
git clone https://github.com/mpatout/monte-carlo-options-pricer.git
cd monte-carlo-options-pricer
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ and an internet connection for market data.

## Usage

Interactive CLI — prompts for strike, expiry, and model:

```bash
python -m mc_pricer
python -m mc_pricer AAPL call
```

Programmatic:

```python
from datetime import datetime
from mc_pricer import run_pricer

result = run_pricer(
    ticker="AAPL",
    spot=225.0,
    strike=230.0,
    expiry=datetime(2026, 1, 16),
    option_type="call",
    model="heston",       # or "gbm"
    n_paths=50_000,
    show_plots=False,
)

print(result["pricing_Q"]["price"])     # fair value under the risk-neutral measure
print(result["greeks"]["delta"])
print(result["forecast_P"]["prob_profit"])
```

`run_pricer` returns a dict with:

| Key | Contents |
|---|---|
| `pricing_Q` | `price`, `confidence_interval`, `std_error`, `exercise_boundary` |
| `greeks` | delta, gamma, vega, rho, theta |
| `forecast_P` | Real-world P&L forecast (`None` if `compute_pnl=False`) |
| `validations` | Bound checks, convergence, European equivalence |
| `metadata` | Inputs, calibrated vol, path count, seed |
| `paths`, `calendar` | Raw simulation output for further analysis |

Lower-level pieces compose directly:

```python
from mc_pricer import calibrate_heston_params, simulate_paths_heston_qe, price_american_lsm
```

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | Global simulation constants |
| `market_data.py` | Spot, option chains, IV surface, dividends, rates |
| `calendar_curves.py` | Trading calendar, discount curves, dividend schedules |
| `models.py` | GBM vol and Heston parameter calibration |
| `simulation.py` | Path generation — GBM and Heston QE, Sobol, antithetics |
| `lsm.py` | Longstaff-Schwartz American exercise, Black-Scholes, control variates |
| `calibration.py` | Fitting a model to the observed market chain |
| `greeks.py` | Pathwise, likelihood-ratio, and adaptive finite-difference Greeks |
| `pnl.py` | Real-world P&L forecasting |
| `validation.py` | Validation checks, convergence sweeps, stability reports |
| `plots.py` | Diagnostic plots |
| `reports.py` | Human-readable pricing and forecast reports |
| `pricer.py` | End-to-end pipeline |
| `cli.py` | Interactive entry point |

## Pipeline

1. **Market data** — spot, IV surface, dividends, Treasury yields
2. **Calendar** — NYSE trading days between valuation date and expiry
3. **Calibration** — GBM volatility, or Heston parameters fit to the smile
4. **Simulation** — GBM or Heston QE paths with Sobol QMC
5. **LSM pricing** — backward induction on a Laguerre basis
6. **Control variates** — optimal beta against the Black-Scholes European
7. **Greeks** — adaptive bumps with Richardson extrapolation
8. **Validation** — intrinsic bounds, convergence, European equivalence

## Documentation

[`docs/technical-notes.md`](docs/technical-notes.md) walks through the mathematics behind each
stage — the implied-volatility inversion, the Feller condition, the QE scheme, the LSM
regression, and the Q-to-P measure change — with the equations as implemented.

## Notes and limitations

- Market data quality is whatever `yfinance` returns. Thin or wide-spread chains produce
  unreliable IV surfaces; `prepare_iv_data` filters by volume, open interest, spread, and
  moneyness (default 0.8–1.2), but garbage in still applies.
- Heston calibration is materially slower than GBM. Use GBM for iteration, Heston when the
  volatility smile matters.
- The borrow rate defaults to zero. Hard-to-borrow names will misprice.
- This is a personal research project, not investment advice, and not a production trading system.

## License

MIT — see [LICENSE](LICENSE).
