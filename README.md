# Monte Carlo Options Pricer

Prices American equity options by least-squares Monte Carlo against live market data.

It pulls the option chain, dividend history and Treasury yields from yfinance, bootstraps a real
discount curve instead of assuming a flat rate, and calibrates either GBM volatility or Heston
parameters to the full implied volatility smile. Early exercise is valued with Longstaff-Schwartz
regression on a Laguerre basis, stepping through an actual NYSE calendar rather than 252 evenly
spaced days.

Heston paths use Andersen's QE discretization. Variance reduction is Sobol QMC, antithetics, and
a Black-Scholes control variate fitted with an optimal beta. Delta comes from a pathwise
estimator and vega from a likelihood ratio; the rest fall back to adaptive finite differences
with Richardson extrapolation, reported next to a stability check across bump sizes so you can
see when a Greek isn't worth trusting.

## Running it

```bash
pip install -r requirements.txt
python -m mc_pricer AAPL call
```

```python
from datetime import datetime
from mc_pricer import run_pricer

result = run_pricer(ticker="AAPL", spot=225.0, strike=230.0,
                    expiry=datetime(2026, 1, 16), option_type="call", model="heston")

print(result["pricing_Q"]["price"], result["greeks"]["delta"])
```

Validation covers intrinsic bounds, European equivalence and convergence sweeps. There's also a
real-world P&L forecast under the fitted exercise policy, kept separate from the risk-neutral
price since it answers a different question.

`docs/technical-notes.md` works through the math: the IV inversion, the Feller condition, the QE
scheme, and the Q-to-P measure change.

Market data is whatever yfinance returns. Thin chains produce unreliable surfaces, so
`prepare_iv_data` filters on volume, spread and moneyness before anything is calibrated.
