"""Monte Carlo American options pricer for US equity options.

Prices American-style equity options with live market data, using
Longstaff-Schwartz least-squares Monte Carlo under either a GBM or a
Heston stochastic-volatility model.

Typical use:

    from datetime import datetime
    from mc_pricer import run_pricer

    result = run_pricer(
        ticker="AAPL",
        spot=225.0,
        strike=230.0,
        expiry=datetime(2026, 1, 16),
        option_type="call",
    )

Or from the command line:

    python -m mc_pricer AAPL call
"""

from .config import BASIS_DEGREE, CONFIDENCE_LEVEL, N_PATHS_DEFAULT, RANDOM_SEED
from .models import HestonParams, calibrate_gbm_vol, calibrate_heston_params
from .simulation import simulate_paths_gbm, simulate_paths_heston_qe
from .lsm import black_scholes_price, price_american_lsm
from .calibration import calibrate_model_to_market
from .greeks import compute_greeks_t0
from .pnl import compute_real_world_pnl
from .validation import convergence_sweep, run_validation_checks
from .pricer import run_pricer
from .cli import price_option

__version__ = "1.0.0"

__all__ = [
    "BASIS_DEGREE",
    "CONFIDENCE_LEVEL",
    "N_PATHS_DEFAULT",
    "RANDOM_SEED",
    "HestonParams",
    "black_scholes_price",
    "calibrate_gbm_vol",
    "calibrate_heston_params",
    "calibrate_model_to_market",
    "compute_greeks_t0",
    "compute_real_world_pnl",
    "convergence_sweep",
    "price_american_lsm",
    "price_option",
    "run_pricer",
    "run_validation_checks",
    "simulate_paths_gbm",
    "simulate_paths_heston_qe",
]
