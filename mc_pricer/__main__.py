"""Command-line entry point: ``python -m mc_pricer [TICKER] [call|put]``."""

import sys
import warnings

from .cli import price_option


def main() -> None:
    # Model calibration and yfinance both emit a lot of benign noise on the
    # happy path; the interactive CLI is unreadable without this.
    warnings.filterwarnings("ignore")

    if len(sys.argv) > 1:
        price_option(" ".join(sys.argv[1:]))
    else:
        price_option()


if __name__ == "__main__":
    main()
