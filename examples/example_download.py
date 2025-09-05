"""Download historical market data examples.

This script demonstrates how to configure exchange rate limits and download
historical candlestick data for one or more symbols using the downloader
utilities provided by `gym_trading_env`.

It modifies `EXCHANGE_LIMIT_RATES` for the `bybit` exchange to control
pagination and pauses, then calls `download()` with a set of parameters.

Args:
    None: The script is intended to be executed directly. Adjust parameters
        in the call to `download()` as needed.

Returns:
    None: Outputs data files under the target directory.

Raises:
    Exception: Propagates any exception raised by the underlying downloader
        or network operations.
"""

import sys

sys.path.append("./src")


import datetime

from gym_trading_env.downloader import EXCHANGE_LIMIT_RATES, download

EXCHANGE_LIMIT_RATES["bybit"] = {
    "limit": 200,  # One request will query 1000 data points (aka candlesticks)
    "pause_every": 120,  # it will pause every 10 request
    "pause": 2,  # the pause will last 1 second
}
download(
    exchange_names=["bybit"],
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="1h",
    dir="examples/data",
    since=datetime.datetime(year=2023, month=1, day=1),
)
