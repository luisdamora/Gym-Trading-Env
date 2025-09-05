"""Asynchronous OHLCV data downloader utilities.

This module provides helper coroutines to download OHLCV data from multiple
exchanges supported by ``ccxt``. It handles rate limits per exchange,
splits long time ranges into smaller windows, and saves results as pickles.

Example:
    The easiest way to use the downloader is via the ``download`` function:

    ```python
    import datetime as dt
    from gym_trading_env.downloader import download

    download(
        exchange_names=["binance"],
        symbols=["BTC/USDT"],
        timeframe="1h",
        dir="data",
        since=dt.datetime(2021, 1, 1),
        until=dt.datetime(2021, 6, 1),
    )
    ```
"""

import asyncio
import datetime
import sys

import ccxt.async_support as ccxt
import nest_asyncio
import pandas as pd

nest_asyncio.apply()


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

EXCHANGE_LIMIT_RATES = {
    "bitfinex2": {
        "limit": 10_000,
        "pause_every": 1,
        "pause": 3,  # seconds
    },
    "binance": {
        "limit": 1_000,
        "pause_every": 10,
        "pause": 1,  # seconds
    },
    "huobi": {
        "limit": 1_000,
        "pause_every": 10,
        "pause": 1,  # seconds
    },
}


async def _ohlcv(exchange, symbol, timeframe, limit, step_since, timedelta):
    """Fetch one OHLCV window and return it as a formatted DataFrame.

    Args:
        exchange: An instantiated ``ccxt`` exchange with async support.
        symbol (str): Trading pair symbol (e.g., ``"BTC/USDT"``).
        timeframe (str): CCXT timeframe string (e.g., ``"1h"``, ``"5m"``).
        limit (int): Max number of candles to fetch in a single call.
        step_since (int): Millisecond timestamp from which to start fetching.
        timedelta (int): Number of milliseconds per candle for the timeframe.

    Returns:
        pandas.DataFrame: A DataFrame with columns
        ``[timestamp_open, open, high, low, close, volume, date_open, date_close]``.
    """
    result = await exchange.fetch_ohlcv(
        symbol=symbol, timeframe=timeframe, limit=limit, since=step_since
    )
    result_df = pd.DataFrame(
        result, columns=["timestamp_open", "open", "high", "low", "close", "volume"]
    )
    for col in ["open", "high", "low", "close", "volume"]:
        result_df[col] = pd.to_numeric(result_df[col])
    result_df["date_open"] = pd.to_datetime(result_df["timestamp_open"], unit="ms")
    result_df["date_close"] = pd.to_datetime(result_df["timestamp_open"] + timedelta, unit="ms")

    return result_df


async def _download_symbol(
    exchange,
    symbol,
    timeframe="5m",
    since=None,
    until=None,
    limit=1000,
    pause_every=10,
    pause=1,
):
    """Download OHLCV data for a single symbol from a given exchange.

    The function chunks the time interval into windows sized by ``limit`` and
    ``timeframe``, schedules async tasks accordingly, and concatenates the
    results.

    Args:
        exchange: A configured async ``ccxt`` exchange instance.
        symbol (str): Trading pair symbol (e.g., ``"BTC/USDT"``).
        timeframe (str): CCXT timeframe string. Defaults to ``"5m"``.
        since (int | None): Millisecond timestamp (inclusive) to start from. If
            ``None``, defaults to Jan 1, 2020.
        until (int | None): Millisecond timestamp (exclusive) to stop at. If
            ``None``, defaults to now.
        limit (int): Max candles per request. Defaults to 1000.
        pause_every (int): After scheduling this many requests, pause for
            ``pause`` seconds. Helps with rate limits.
        pause (int | float): Pause duration in seconds between request batches.

    Returns:
        pandas.DataFrame: Cleaned OHLCV DataFrame indexed by ``date_open`` and
        sorted by index without duplicates.
    """
    if since is None:
        since = int(datetime.datetime(year=2020, month=1, day=1).timestamp() * 1e3)
    if until is None:
        until = int(datetime.datetime.now().timestamp() * 1e3)
    timedelta = int(pd.Timedelta(timeframe).to_timedelta64() / 1e6)
    tasks = []
    results = []
    for step_since in range(since, until, limit * timedelta):
        tasks.append(
            asyncio.create_task(_ohlcv(exchange, symbol, timeframe, limit, step_since, timedelta))
        )
        if len(tasks) >= pause_every:
            results.extend(await asyncio.gather(*tasks))
            await asyncio.sleep(pause)
            tasks = []
    if len(tasks) > 0:
        results.extend(await asyncio.gather(*tasks))
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.loc[
        (since < final_df["timestamp_open"]) & (final_df["timestamp_open"] < until), :
    ]
    del final_df["timestamp_open"]
    final_df.set_index("date_open", drop=True, inplace=True)
    final_df.sort_index(inplace=True)
    final_df.dropna(inplace=True)
    final_df.drop_duplicates(inplace=True)
    return final_df


async def _download_symbols(exchange_name, symbols, dir, timeframe, **kwargs):
    """Download OHLCV data for multiple symbols from a single exchange.

    Args:
        exchange_name (str): The CCXT exchange id (e.g., ``"binance"``).
        symbols (list[str]): List of symbols to download.
        dir (str): Directory where results will be saved as pickles.
        timeframe (str): CCXT timeframe string.
        **kwargs: Forwarded to ``_download_symbol`` (e.g., ``limit``, ``since``).

    Returns:
        None
    """
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    for symbol in symbols:
        df = await _download_symbol(exchange=exchange, symbol=symbol, timeframe=timeframe, **kwargs)
        save_file = f"{dir}/{exchange_name}-{symbol.replace('/', '')}-{timeframe}.pkl"
        print(f"{symbol} downloaded from {exchange_name} and stored at {save_file}")
        df.to_pickle(save_file)
    await exchange.close()


async def _download(
    exchange_names,
    symbols,
    timeframe,
    dir,
    since: datetime.datetime,
    until: datetime.datetime = None,
):
    """Coordinate downloads across multiple exchanges concurrently.

    For each exchange, configures rate-limit parameters based on
    ``EXCHANGE_LIMIT_RATES`` and schedules symbol downloads.

    Args:
        exchange_names (list[str]): CCXT exchange ids to use.
        symbols (list[str]): Trading symbols to download for each exchange.
        timeframe (str): CCXT timeframe string.
        dir (str): Destination directory for pickles.
        since (datetime.datetime): Start datetime (inclusive).
        until (datetime.datetime | None): End datetime (exclusive). Defaults to now.

    Returns:
        None
    """
    if until is None:
        until = datetime.datetime.now()
    tasks = []
    for exchange_name in exchange_names:
        limit = EXCHANGE_LIMIT_RATES[exchange_name]["limit"]
        pause_every = EXCHANGE_LIMIT_RATES[exchange_name]["pause_every"]
        pause = EXCHANGE_LIMIT_RATES[exchange_name]["pause"]
        tasks.append(
            _download_symbols(
                exchange_name=exchange_name,
                symbols=symbols,
                timeframe=timeframe,
                dir=dir,
                limit=limit,
                pause_every=pause_every,
                pause=pause,
                since=int(since.timestamp() * 1e3),
                until=int(until.timestamp() * 1e3),
            )
        )
    await asyncio.gather(*tasks)


def download(*args, **kwargs):
    """Synchronous wrapper to run the asynchronous download pipeline.

    This function starts a fresh event loop to run ``_download``.

    Args:
        *args: Forwarded to ``_download``.
        **kwargs: Forwarded to ``_download``.

    Returns:
        None
    """
    # loop = asyncio.get_event_loop()
    asyncio.run(_download(*args, **kwargs))


async def main():
    """Example entry point to trigger a sample data download.

    Note:
        Intended for manual testing. When executed as a script, downloads two
        symbols across several exchanges.

    Returns:
        None
    """
    await _download(
        ["binance", "bitfinex2", "huobi"],
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="30m",
        dir="test/data",
        since=datetime.datetime(year=2019, month=1, day=1),
    )


if __name__ == "__main__":
    asyncio.run(main())
