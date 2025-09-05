import sys

sys.path.append("./src")

__doc__ = """Run a vectorized trading environment with a single dataset.

This example downloads market data, engineers a minimal set of features, and
launches a vectorized `TradingEnv` using Gymnasium's vector API. It then
executes a simple loop with fixed actions to demonstrate environment stepping.
"""

import datetime

import gymnasium as gym
import numpy as np
import pandas as pd

from gym_trading_env.downloader import download

download(
    exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="30m",
    dir="examples/data",
    since=datetime.datetime(year=2019, month=1, day=1),
)

# Import your datas
df = pd.read_pickle("examples/data/binance-BTCUSDT-30m.pkl")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Generating features
# WARNING : the column names need to contain keyword 'feature' !
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
df.dropna(inplace=True)

print(df)


def reward_function(history):
    """Compute the log-return between the last two portfolio valuations.

    Args:
        history (Any): A history accessor that supports tuple-style access
            like `history["portfolio_valuation", -1]` to retrieve the most
            recent and previous portfolio valuations.

    Returns:
        float: The logarithmic return, i.e., `log(p_t / p_{t-1})`.

    Raises:
        KeyError: If the required keys are not present in `history`.
        IndexError: If the history does not contain at least two entries.
        TypeError: If the accessed values do not support numeric operations.
    """
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )  # log (p_t / p_t-1 )


if __name__ == "__main__":
    env = gym.make_vec(
        id="TradingEnv",
        num_envs=3,
        name="BTCUSD",
        df=df,
        windows=5,
        positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
        initial_position=0,
        trading_fees=0.01 / 100,
        borrow_interest_rate=0.0003 / 100,
        reward_function=reward_function,
        portfolio_initial_value=1000,
    )
    # Run the simulation
    observation, info = env.reset()
    while True:
        actions = [1, 2, 3]
        observation, reward, done, truncated, info = env.step(actions)
