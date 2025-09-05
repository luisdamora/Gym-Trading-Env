import sys

sys.path.append("./src")

__doc__ = """Run a vectorized trading environment across multiple datasets.

This example demonstrates `MultiDatasetTradingEnv` with a preprocessing
function to engineer features for each dataset discovered in a directory.
It runs a simple stepping loop with fixed actions for illustration purposes.
"""


import gymnasium as gym
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame):
    """Add basic technical features required by the environment.

    The produced columns must include the keyword "feature" in their names
    to be recognized by the environment as part of the observation space.

    Args:
        df (pd.DataFrame): Input OHLCV dataframe with columns such as
            `open`, `high`, `low`, `close`, and `volume`.

    Returns:
        pd.DataFrame: The same dataframe with added feature columns and
        cleaned of NaNs after feature computation.
    """
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df.dropna(inplace=True)

    return df


def reward_function(history):
    """Compute the log-return between the last two portfolio valuations.

    Args:
        history (Any): A history accessor that supports tuple-style indexing
            like `history["portfolio_valuation", -1]`.

    Returns:
        float: The logarithmic return, i.e., `log(p_t / p_{t-1})`.

    Raises:
        KeyError: If the required keys are missing.
        IndexError: If fewer than two entries are available.
        TypeError: If the values cannot be used in numeric operations.
    """
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )  # log (p_t / p_t-1 )


if __name__ == "__main__":
    # Uncomment if needed
    # download(
    #     exchange_names = ["binance", "bitfinex2", "huobi"],
    #     symbols= ["BTC/USDT", "ETH/USDT"],
    #     timeframe= "30m",
    #     dir = "examples/data",
    #     since= datetime.datetime(year= 2019, month= 1, day=1),
    # )
    env = gym.make_vec(
        id="MultiDatasetTradingEnv",
        num_envs=3,
        dataset_dir="examples/data/*.pkl",
        preprocess=add_features,
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
