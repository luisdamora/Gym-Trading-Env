"""Multi-dataset environment example using `MultiDatasetTradingEnv`.

This script demonstrates how to use a directory of preprocessed datasets
with a custom `preprocess` function and a simple log-return `reward_function`.
It runs episodes continuously, switching datasets every N episodes as
configured in the environment.

Args:
    None: This module is intended to be executed directly.

Returns:
    None: Runs an infinite loop of episodes; prints nothing by default.

Raises:
    Exception: Any error in data loading, preprocessing, or environment
        interaction may propagate.
"""

import sys

sys.path.append("./src")

import gymnasium as gym
import numpy as np
import pandas as pd

# WARNING : Please use the example_download.py script to download the data before!


# Generating features
# WARNING : the column names need to contain keyword 'feature' !
def preprocess(df: pd.DataFrame):
    """Add basic technical features required by the environment.

    The environment expects feature columns to contain the keyword
    `feature` in their names.

    Args:
        df (pd.DataFrame): Raw OHLCV-like dataframe with columns such as
            `open`, `high`, `low`, `close`, and `volume`.

    Returns:
        pd.DataFrame: The input dataframe with added feature columns and
        NaNs removed.
    """
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df.dropna(inplace=True)
    return df


# Create your own reward function with the history object
def reward_function(history):
    """Compute the per-step log return of the portfolio valuation.

    Args:
        history: A history-like structure exposing `portfolio_valuation`.

    Returns:
        float: Log of the ratio between the last and previous valuations.

    Raises:
        KeyError: If `portfolio_valuation` is not present.
        IndexError: If not enough history is available.
    """
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )  # log (p_t / p_t-1 )


env = gym.make(
    "MultiDatasetTradingEnv",
    dataset_dir="./examples/data/*.pkl",
    preprocess=preprocess,
    windows=5,
    positions=[
        -1,
        -0.5,
        0,
        0.5,
        1,
        1.5,
        2,
    ],  # From -1 (=full SHORT), to +1 (=full LONG) with 0 = no position
    initial_position=0,  # Initial position
    trading_fees=0.01 / 100,  # 0.01% per stock buy / sell
    borrow_interest_rate=0.0003 / 100,  # per timestep (= 1h here)
    reward_function=reward_function,
    portfolio_initial_value=1000,  # in FIAT (here, USD)
    max_episode_duration=500,
    episodes_between_dataset_switch=10,
)

# Run the simulation
while True:
    truncated = False
    observation, info = env.reset()
    while not truncated:
        action = env.action_space.sample()  # OR manually : action = int(input("Action : "))
        observation, reward, done, truncated, info = env.step(action)

# Render
# env.save_for_render()
