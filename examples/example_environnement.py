"""Single-environment example using `TradingEnv`.

This example loads a BTC/USD hourly dataset, engineers basic features,
defines a simple log-return reward function, and runs a short episode in
the `TradingEnv`. At the end, it saves logs for later rendering.

Args:
    None: This module is intended to be executed directly.

Returns:
    None: Prints observations and saves render logs to disk.

Raises:
    Exception: Any exception from data loading, preprocessing, or the
        environment interaction will propagate.
"""

import sys

sys.path.append("./src")


import gymnasium as gym
import numpy as np
import pandas as pd

# Import your datas
df = pd.read_csv("examples/data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Generating features
# WARNING : the column names need to contain keyword 'feature' !
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()
df.dropna(inplace=True)


# Create your own reward function with the history object
def reward_function(history):
    """Compute step reward as the log change of portfolio valuation.

    Args:
        history: A history-like structure exposing the `portfolio_valuation`
            time series that can be indexed with `["portfolio_valuation", idx]`.

    Returns:
        float: The log return between the last and the previous step.

    Raises:
        KeyError: If `"portfolio_valuation"` is missing in `history`.
        IndexError: If the last or previous entries are not available.
    """
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )  # log (p_t / p_t-1 )


env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],  # From -1 (=SHORT), to +1 (=LONG)
    initial_position="random",  # Initial position
    trading_fees=0.01 / 100,  # 0.01% per stock buy / sell
    borrow_interest_rate=0.0003 / 100,  # per timestep (= 1h here)
    reward_function=reward_function,
    portfolio_initial_value=1000,  # in FIAT (here, USD)
    max_episode_duration=500,
    disable_env_checker=True,
)

env.add_metric("Position Changes", lambda history: np.sum(np.diff(history["position"]) != 0))
env.add_metric("Episode Lenght", lambda history: len(history["position"]))

done, truncated = False, False
observation, info = env.reset()
print(info)
while not done and not truncated:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(observation)
# Save for render
env.save_for_render()
