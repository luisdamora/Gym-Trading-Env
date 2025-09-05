<h1 align='center'>
   <img src = 'https://github.com/ClementPerroud/Gym-Trading-Env/raw/main/docs/source/images/logo_light-bg.png' width='500'>
</h1>

<section class="shields" align="center">
   <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
         alt="python">
   </a>
   <a href="https://pypi.org/project/gym-trading-env/">
      <img src="https://img.shields.io/badge/pypi-v1.1.3-brightgreen.svg"
         alt="PyPI">
   </a>
   <a href="https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/LICENSE.txt">
   <img src="https://img.shields.io/badge/license-MIT%202.0%20Clause-green"
         alt="Apache 2.0 with Commons Clause">
   </a>
   <a href='https://gym-trading-env.readthedocs.io/en/latest/?badge=latest'>
         <img src='https://readthedocs.org/projects/gym-trading-env/badge/?version=latest' alt='Documentation Status' />
   </a>
   <a href="https://github.com/ClementPerroud/Gym-Trading-Env">
      <img src="https://img.shields.io/github/stars/ClementPerroud/gym-trading-env?style=social" alt="Github stars">
   </a>
</section>
  
# Gym Trading Env

Gymnasium-based environment to simulate markets and train Reinforcement Learning (RL) agents for trading. It is designed to be fast, easy to use, and highly customizable.

| [Documentation](https://gym-trading-env.readthedocs.io/en/latest/index.html) |

## Key features

- Fast OHLCV data download from multiple exchanges via `ccxt`.
- Simple and fast environments supporting advanced operations (shorting, borrow interest, fees) with continuous or discrete positions.
- High-performance web renderer to visualize thousands of candles, positions, and custom metrics.
- Support for multiple datasets and vectorized execution with the Gymnasium API.

Rendering example:

![Render animated image](https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/docs/source/images/render.gif)

## Registered environment IDs

Environments are registered in `src/gym_trading_env/__init__.py` and can be created with `gym.make(...)` or `gym.make_vec(...)`:

- `TradingEnv`: environment for a single dataset.
- `MultiDatasetTradingEnv`: environment that iterates over multiple preprocessed datasets.

## Installation

Requires Python 3.9+ (Linux, macOS or Windows).

- Option 1 (PyPI):

```bash
pip install gym-trading-env
```

- Option 2 (clone the repo):

```bash
git clone https://github.com/ClementPerroud/Gym-Trading-Env
cd Gym-Trading-Env
```

If you use Poetry to manage this project's dependencies:

```bash
poetry install
```

## Input data format

DataFrames must have a `DatetimeIndex` and at least the columns: `open`, `high`, `low`, `close`, and `volume` (or `Volume USD` in some examples). Features that will be part of the observation must include the word `feature` in their name, e.g., `feature_close`, `feature_volume`, etc.

## Quick example (single dataset)

```python
import gymnasium as gym
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("examples/data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date").sort_index()
df = df.dropna().drop_duplicates()

# Create features (note: names must contain the word 'feature')
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()
df.dropna(inplace=True)

def reward_function(history):
    # Reward: log-return of the portfolio value
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position="random",
    trading_fees=0.01 / 100,
    borrow_interest_rate=0.0003 / 100,
    reward_function=reward_function,
    portfolio_initial_value=1000,
    max_episode_duration=500,
)

obs, info = env.reset()
done, truncated = False, False
while not done and not truncated:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

# Save logs for the renderer
env.save_for_render()
```

See `examples/example_environnement.py` for a complete script.

## Vectorized execution (single dataset)

```python
import gymnasium as gym
import numpy as np
import pandas as pd

# Prepare df and reward_function as in the previous example...

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

obs, info = env.reset()
while True:
    actions = [1, 2, 3]
    obs, reward, done, truncated, info = env.step(actions)
```

See `examples/example_vectorized_environment.py`.

## Multiple datasets (with preprocessing)

```python
import gymnasium as gym
import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame):
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df.dropna(inplace=True)
    return df

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

env = gym.make(
    "MultiDatasetTradingEnv",
    dataset_dir="examples/data/*.pkl",
    preprocess=preprocess,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position=0,
    trading_fees=0.01 / 100,
    borrow_interest_rate=0.0003 / 100,
    reward_function=reward_function,
    portfolio_initial_value=1000,
    max_episode_duration=500,
    episodes_between_dataset_switch=10,
)
```

Vectorized versions: `examples/example_vectorized_multi_environment.py`.

## Data download (ccxt)

Use `gym_trading_env.downloader.download` to download OHLCV and save ready-to-use `.pkl` files:

```python
import datetime as dt
from gym_trading_env.downloader import EXCHANGE_LIMIT_RATES, download

# Optional: adjust limits per exchange
EXCHANGE_LIMIT_RATES["bybit"] = {"limit": 200, "pause_every": 120, "pause": 2}

download(
    exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="1h",
    dir="examples/data",
    since=dt.datetime(2023, 1, 1),
)
```

See `examples/example_download.py`.

## Web renderer

```python
from gym_trading_env.renderer import Renderer

renderer = Renderer(render_logs_dir="render_logs")

# Custom lines and metrics (optional)
# renderer.add_line("sma10", lambda df: df["close"].rolling(10).mean(), {"width": 1, "color": "purple"})
# renderer.add_metric("Annual Market Return", lambda df: "...")

renderer.run()
```

See `examples/example_render.py`.

## How to run the examples

From the repo root:

```bash
# With Poetry (recommended in this repo)
poetry run python examples/example_environnement.py
poetry run python examples/example_vectorized_environment.py
poetry run python examples/example_multi_environnement.py
poetry run python examples/example_vectorized_multi_environment.py
poetry run python examples/example_render.py
```

You can also run with `python` directly if you already have the dependencies installed in your current environment.

## Relevant project structure

- `src/gym_trading_env/`: main implementation
  - `environments.py`: `TradingEnv` and `MultiDatasetTradingEnv` classes
  - `downloader.py`: asynchronous utilities to download OHLCV (`ccxt`)
  - `renderer.py`: Flask server to visualize sessions
  - `utils/`: history, portfolio, and plotting utilities
- `examples/`: ready-to-run scripts covering download, execution, vectorization, and rendering

## License

This project is distributed under the MIT license. See `LICENSE.txt` for details.

[Documentation available here](https://gym-trading-env.readthedocs.io/en/latest/index.html)
