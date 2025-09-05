import datetime
import glob
import os
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .utils.history import History
from .utils.portfolio import TargetPortfolio

warnings.filterwarnings("error")


def basic_reward_function(history: History):
    """Compute the default reward as log return of portfolio valuation.

    The reward is computed as the natural logarithm of the ratio between the
    last portfolio valuation and the previous one. This favors smooth
    compounding behavior and is a common choice in trading environments.

    Args:
        history (History): The history buffer containing tracked environment
            values. Must support indexing like ``history["portfolio_valuation", -1]``.

    Returns:
        float: The log return between the last and previous portfolio valuation.

    Raises:
        KeyError: If the required keys are not present in history.
        IndexError: If there are not enough entries in history to compute the
            reward.
    """
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


def dynamic_feature_last_position_taken(history):
    """Return the last position taken by the agent.

    Args:
        history (History): The history buffer that stores the last position
            selected by the agent.

    Returns:
        float | int: The last position value recorded in history.

    Raises:
        KeyError: If the "position" key is missing in history.
        IndexError: If history does not contain any entries yet.
    """
    return history["position", -1]


def dynamic_feature_real_position(history):
    """Return the real (effective) position of the portfolio.

    This may differ from the target position due to price fluctuations or
    trading costs.

    Args:
        history (History): The history buffer that stores the effective
            position.

    Returns:
        float: The current effective position of the portfolio.

    Raises:
        KeyError: If the "real_position" key is missing in history.
        IndexError: If history does not contain any entries yet.
    """
    return history["real_position", -1]


class TradingEnv(gym.Env):
    """
    An easy trading environment for OpenAI gym. It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('TradingEnv', ...)


    :param df: The market DataFrame. It must contain 'open', 'high', 'low', 'close'.
        Index must be DatetimeIndex. Your desired inputs need to contain 'feature'
        in their column name: this way, they will be returned as observation at each step.
    :type df: pandas.DataFrame

    :param positions: List of the positions allowed by the environment.
    :type positions: optional - list[int or float]

    :param dynamic_feature_functions: The list of the dynamic features functions.
        By default, two dynamic features are added:

        * the last position taken by the agent.
        * the real position of the portfolio (that varies according to the price fluctuations)

    :type dynamic_feature_functions: optional - list

    :param reward_function: Take the History object of the environment and must return a float.
    :type reward_function: optional - function<History->float>

    :param windows: Default is None. If it is set to an int: N, every step observation
        will return the past N observations. It is recommended for Recurrent Neural
        Network based Agents.
    :type windows: optional - None or int

    :param trading_fees: Transaction trading fees (buy and sell operations).
        eg: 0.01 corresponds to 1% fees
    :type trading_fees: optional - float

    :param borrow_interest_rate: Borrow interest rate per step (only when position < 0
        or position > 1). eg: 0.01 corresponds to 1% borrow interest rate per STEP;
        if you know that your borrow interest rate is 0.05% per day and your timestep
        is 1 hour, you need to divide it by 24 -> 0.05/100/24.
    :type borrow_interest_rate: optional - float

    :param portfolio_initial_value: Initial valuation of the portfolio.
    :type portfolio_initial_value: float or int

    :param initial_position: You can specify the initial position of the environment
        or set it to 'random'. It must be contained in the list parameter 'positions'.
    :type initial_position: optional - float or int

    :param max_episode_duration: If an integer value is used, each episode will be
        truncated after reaching the desired max duration in steps (by returning
        `truncated` as `True`). When using a max duration, each episode will start at
        a random starting point.
    :type max_episode_duration: optional - int or 'max'

    :param verbose: If 0, no log is outputted. If 1, the env send episode result logs.
    :type verbose: optional - int

    :param name: The name of the environment (eg. 'BTC/USDT')
    :type name: optional - str

    """

    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list = None,
        dynamic_feature_functions=None,
        reward_function=basic_reward_function,
        windows=None,
        trading_fees=0,
        borrow_interest_rate=0,
        portfolio_initial_value=1000,
        initial_position="random",
        max_episode_duration="max",
        verbose=1,
        name="Stock",
        render_mode="logs",
    ):
        """Initialize the trading environment.

        Args:
            df (pandas.DataFrame): Market data with required columns (e.g.,
                ``open``, ``high``, ``low``, ``close``). Index should be a
                ``DatetimeIndex``. Columns containing the substring
                "feature" will be used as observations.
            positions (list, optional): Allowed discrete positions (e.g.,
                ``[0, 1]`` or ``[-1, 0, 1]``). Defaults to ``[0, 1]`` if not
                provided.
            dynamic_feature_functions (list, optional): List of callables that
                map ``History`` to a float feature appended to the observation.
                Defaults to two built-ins: last position and real position.
            reward_function (callable): Function mapping ``History`` to a float
                reward. Defaults to ``basic_reward_function``.
            windows (int | None, optional): If provided, the observation stacks
                the last ``windows`` frames. Useful for RNN-based agents.
            trading_fees (float, optional): Proportional transaction fee applied
                on trades. Example: ``0.01`` for 1% fees. Defaults to 0.
            borrow_interest_rate (float, optional): Per-step borrow interest
                rate applied when leverage-like positions are held. Defaults to 0.
            portfolio_initial_value (float | int, optional): Initial portfolio
                valuation. Defaults to 1000.
            initial_position (float | int | str, optional): Initial position or
                ``"random"`` to sample from ``positions``. Defaults to
                ``"random"``.
            max_episode_duration (int | str, optional): Truncate episode after
                this many steps. Use ``"max"`` to run to dataset end. Defaults to
                ``"max"``.
            verbose (int, optional): If > 0, prints metrics at episode end.
                Defaults to 1.
            name (str, optional): Environment name (e.g., ``"BTC/USDT"``).
                Defaults to ``"Stock"``.
            render_mode (str | None, optional): Rendering mode. Only "logs" is
                supported. Defaults to "logs".

        Raises:
            AssertionError: If ``initial_position`` is invalid or ``render_mode``
                is not supported.
        """
        if positions is None:
            positions = [0, 1]
        if dynamic_feature_functions is None:
            dynamic_feature_functions = [
                dynamic_feature_last_position_taken,
                dynamic_feature_real_position,
            ]
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        assert self.initial_position in self.positions or self.initial_position == "random", (
            "The 'initial_position' parameter must be 'random' or a position mentionned "
            "in the 'position' (default is [0, 1]) parameter."
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode
        self._set_df(df)

        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._nb_features])
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=[self.windows, self._nb_features]
            )

        self.log_metrics = []

    def _set_df(self, df):
        """Prepare and cache arrays and columns from the input dataset.

        Creates internal arrays for observations, info features and prices, and
        augments the DataFrame with dynamic feature placeholders.

        Args:
            df (pandas.DataFrame): The market data. Must include a ``close``
                column and any number of columns tagged as features by
                including the substring "feature" in their names.

        Returns:
            None: This method mutates internal state.
        """
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype=np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])

    def _get_ticker(self, delta=0):
        """Get the row in the dataset at the current index plus a delta.

        Args:
            delta (int): Offset from the current index. Defaults to 0.

        Returns:
            pandas.Series: The row representing the ticker information.
        """
        return self.df.iloc[self._idx + delta]

    def _get_price(self, delta=0):
        """Return the close price at the current index plus a delta.

        Args:
            delta (int): Offset from the current index. Defaults to 0.

        Returns:
            float: The price value.
        """
        return self._price_array[self._idx + delta]

    def _get_obs(self):
        """Build the observation vector (and dynamic features) for the step.

        Returns the observation at the current index, optionally stacking the
        last ``windows`` frames if configured.

        Returns:
            numpy.ndarray: Observation array with static and dynamic features.
        """
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(
                self.historical_info
            )

        if self.windows is None:
            _step_index = self._idx
        else:
            _step_index = np.arange(self._idx + 1 - self.windows, self._idx + 1)
        return self._obs_array[_step_index]

    def reset(self, seed=None, options=None, **kwargs):
        """Reset the environment state at the beginning of an episode.

        This initializes portfolio, positions, indices, and the history buffer.

        Args:
            seed (int | None): Random seed for reproducibility. Optional.
            options (dict | None): Additional options for Gymnasium. Optional.
            **kwargs: Forwarded to ``gym.Env.reset``.

        Returns:
            tuple[numpy.ndarray, dict]: The initial observation and initial info
            dict (first entry of history).
        """
        super().reset(seed=seed, options=options, **kwargs)

        self._step = 0
        self._position = (
            np.random.choice(self.positions)
            if self.initial_position == "random"
            else self.initial_position
        )
        self._limit_orders = {}

        self._idx = 0
        if self.windows is not None:
            self._idx = self.windows - 1
        if self.max_episode_duration != "max":
            self._idx = np.random.randint(
                low=self._idx, high=len(self.df) - self.max_episode_duration - self._idx
            )

        self._portfolio = TargetPortfolio(
            position=self._position, value=self.portfolio_initial_value, price=self._get_price()
        )

        self.historical_info = History(max_size=len(self.df))
        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._position,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio_initial_value,
            portfolio_distribution=self._portfolio.get_portfolio_distribution(),
            reward=0,
        )

        return self._get_obs(), self.historical_info[0]

    def render(self):
        """Render the environment.

        Note:
            Rendering is not implemented for this environment. Use
            ``save_for_render`` in conjunction with the renderer utility.

        Returns:
            None: No rendering is performed.
        """
        pass

    def _trade(self, position, price=None):
        """Execute a trade to move the portfolio to a target position.

        Args:
            position (float | int): Target position to set in the portfolio.
            price (float | None): Execution price. If ``None``, uses current
                market price from the dataset.

        Returns:
            None: The internal portfolio state is updated.
        """
        self._portfolio.trade_to_position(
            position,
            price=self._get_price() if price is None else price,
            trading_fees=self.trading_fees,
        )
        self._position = position
        return

    def _take_action(self, position):
        """Update position by trading if the requested position differs.

        Args:
            position (float | int): Desired position.

        Returns:
            None
        """
        if position != self._position:
            self._trade(position)

    def _take_action_order_limit(self):
        """Evaluate and execute pending limit orders if their conditions meet.

        Iterates over stored limit orders and triggers trades when the current
        ticker's high/low encompasses the limit price. Non-persistent orders are
        removed once executed.

        Returns:
            None
        """
        if len(self._limit_orders) > 0:
            ticker = self._get_ticker()
            for position, params in self._limit_orders.items():
                if (
                    position != self._position
                    and params["limit"] <= ticker["high"]
                    and params["limit"] >= ticker["low"]
                ):
                    self._trade(position, price=params["limit"])
                    if not params["persistent"]:
                        del self._limit_orders[position]

    def add_limit_order(self, position, limit, persistent=False):
        """Add a limit order to be executed when price reaches a level.

        Args:
            position (float | int): Target position when the order is triggered.
            limit (float): The limit price for execution.
            persistent (bool): If ``True``, the order remains after execution
                (useful for grid-like strategies). Defaults to ``False``.

        Returns:
            None
        """
        self._limit_orders[position] = {"limit": limit, "persistent": persistent}

    def step(self, position_index=None):
        """Advance the environment by one step.

        Applies the selected action, updates the portfolio, computes reward,
        and returns the new observation along with termination flags and info.

        Args:
            position_index (int | None): Index within ``positions`` to select.
                If ``None``, keeps the current position.

        Returns:
            tuple: ``(obs, reward, done, truncated, info)`` where
                - ``obs`` (numpy.ndarray): Next observation.
                - ``reward`` (float): Current step reward.
                - ``done`` (bool): Whether the episode terminated due to failure.
                - ``truncated`` (bool): Whether the episode was truncated due to
                  max duration or dataset end.
                - ``info`` (dict): Latest recorded info from history.
        """
        if position_index is not None:
            self._take_action(self.positions[position_index])
        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=position_index,
            position=self._position,
            real_position=self._portfolio.real_position(price),
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=portfolio_value,
            portfolio_distribution=portfolio_distribution,
            reward=0,
        )
        if not done:
            reward = self.reward_function(self.historical_info)
            self.historical_info["reward", -1] = reward

        if done or truncated:
            self.calculate_metrics()
            self.log()
        return (
            self._get_obs(),
            self.historical_info["reward", -1],
            done,
            truncated,
            self.historical_info[-1],
        )

    def add_metric(self, name, function):
        """Register a custom metric to be computed at episode end.

        Args:
            name (str): Display name of the metric.
            function (callable): Function mapping ``History`` to a value.

        Returns:
            None
        """
        self.log_metrics.append({"name": name, "function": function})

    def calculate_metrics(self):
        """Compute default and custom episode metrics.

        Populates ``self.results_metrics`` with market and portfolio returns and
        any custom metrics added via ``add_metric``.

        Returns:
            None
        """
        market_return_val = 100 * (
            self.historical_info["data_close", -1] / self.historical_info["data_close", 0] - 1
        )
        portfolio_return_val = 100 * (
            self.historical_info["portfolio_valuation", -1]
            / self.historical_info["portfolio_valuation", 0]
            - 1
        )
        self.results_metrics = {
            "Market Return": f"{market_return_val:5.2f}%",
            "Portfolio Return": f"{portfolio_return_val:5.2f}%",
        }

        for metric in self.log_metrics:
            self.results_metrics[metric["name"]] = metric["function"](self.historical_info)

    def get_metrics(self):
        """Return the computed episode metrics.

        Returns:
            dict: Mapping metric names to their values (usually strings).
        """
        return self.results_metrics

    def log(self):
        """Print metrics to stdout when verbosity is enabled.

        Returns:
            None
        """
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

    def save_for_render(self, dir="render_logs"):
        """Persist environment and history data for external rendering.

        Saves a merged DataFrame containing market data and environment history
        to a timestamped pickle file. This can later be visualized with the
        renderer utility.

        Args:
            dir (str): Directory where render logs will be written. Created if
                it does not exist. Defaults to ``"render_logs"``.

        Returns:
            None

        Raises:
            AssertionError: If required OHLC columns are missing in ``self.df``.
        """
        assert (
            "open" in self.df and "high" in self.df and "low" in self.df and "close" in self.df
        ), "Your DataFrame needs to contain columns : open, high, low, close to render !"
        columns = list(
            set(self.historical_info.columns) - set([f"date_{col}" for col in self._info_columns])
        )
        history_df = pd.DataFrame(self.historical_info[columns], columns=columns)
        history_df.set_index("date", inplace=True)
        history_df.sort_index(inplace=True)
        render_df = self.df.join(history_df, how="inner")

        if not os.path.exists(dir):
            os.makedirs(dir)
        render_df.to_pickle(
            f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        )


class MultiDatasetTradingEnv(TradingEnv):
    """
    (Inherits from TradingEnv) A TradingEnv environment that handles multiple datasets.
    It automatically switches from one dataset to another at the end of an episode.
    Bringing diversity by having several datasets, even from the same pair from
    different exchanges, is a good idea. This should help avoiding overfitting.

    It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('MultiDatasetTradingEnv',
            dataset_dir = 'data/*.pkl',
            ...
        )



    :param dataset_dir: A `glob path <https://docs.python.org/3.6/library/glob.html>`_
        that needs to match your datasets. All of your datasets need to match the
        dataset requirements (see docs from TradingEnv). If it is not the case, you
        can use the ``preprocess`` param to make your datasets match the requirements.
    :type dataset_dir: str

    :param preprocess: This function takes a pandas.DataFrame and returns a
        pandas.DataFrame. This function is applied to each dataset before being used
        in the environment.

        For example, imagine you have a folder named 'data' with several datasets
        (formatted as .pkl)

        .. code-block:: python

            import pandas as pd
            import numpy as np
            import gymnasium as gym
            from gym_trading_env

            # Generating features.
            def preprocess(df : pd.DataFrame):
                # You can easily change your inputs this way
                df["feature_close"] = df["close"].pct_change()
                df["feature_open"] = df["open"]/df["close"]
                df["feature_high"] = df["high"]/df["close"]
                df["feature_low"] = df["low"]/df["close"]
                df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
                df.dropna(inplace= True)
                return df

            env = gym.make(
                    "MultiDatasetTradingEnv",
                    dataset_dir= 'examples/data/*.pkl',
                    preprocess= preprocess,
                )

    :type preprocess: function<pandas.DataFrame->pandas.DataFrame>

    :param episodes_between_dataset_switch: Number of times a dataset is used to create
        an episode, before moving on to another dataset. It can be useful for
        performances when `max_episode_duration` is low.
    :type episodes_between_dataset_switch: optional - int
    """

    def __init__(
        self,
        dataset_dir,
        *args,
        preprocess=lambda df: df,
        episodes_between_dataset_switch=1,
        **kwargs,
    ):
        """Initialize the multi-dataset trading environment.

        Args:
            dataset_dir (str): Glob path matching datasets to be used.
            *args: Positional arguments forwarded to ``TradingEnv``.
            preprocess (callable): Function ``(pd.DataFrame) -> pd.DataFrame``
                applied to each dataset before use. Defaults to identity.
            episodes_between_dataset_switch (int): Number of episodes to run on
                the same dataset before switching. Defaults to 1.
            **kwargs: Keyword arguments forwarded to ``TradingEnv``.

        Raises:
            FileNotFoundError: If no datasets are found at ``dataset_dir``.
        """
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)
        if len(self.dataset_pathes) == 0:
            raise FileNotFoundError(f"No dataset found with the path : {self.dataset_dir}")
        self.dataset_nb_uses = np.zeros(shape=(len(self.dataset_pathes),))
        super().__init__(self.next_dataset(), *args, **kwargs)

    def next_dataset(self):
        """Pick the next dataset to use based on least-usage strategy.

        Returns:
            pandas.DataFrame: The preprocessed DataFrame of the selected dataset.
        """
        self._episodes_on_this_dataset = 0
        # Find the indexes of the less explored dataset
        potential_dataset_pathes = np.where(self.dataset_nb_uses == self.dataset_nb_uses.min())[0]
        # Pick one of them
        random_int = np.random.randint(potential_dataset_pathes.size)
        dataset_idx = potential_dataset_pathes[random_int]
        dataset_path = self.dataset_pathes[dataset_idx]
        self.dataset_nb_uses[dataset_idx] += 1  # Update nb use counts

        self.name = Path(dataset_path).name
        return self.preprocess(pd.read_pickle(dataset_path))

    def reset(self, seed=None, options=None, **kwargs):
        """Reset state and optionally switch to a different dataset.

        After a configurable number of episodes, the underlying dataset is
        switched to diversify training and reduce overfitting.

        Args:
            seed (int | None): Random seed for reproducibility. Optional.
            options (dict | None): Additional options for Gymnasium. Optional.
            **kwargs: Forwarded to parent ``reset``.

        Returns:
            tuple[numpy.ndarray, dict]: The initial observation and info dict.
        """
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_df(self.next_dataset())
        if self.verbose > 1:
            print(f"Selected dataset {self.name} ...")
        return super().reset(seed=seed, options=options, **kwargs)
