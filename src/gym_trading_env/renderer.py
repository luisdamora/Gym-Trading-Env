"""Flask-based renderer to visualize environment runs and metrics.

This module exposes a small Flask app that serves a front-end to visualize
price series, portfolio valuation, and custom lines/metrics computed from
render logs saved by the trading environment.
"""

import glob
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template

from .utils.charts import charts


class Renderer:
    """Render server that serves charts and metrics from saved logs.

    Args:
        render_logs_dir (str): Directory containing ``.pkl`` render logs as
            produced by ``TradingEnv.save_for_render``.
    """

    def __init__(self, render_logs_dir):
        self.app = Flask(__name__, static_folder="./templates/")
        # self.app.debug = True
        self.app.config["EXPLAIN_TEMPLATE_LOADING"] = True
        self.df = None
        self.render_logs_dir = render_logs_dir
        self.metrics = [
            {"name": "Market Return", "function": self._metric_market_return},
            {"name": "Portfolio Return", "function": self._metric_portfolio_return},
        ]
        self.lines = []

    def add_metric(self, name, function):
        """Register a metric to be computed and displayed in the UI.

        Args:
            name (str): Human-readable name of the metric.
            function (callable): Function ``(pd.DataFrame) -> str`` returning a
                formatted textual value.
        """
        self.metrics.append({"name": name, "function": function})

    def add_line(self, name, function, line_options=None):
        """Register a custom line to be overlaid in the main chart.

        Args:
            name (str): Line label.
            function (callable): Function ``(pd.DataFrame) -> list[dict]`` or a
                structure consumed by ``charts`` to add a line/series.
            line_options (dict | None): Optional plotting config merged into the
                produced line.
        """
        self.lines.append({"name": name, "function": function})
        if line_options is not None:
            self.lines[-1]["line_options"] = line_options

    @staticmethod
    def _metric_market_return(df: pd.DataFrame) -> str:
        """Compute market return percentage from the provided DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame including a ``close`` column.

        Returns:
            str: Market return formatted as a percentage string.
        """
        value = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        return f"{value:0.2f}%"

    @staticmethod
    def _metric_portfolio_return(df: pd.DataFrame) -> str:
        """Compute portfolio return percentage from the provided DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame including ``portfolio_valuation``.

        Returns:
            str: Portfolio return formatted as a percentage string.
        """
        value = (df["portfolio_valuation"].iloc[-1] / df["portfolio_valuation"].iloc[0] - 1) * 100
        return f"{value:0.2f}%"

    def compute_metrics(self, df):
        """Evaluate all registered metrics and cache their values.

        Args:
            df (pandas.DataFrame): The dataset to compute metrics on.
        """
        for metric in self.metrics:
            metric["value"] = metric["function"](df)

    def run(
        self,
    ):
        """Start the Flask development server and expose routes.

        Routes:
            - ``/``: Index page listing available render logs.
            - ``/update_data/<name>``: Returns chart options for a selected log.
            - ``/metrics``: Returns computed metrics as JSON.
        """

        @self.app.route("/")
        def index():
            render_pathes = glob.glob(f"{self.render_logs_dir}/*.pkl")
            render_names = [Path(path).name for path in render_pathes]
            return render_template("index.html", render_names=render_names)

        @self.app.route("/update_data/<name>")
        def update(name=None):
            if name is None or name == "":
                render_pathes = glob.glob(f"{self.render_logs_dir}/*.pkl")
                name = Path(render_pathes[-1]).name
            self.df = pd.read_pickle(f"{self.render_logs_dir}/{name}")
            chart = charts(self.df, self.lines)
            return chart.dump_options_with_quotes()

        @self.app.route("/metrics")
        def get_metrics():
            self.compute_metrics(self.df)
            return jsonify(
                [{"name": metric["name"], "value": metric["value"]} for metric in self.metrics]
            )

        self.app.run()
