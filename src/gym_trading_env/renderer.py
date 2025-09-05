import glob
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template

from .utils.charts import charts


class Renderer:
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
        self.metrics.append({"name": name, "function": function})

    def add_line(self, name, function, line_options=None):
        self.lines.append({"name": name, "function": function})
        if line_options is not None:
            self.lines[-1]["line_options"] = line_options

    @staticmethod
    def _metric_market_return(df: pd.DataFrame) -> str:
        value = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        return f"{value:0.2f}%"

    @staticmethod
    def _metric_portfolio_return(df: pd.DataFrame) -> str:
        value = (
            df["portfolio_valuation"].iloc[-1] / df["portfolio_valuation"].iloc[0] - 1
        ) * 100
        return f"{value:0.2f}%"

    def compute_metrics(self, df):
        for metric in self.metrics:
            metric["value"] = metric["function"](df)

    def run(
        self,
    ):
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
