from __future__ import annotations

import os
import runpy
from pathlib import Path


def _project_root() -> Path:
    # This file lives at: <root>/src/gym_trading_env/cli_examples.py
    here = Path(__file__).resolve()
    # parents[0] = gym_trading_env, [1] = src, [2] = project root
    root = here.parents[2]
    return root


def _run_example(relative_path: str) -> None:
    root = _project_root()
    examples_dir = root / "examples"
    script_path = examples_dir / relative_path

    if not script_path.exists():
        raise FileNotFoundError(f"Example script not found: {script_path}")

    # Ensure the example's relative file references (like examples/data) work
    prev_cwd = Path.cwd()
    try:
        os.chdir(root)
        # Execute the script as if it were __main__
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(prev_cwd)


def example_download() -> None:
    _run_example("example_download.py")


def example_env() -> None:
    _run_example("example_environnement.py")


def example_multi_env() -> None:
    _run_example("example_multi_environnement.py")


def example_render() -> None:
    _run_example("example_render.py")


def example_vec_env() -> None:
    _run_example("example_vectorized_environment.py")


def example_vec_multi_env() -> None:
    _run_example("example_vectorized_multi_environment.py")
