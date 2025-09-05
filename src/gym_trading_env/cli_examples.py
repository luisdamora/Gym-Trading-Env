from __future__ import annotations

import os
import runpy
from pathlib import Path


def _project_root() -> Path:
    """Determine the project root directory path.

    This function calculates the absolute path to the project root by navigating
    up the directory tree from the current file's location. The file is assumed
    to be located at <root>/src/gym_trading_env/cli_examples.py, so it navigates
    up two parent directories to reach the project root.

    Returns:
        Path: The absolute path to the project root directory.
    """


def _run_example(relative_path: str) -> None:
    """Execute an example script from the examples directory.

    This function runs a Python script located in the project's examples directory.
    It temporarily changes the working directory to the project root to ensure
    that relative file references in the example scripts work correctly.

    Args:
        relative_path (str): The relative path to the example script from the
            examples directory (e.g., "example_download.py").

    Raises:
        FileNotFoundError: If the specified example script does not exist.
    """
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
    """Run the data download example script.

    Executes the example_download.py script which demonstrates how to download
    trading data for use with the Gym Trading Environment.
    """


def example_env() -> None:
    """Run the basic environment example script.

    Executes the example_environnement.py script which demonstrates how to create
    and interact with a basic Gym Trading Environment instance.
    """


def example_multi_env() -> None:
    """Run the multi-environment example script.

    Executes the example_multi_environnement.py script which demonstrates how to
    work with multiple Gym Trading Environment instances simultaneously.
    """


def example_render() -> None:
    """Run the rendering example script.

    Executes the example_render.py script which demonstrates how to render
    and visualize the Gym Trading Environment state.
    """


def example_vec_env() -> None:
    """Run the vectorized environment example script.

    Executes the example_vectorized_environment.py script which demonstrates how to
    use vectorized environments for efficient batch processing in reinforcement learning.
    """


def example_vec_multi_env() -> None:
    """Run the vectorized multi-environment example script.

    Executes the example_vectorized_multi_environment.py script which demonstrates
    how to combine vectorization with multiple environments for highly efficient
    parallel reinforcement learning training.
    """
