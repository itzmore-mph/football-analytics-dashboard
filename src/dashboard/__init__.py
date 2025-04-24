# src/dashboard/__init__.py

__version__ = "0.1.0"

# expose the high-level app entrypoint
from .app import main

# expose data-loading functions
from .data import load_shot_data, load_passing_data

# expose plotting functions
from .plots import plot_shot_map, plot_passing_network

__all__ = [
    "main",
    "load_shot_data",
    "load_passing_data",
    "plot_shot_map",
    "plot_passing_network",
]
