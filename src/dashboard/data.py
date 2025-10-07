# src/dashboard/data.py
import os
from typing import Optional
import pandas as pd


def load_shot_data(path: str) -> Optional[pd.DataFrame]:
    """Loads processed shot data from CSV.

    Args:
        path: Path to the processed_shots.csv file.
    Returns:
        A DataFrame if the file exists and is non-empty; otherwise None.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df if not df.empty else None
    return None


def load_passing_data(path: str) -> Optional[pd.DataFrame]:
    """Loads processed passing data from CSV.

    Args:
        path: Path to the passing_data.csv file.
    Returns:
        A DataFrame if the file exists and is non-empty; otherwise None.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df if not df.empty else None
    return None
