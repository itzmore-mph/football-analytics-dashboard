# streamlit_app.py
from __future__ import annotations

from src.dashboard.app import run

import os
import sys
sys.path.insert(0, os.path.abspath("."))  # ensure 'src' is importable on Cloud

if __name__ == "__main__":
    run()
