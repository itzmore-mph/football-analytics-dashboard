from __future__ import annotations

import os
import sys

from src.dashboard.app import run

# Ensure 'src' is importable on cloud runners
sys.path.insert(0, os.path.abspath("."))


if __name__ == "__main__":
    run()
