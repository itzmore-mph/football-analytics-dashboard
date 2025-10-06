# streamlit_app.py
from pathlib import Path
import sys

# ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.app import main

if __name__ == "__main__":
    main()
