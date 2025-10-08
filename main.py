# main.py
from __future__ import annotations

import os
import subprocess
import sys


def run_pipeline() -> None:
    """Orchestrate the full data pipeline: fetch, process, train."""
    print("Starting Football Analytics Pipeline…")

    scripts = [
        "fetch_statsbomb.py",
        "fetch_shots_data.py",
        "fetch_passing_data.py",
        "preprocess_xG.py",
        "train_xG_model.py",
    ]

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    for script in scripts:
        script_path = os.path.join(base_dir, script)
        print(f"▶ Running {script}…")
        result = subprocess.run([sys.executable, script_path], check=False)
        if result.returncode != 0:
            print(f"Error while running {script}. Check logs.")
            return

    print("Data pipeline completed successfully!")
    print("You can now run the dashboard with:")
    print(" streamlit run streamlit_app.py")


if __name__ == "__main__":
    run_pipeline()
