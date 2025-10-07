# src/dashboard/app.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
from src.ui import render_dashboard


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Football Analytics Dashboard", layout="centered")
    # repo root: .../src/dashboard/app.py -> parents[2]
    root = Path(__file__).resolve().parents[2]
    render_dashboard(root)


if __name__ == "__main__":
    main()
