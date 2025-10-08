"""Streamlit app entrypoint.

This file is intended to be run by `streamlit run streamlit_app.py`
It locates the repository root (parent of this file) and passes it to the
dashboard renderer so data paths are resolved reliably.
"""
from pathlib import Path
import streamlit as st

try:
    # Late import so tests / static checks don't import Streamlit immediately
    from src.ui import render_dashboard
except Exception:
    # Let the import error surface for debugging in development environments
    raise

if __name__ == "__main__":
    # Use the directory containing this file as root. Avoids hard-coded CWD.
    repo_root = Path(__file__).resolve().parent
    # Friendly note for users
    st.set_page_config(
        page_title="Football Analytics Dashboard",
        layout="wide"
        )
    render_dashboard(repo_root)
