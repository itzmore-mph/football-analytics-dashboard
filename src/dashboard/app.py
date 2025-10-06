from pathlib import Path
import streamlit as st
from .ui import render_dashboard


def main():
    st.set_page_config(
        page_title="Football Analytics Dashboard",
        layout="wide"
    )
    root = Path(__file__).resolve().parents[1]
    render_dashboard(root)


if __name__ == "__main__":
    main()
