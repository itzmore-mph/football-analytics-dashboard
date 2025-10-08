# tests/conftest.py

import importlib


def test_streamlit_app_importable():
    # Import via importlib so flake8 sees importlib used
    mod = importlib.import_module("streamlit_app")
    assert mod is not None
