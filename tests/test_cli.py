from __future__ import annotations
import importlib


def test_cli_imports():
    importlib.import_module("src.cli")
