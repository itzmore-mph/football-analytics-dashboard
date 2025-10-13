# setup.py
from setuptools import setup

setup(
    name="football_analytics_dashboard",
    version="0.1.0",
    py_modules=["main"],          # exposes the pipeline
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "football-pipeline=main:run_pipeline",
        ],
    },
)
