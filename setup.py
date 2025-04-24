from setuptools import setup, find_packages

setup(
    name="football_analytics_portfolio",
    version="0.1.0",
    description=(
        "A Streamlit-based football analytics dashboard and data pipeline"
    ),
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "streamlit",
        "matplotlib",
        "networkx",
        "mplsoccer",
        "xgboost",
        "scikit-learn",
        "requests",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "football-dashboard=dashboard.app:main",
            "football-pipeline=main:run_pipeline",
        ],
    },
)
