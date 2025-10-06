from setuptools import setup, find_packages

setup(
    name="football_analytics_dashboard",
    version="0.1.0",
    description="A Streamlit-based football analytics dashboard and data pipeline",
    author="Moritz Philipp Haaf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            # Match the actual app entry point you use
            "football-dashboard=src.app:main",
            "football-pipeline=main:run_pipeline",
        ],
    },
)
