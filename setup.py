from __future__ import annotations
from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="football_analytics_dashboard",
        version="0.1.0",
        description="Streamlit dashboard for football analytics",
        author="Moritz Philipp Haaf",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        py_modules=["main"],
        install_requires=[],
        python_requires=">=3.11",
        entry_points={
            "console_scripts": [
                "football-dashboard=dashboard.app:main",
                "football-pipeline=main:run_pipeline",
            ],
        },
    )
