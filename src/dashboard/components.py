from __future__ import annotations
from typing import Any
import streamlit as st


def sidebar_filters() -> dict:
    st.sidebar.header("Filters")
    min_minute, max_minute = st.sidebar.slider(
        "Minute range", 0, 120, (0, 120), key="minute_range"
    )
    pass_threshold = st.sidebar.slider(
        "Pass edge threshold", 1, 10, 3, key="pass_threshold"
    )
    return {
        "minute_range": (min_minute, max_minute),
        "pass_threshold": pass_threshold,
    }


def metric_badge(
    label: str,
    value: Any,
    delta: Any | None = None,
    help: str | None = None,
) -> None:
    st.metric(label=label, value=value, delta=delta, help=help)
