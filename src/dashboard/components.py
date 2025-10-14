# src/dashboard/components.py
import streamlit as st


def sidebar_filters():
    st.sidebar.header("Filters")

    min_minute, max_minute = st.sidebar.slider(
        "Minute range",
        min_value=0,
        max_value=120,
        value=(0, 120),
        key="minute_range",
    )

    pass_threshold = st.sidebar.slider(
        "Pass edge threshold",
        min_value=1,
        max_value=10,
        value=3,
        key="pass_threshold",
    )

    return {
        "minute_range": (min_minute, max_minute),
        "pass_threshold": pass_threshold,
    }
