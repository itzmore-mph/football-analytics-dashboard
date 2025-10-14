from __future__ import annotations
import streamlit as st


def metric_badge(
    label: str,
    value: float | int | str,
    help_text: str | None = None
):
    col = st.container(border=True)
    with col:
        st.markdown(f"### {label}")
        st.markdown(
            f"<div style='font-size: 28px; font-weight:700'>{value}</div>",
            unsafe_allow_html=True
        )
        if help_text:
            st.caption(help_text)


def sidebar_filters():
    st.sidebar.header("Filters")
    min_minute, max_minute = st.sidebar.slider(
        "Minute range", 0, 120, (0, 120)
    )
    pass_threshold = st.sidebar.slider("Pass edge threshold", 1, 10, 3)
    return {
        "minute_range": (min_minute, max_minute),
        "pass_threshold": pass_threshold,
    }
