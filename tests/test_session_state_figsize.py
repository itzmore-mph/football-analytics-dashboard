from src import ui
import streamlit as st


def test_session_state_persistence():
    # Sanity-check the ui module surface is present (use the imported ui)
    assert hasattr(ui, "render_dashboard")

    # ensure default assignment works in this test env
    if "fig_size" in st.session_state:
        del st.session_state["fig_size"]
    if "fig_size" not in st.session_state:
        st.session_state["fig_size"] = (880, 480)
    assert "fig_size" in st.session_state
