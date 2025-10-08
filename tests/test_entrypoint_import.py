def test_streamlit_app_importable():
    # Ensure streamlit_app imports without immediately starting long work.
    import streamlit_app
    assert streamlit_app is not None
