def test_validation_module_importable():
    import importlib
    mod = importlib.import_module("src.validation")
    assert mod is not None
