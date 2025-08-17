def test_import_infer_modules():
    import src.infer.posterior  # noqa: F401
    import src.infer.early_exit  # noqa: F401
    import src.infer.calibration  # noqa: F401
    import src.infer.ood  # noqa: F401
