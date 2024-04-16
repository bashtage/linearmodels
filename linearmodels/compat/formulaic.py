def monkey_patch_materializers():
    from formulaic.materializers.base import FormulaMaterializer
    from formulaic.materializers.pandas import PandasMaterializer

    if "pandas.DataFrame" not in FormulaMaterializer.REGISTERED_INPUTS:
        FormulaMaterializer.REGISTERED_INPUTS["pandas.DataFrame"] = (
            FormulaMaterializer.REGISTERED_INPUTS["pandas.core.frame.DataFrame"]
        )
    if "pandas.DataFrame" not in PandasMaterializer.REGISTERED_INPUTS:
        PandasMaterializer.REGISTERED_INPUTS["pandas.DataFrame"] = (
            PandasMaterializer.REGISTERED_INPUTS["pandas.core.frame.DataFrame"]
        )
