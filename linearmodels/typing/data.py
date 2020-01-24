from typing import Optional, Union

import numpy as np
import pandas as pd

base_data_types = [np.ndarray, pd.DataFrame, pd.Series]
try:
    import xarray as xr

    ArrayLike = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

except ImportError:
    ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]  # type: ignore


OptionalArrayLike = Optional[ArrayLike]
OptionalDataFrame = Optional[pd.DataFrame]
