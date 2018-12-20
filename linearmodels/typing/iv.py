from typing import Union

import numpy as np
import pandas as pd

base_data_types = [np.ndarray, pd.DataFrame, pd.Series]
try:
    import xarray as xr

    ArrayLike = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

except ImportError:
    ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]

OptionalArrayLike = Union[ArrayLike, None]
