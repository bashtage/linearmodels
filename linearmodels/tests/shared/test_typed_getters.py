from typing import Any, Tuple, Type

import numpy as np
import pandas as pd
import pytest

from linearmodels.panel.data import PanelData
from linearmodels.shared.typed_getters import (
    get_array_like,
    get_bool,
    get_float,
    get_panel_data_like,
    get_string,
)

ARRAY_LIKE: Tuple[Type, ...] = (np.ndarray, pd.Series, pd.DataFrame)
PANEL_LIKE: Tuple[Type, ...] = ARRAY_LIKE + (PanelData,)
ARRAYS: Tuple[Any, ...] = (np.array([1.0]), pd.Series([1.0]), pd.DataFrame([[1.0]]))
PANELS: Tuple[Any, ...] = ARRAYS + (PanelData(np.array([[[1.0]]])),)
try:
    import xarray as xr

    ARRAY_LIKE += (xr.DataArray,)
    PANEL_LIKE += (xr.DataArray,)
    ARRAYS += (xr.DataArray(ARRAYS[0]),)
    PANELS += (xr.DataArray(ARRAYS[0]),)
except ImportError:
    pass


@pytest.fixture(params=ARRAYS)
def arr(request):
    return request.param


@pytest.fixture(params=PANELS)
def panel(request):
    return request.param


def test_bool():
    assert get_bool({"v": True}, "v")
    assert get_bool({"v": True}, "a") is False
    with pytest.raises(TypeError, match=r".not a bool"):
        get_bool({"v": "True"}, "v")
    with pytest.raises(TypeError, match=r".not a bool"):
        get_bool({"v": 1}, "v")
    with pytest.raises(TypeError, match=r".not a bool"):
        get_bool({"v": 1.0}, "v")


def test_float():
    assert get_float({"v": True}, "v") == 1.0
    assert get_float({"v": True}, "a") is None
    with pytest.raises(TypeError, match=r".not a float"):
        get_float({"v": "1.0"}, "v")


def test_string():
    assert get_string({"v": "1"}, "v") == "1"
    assert get_string({"v": True}, "a") is None
    with pytest.raises(TypeError, match=r".not a str"):
        get_string({"v": 1.0}, "v")
    with pytest.raises(TypeError, match=r".not a str"):
        get_string({"v": b"1"}, "v")


def test_array_like(arr):
    assert isinstance(get_array_like({"v": arr}, "v"), ARRAY_LIKE)
    assert get_array_like({"v": arr}, "a") is None
    with pytest.raises(TypeError, match=r".not array-like"):
        get_array_like({"v": 1}, "v")
    with pytest.raises(TypeError, match=r".not array-like"):
        get_array_like({"v": [1]}, "v")


def test_panel_data_like(panel):
    assert isinstance(get_panel_data_like({"v": panel}, "v"), PANEL_LIKE)
    assert get_panel_data_like({"v": panel}, "a") is None
    with pytest.raises(TypeError):
        get_panel_data_like({"v": 1}, "v")
    with pytest.raises(TypeError):
        get_panel_data_like({"v": [1]}, "v")
