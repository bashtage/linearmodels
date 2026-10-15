"""
Getters that check the type of a value in a mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from linearmodels.panel.data import PanelData, PanelDataLike
import linearmodels.typing.data


def get_string(d: Mapping[str, Any], key: str) -> str | None:
    """
    Get a string or None from a mapping.

    Parameters
    ----------
    d : dict of str to object
        A mapping.
    key : str
        The key to look up.

    Returns
    -------
    str or None
        The string, or None if the key is not in the mapping or its value is
        None.

    Raises
    ------
    TypeError
        If the value of the key is not a str or None.
    """
    out: str | None = None
    if key in d:
        out = d[key]
        if out is not None:
            if isinstance(out, str):
                return out
            else:
                raise TypeError(f"{key} found in the dictionary but it is not a str.")
    return out


def get_float(d: Mapping[str, Any], key: str) -> float | None:
    """
    Get a float or None from a mapping.

    Parameters
    ----------
    d : dict of str to object
        A mapping.
    key : str
        The key to look up.

    Returns
    -------
    float or None
        The value as a float, or None if the key is not in the mapping or its
        value is None.

    Raises
    ------
    TypeError
        If the value of the key is not an int, a float or None.
    """
    out: float | None = None
    if key in d:
        out = d[key]
        if out is not None:
            if isinstance(out, (int, float, np.integer, np.floating)):
                return float(out)
            else:
                raise TypeError(f"{key} found in the dictionary but it is not a float.")
    return out


def get_bool(d: Mapping[str, Any], key: str) -> bool:
    """
    Get a bool from a mapping, defaulting to False.

    Parameters
    ----------
    d : dict of str to object
        A mapping.
    key : str
        The key to look up.

    Returns
    -------
    bool
        The value if the key is in the mapping. False if the key is not in the
        mapping or its value is None.

    Raises
    ------
    TypeError
        If the value of the key is not a bool or None.
    """
    out: bool | None = False
    if key in d:
        out = d[key]
        if not (out is None or isinstance(out, (bool, np.bool_))):
            raise TypeError(f"{key} found in the dictionary but it is not a bool.")
    return bool(out)


def get_array_like(
    d: Mapping[str, Any], key: str
) -> linearmodels.typing.data.DataContainer | None:
    """
    Get an array_like or None from a mapping.

    Parameters
    ----------
    d : dict of str to object
        A mapping.
    key : str
        The key to look up.

    Returns
    -------
    ndarray, DataFrame, Series, DataArray or None
        The value, or None if the key is not in the mapping or its value is
        None. A DataArray is possible only if xarray is installed.

    Raises
    ------
    TypeError
        If the value of the key is not an ndarray, a DataFrame, a Series, a
        DataArray or None.
    """

    out: linearmodels.typing.data.DataContainer | None = None
    if key in d:
        out = d[key]
        if out is not None:
            array_like: Any = (np.ndarray, DataFrame, Series)
            try:
                import xarray as xr  # noqa: PLC0415

                array_like += (xr.DataArray,)
            except ImportError:
                pass

            if isinstance(out, array_like):
                return out
            else:
                raise TypeError(
                    f"{key} found in the dictionary but it is not array-like."
                )
    return out


def get_panel_data_like(d: Mapping[str, Any], key: str) -> PanelDataLike | None:
    """
    Get a value that can be converted to panel data, or None, from a mapping.

    Parameters
    ----------
    d : dict of str to object
        A mapping.
    key : str
        The key to look up.

    Returns
    -------
    ndarray, DataFrame, Series, DataArray, PanelData or None
        The value, or None if the key is not in the mapping or its value is
        None. A DataArray is possible only if xarray is installed.

    Raises
    ------
    TypeError
        If the value of the key is not an ndarray, a DataFrame, a Series, a
        DataArray, a PanelData or None.
    """

    out: PanelDataLike | None = None
    if key in d:
        out = d[key]
        if out is not None:
            panel_data_like: Any = (np.ndarray, DataFrame, Series, PanelData)
            try:
                import xarray as xr  # noqa: PLC0415

                panel_data_like += (xr.DataArray,)
            except ImportError:
                pass

            if isinstance(out, panel_data_like):
                return out
            else:
                raise TypeError(
                    f"{key} found in the dictionary but it is not array-like."
                )
    return out
