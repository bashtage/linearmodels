from typing import Any, Mapping, Optional, Union

import numpy as np
from pandas import DataFrame, Series

from linearmodels.panel.data import PanelData, PanelDataLike
from linearmodels.typing import ArrayLike


def get_string(d: Mapping[str, Any], key: str) -> Optional[str]:
    """
    Helper function that gets a string or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {str, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """
    out: Optional[str] = None
    if key in d:
        out = d[key]
        if out is not None:
            if isinstance(out, str):
                return out
            else:

                raise TypeError(f"{key} found in the dictionary but it is not a str.")
    return out


def get_float(d: Mapping[str, Any], key: str) -> Optional[float]:
    """
    Helper function that gets a float or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {float, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """
    out: Optional[float] = None
    if key in d:
        out = d[key]
        if out is not None:
            if isinstance(out, (int, float, np.floating)):
                return float(out)
            else:
                raise TypeError(f"{key} found in the dictionary but it is not a float.")
    return out


def get_bool(d: Mapping[str, Any], key: str) -> bool:
    """
    Helper function that gets a bool, defaulting to False.

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    bool
        The boolean if the key is in the dictionary. If not found, returns
        False.
    """
    out: Optional[bool] = False
    if key in d:
        out = d[key]
        if not (out is None or isinstance(out, bool)):
            raise TypeError(f"{key} found in the dictionary but it is not a bool.")
    return bool(out)


def get_array_like(d: Mapping[str, Any], key: str) -> Optional[ArrayLike]:
    """
    Helper function that gets an array_like or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {array_like, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """

    out: Optional[ArrayLike] = None
    if key in d:
        out = d[key]
        if out is not None:
            array_like: Union[Any] = (np.ndarray, DataFrame, Series)
            try:
                import xarray as xr

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


def get_panel_data_like(d: Mapping[str, Any], key: str) -> Optional[PanelDataLike]:
    """
    Helper function that gets an panel_data_like or None

    Parameters
    ----------
    d : Mapping[str, Any]
        A mapping.
    key : str
        The key to lookup.

    Returns
    -------
    {panel_data_like, None}
        The string or None if the key is not in the dictionary. If in the
        dictionary, a type check is performed and TypeError is raised if
        not found.
    """

    out: Optional[PanelDataLike] = None
    if key in d:
        out = d[key]
        if out is not None:
            panel_data_like: Union[Any] = (np.ndarray, DataFrame, Series, PanelData)
            try:
                import xarray as xr

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
