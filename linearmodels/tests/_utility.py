"""Helpers for the tests of all packages"""

from collections.abc import Mapping
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from linearmodels.shared.exceptions import DefaultBandwidthWarning, MissingValueWarning


def has_missing(*inputs):
    """
    Whether any of the inputs of a model has a missing value

    Parameters
    ----------
    *inputs
        NumPy arrays, pandas Series and DataFrames, xarray DataArrays, None, or
        dicts of these, for example the equations of a system model.

    Returns
    -------
    bool
        True if any of the inputs has a NaN or a missing value of another type.
    """
    for value in inputs:
        if value is None:
            continue
        if isinstance(value, Mapping):
            missing = has_missing(*value.values())
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            missing = value.isna().to_numpy().any()
        else:
            # NumPy arrays, and the values of an xarray DataArray
            missing = pd.isna(np.asarray(getattr(value, "values", value))).any()
        if missing:
            return True
    return False


def warns_if_missing(*inputs, match="missing values"):
    """
    Context that expects the warning for the missing values that a model drops

    A model drops the observations that have a missing value in any of its inputs
    when it is created and issues a MissingValueWarning. The tests of the
    estimators use inputs with and without missing values, and this expects the
    warning exactly if the inputs have a missing value.

    Parameters
    ----------
    *inputs
        The inputs of the model that can have missing values, for example the
        dependent variable and the regressors.
    match : str, optional
        A regular expression that the message of the warning has to match, or None
        to accept any MissingValueWarning.

    Returns
    -------
    context manager
        pytest.warns for a MissingValueWarning if any of the inputs has a missing
        value and a context that does nothing if none has.
    """
    if has_missing(*inputs):
        return pytest.warns(MissingValueWarning, match=match)
    return nullcontext()


def warns_if_default_bandwidth(**options):
    """
    Context that expects the warning for the default bandwidth of a kernel estimator

    The kernel estimators of the covariance of the moment conditions of the GMM
    models estimate the bandwidth from the data and issue a DefaultBandwidthWarning
    if neither bandwidth nor optimal_bw is given.

    Parameters
    ----------
    **options
        The options of the kernel estimator. Only bandwidth and optimal_bw matter,
        and None is the same as not given.

    Returns
    -------
    context manager
        pytest.warns for a DefaultBandwidthWarning if neither bandwidth nor
        optimal_bw is given and a context that does nothing if one of them is.
    """
    if options.get("bandwidth") is None and options.get("optimal_bw") is None:
        return pytest.warns(DefaultBandwidthWarning, match=r"7\.1")
    return nullcontext()
