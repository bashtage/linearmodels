"""Helpers for the tests of all packages"""

from collections.abc import Mapping
from contextlib import nullcontext
import re

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.shared.exceptions import DefaultBandwidthWarning, MissingValueWarning

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def assert_same_text(text, expected, label="", rtol=1e-8, atol=1e-12):
    """
    Assert that two texts have the same words and numbers that are close

    A summary or a representation of a result prints numbers that are computed, and
    the last digits of a number that is zero in exact arithmetic depend on the
    platform: -2.22e-16, 0.0000 and 1.11e-16 are printed for the same zero. The words
    of the two texts, with the numbers replaced by a placeholder, are compared as
    text, and the numbers are compared as numbers.

    Parameters
    ----------
    text : str
        The text to check.
    expected : str
        The text with the words and the numbers that are expected.
    label : str, optional
        A message that is shown if the texts differ.
    rtol : float, optional
        Relative tolerance of the numbers.
    atol : float, optional
        Absolute tolerance of the numbers.
    """
    words = [" ".join(NUMBER.sub("#", t).split()) for t in (text, expected)]
    assert words[0] == words[1], label
    numbers = [np.array(NUMBER.findall(t), dtype=float) for t in (text, expected)]
    assert_allclose(numbers[0], numbers[1], rtol=rtol, atol=atol, err_msg=label)


def assert_close_matrix(actual, desired, rtol=1e-10):
    """
    Assert that two matrices are close, relative to their largest element

    The relative rounding error of an element that is close to 0 is larger than that
    of the largest elements, so that the tolerance of every element is relative to
    the largest.

    Parameters
    ----------
    actual : array_like
        The matrix to check.
    desired : array_like
        The matrix that is expected.
    rtol : float, optional
        The tolerance, relative to each element and to the largest element.
    """
    desired = np.asarray(desired)
    assert_allclose(actual, desired, rtol=rtol, atol=rtol * np.abs(desired).max())


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
