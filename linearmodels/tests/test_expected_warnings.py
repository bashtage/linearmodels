"""
Tests of the helpers that the tests use to expect the warnings of the models

The estimators are tested with inputs that do and do not have missing values, and the
models issue a MissingValueWarning only if they drop observations. The kernel
estimators of the GMM models issue a DefaultBandwidthWarning only if they are not given
a bandwidth.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linearmodels.shared.exceptions import DefaultBandwidthWarning, MissingValueWarning
from linearmodels.tests._utility import (
    has_missing,
    warns_if_default_bandwidth,
    warns_if_missing,
)

MISSING_MESSAGE = "Inputs contain missing values. Dropping rows."
FRAME = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
FRAME_MISSING = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
ARRAY = np.arange(8.0).reshape((2, 2, 2))
ARRAY_MISSING = ARRAY.copy()
ARRAY_MISSING[1, 0, 1] = np.nan

INPUTS = {
    "ndarray": (ARRAY, ARRAY_MISSING),
    "int ndarray": (np.arange(4), None),
    "DataFrame": (FRAME, FRAME_MISSING),
    "Series": (FRAME["a"], FRAME_MISSING["a"]),
    "DataArray": (xr.DataArray(ARRAY), xr.DataArray(ARRAY_MISSING)),
    "object ndarray": (np.array(["a", 1], dtype=object), np.array(["a", None, 1.0])),
}


@pytest.mark.parametrize("kind", list(INPUTS))
def test_has_missing(kind):
    clean, missing = INPUTS[kind]
    assert not has_missing(clean)
    if missing is not None:
        assert has_missing(missing)
        assert has_missing(clean, missing)
        assert has_missing(missing, clean)


def test_none_is_not_missing():
    assert not has_missing(None)
    assert not has_missing(FRAME, None)
    assert not has_missing()


def test_warns_if_missing_expects_the_warning_of_inputs_with_missing_values():
    with warns_if_missing(FRAME, FRAME_MISSING):
        warnings.warn(MISSING_MESSAGE, MissingValueWarning, stacklevel=1)


def test_warns_if_missing_fails_if_the_model_does_not_warn():
    with pytest.raises(pytest.fail.Exception, match="DID NOT WARN"):
        with warns_if_missing(FRAME, FRAME_MISSING):
            pass


def test_warns_if_missing_does_not_expect_a_warning_without_missing_values():
    with warns_if_missing(FRAME, ARRAY):
        pass
    # and a warning in the context is not caught
    with pytest.warns(MissingValueWarning, match="missing values"):
        with warns_if_missing(FRAME, ARRAY):
            warnings.warn(MISSING_MESSAGE, MissingValueWarning, stacklevel=1)


def test_warns_if_missing_matches_the_message():
    message = "1 of the cross-sectional regressions cannot be estimated."
    # The warning does not match, and is issued again when the context closes
    with (
        pytest.warns(MissingValueWarning, match="cross-sectional"),
        pytest.raises(pytest.fail.Exception, match="did not match"),
        warns_if_missing(FRAME_MISSING),
    ):
        warnings.warn(message, MissingValueWarning, stacklevel=1)
    # unless any message is accepted
    with warns_if_missing(FRAME_MISSING, match=None):
        warnings.warn(message, MissingValueWarning, stacklevel=1)


@pytest.mark.parametrize(
    ("options", "warns"),
    [
        ({}, True),
        ({"bandwidth": None}, True),
        ({"optimal_bw": None, "bandwidth": None}, True),
        ({"bandwidth": 5}, False),
        ({"bandwidth": 0}, False),
        ({"optimal_bw": True}, False),
        ({"optimal_bw": False}, False),
        ({"bandwidth": 5, "optimal_bw": True}, False),
    ],
)
def test_warns_if_default_bandwidth(options, warns):
    message = "The default bandwidth changed in linearmodels 7.1."
    if warns:
        with warns_if_default_bandwidth(**options):
            warnings.warn(message, DefaultBandwidthWarning, stacklevel=1)
        with pytest.raises(pytest.fail.Exception, match="DID NOT WARN"):
            with warns_if_default_bandwidth(**options):
                pass
    else:
        with warns_if_default_bandwidth(**options):
            pass
