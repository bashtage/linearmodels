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

KINDS = ["ndarray", "int ndarray", "DataFrame", "Series", "DataArray", "object ndarray"]


def inputs_of_kind(kind):
    """An input without a missing value and one with a missing value, or None"""
    if kind == "DataArray":
        xr = pytest.importorskip("xarray")
        return xr.DataArray(ARRAY), xr.DataArray(ARRAY_MISSING)
    return {
        "ndarray": (ARRAY, ARRAY_MISSING),
        "int ndarray": (np.arange(4), None),
        "DataFrame": (FRAME, FRAME_MISSING),
        "Series": (FRAME["a"], FRAME_MISSING["a"]),
        "object ndarray": (
            np.array(["a", 1], dtype=object),
            np.array(["a", None, 1.0]),
        ),
    }[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_has_missing(kind):
    clean, missing = inputs_of_kind(kind)
    clean_has_missing = has_missing(clean)
    assert not clean_has_missing
    if missing is not None:
        with_missing = has_missing(missing)
        both = has_missing(clean, missing)
        reversed_both = has_missing(missing, clean)
        assert with_missing
        assert both
        assert reversed_both


def test_has_missing_looks_into_dicts():
    equations = {
        "a": {"dependent": FRAME["a"], "exog": ARRAY},
        "b": {"dependent": FRAME["b"], "exog": None},
    }
    equations_missing = {
        "a": {"dependent": FRAME["a"], "exog": ARRAY},
        "b": {"dependent": FRAME_MISSING["a"], "exog": None},
    }
    clean = has_missing(equations)
    with_missing = has_missing(equations_missing)
    assert not clean
    assert with_missing


def test_none_is_not_missing():
    results = [has_missing(None), has_missing(FRAME, None), has_missing()]
    assert not any(results)


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
