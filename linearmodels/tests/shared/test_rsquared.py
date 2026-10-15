"""
Tests of the coefficient of determination when the total sum of squares is zero

The coefficient of determination is 1 - RSS / TSS. A dependent variable that does not
vary has a TSS of 0, or a TSS that is only rounding error, e.g., the mean of a variable
that is always 0.1 is not exactly 0.1. The ratio is not defined then, and it is NaN, with
no error and no warning, as it is for an R-squared that is not defined elsewhere. The
bare ZeroDivisionError of the panel models and the rounding noise, e.g., 0.0042 for a
constant dependent variable, are gone.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from linearmodels.shared.rsquared import r_squared, total_ss_is_zero

EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize("seed", range(10))
def test_ordinary_values_are_one_minus_the_ratio(seed):
    rg = np.random.default_rng(seed)
    residual_ss = float(rg.uniform(0.1, 10))
    total_ss = float(residual_ss + rg.uniform(0.1, 10))
    y_ss = float(total_ss + rg.uniform(0, 100))
    # The same operations, so that the values are identical
    result = r_squared(residual_ss, total_ss, y_ss, 100)
    assert_allclose(result, 1 - residual_ss / total_ss, rtol=1e-12)
    # A model can fit worse than a constant, and the R-squared is negative
    result = r_squared(total_ss + 1.0, total_ss, y_ss, 100)
    assert result < 0.0
    result = r_squared(0.0, total_ss, y_ss, 100)
    assert_allclose(result, 1.0, rtol=1e-12)


@pytest.mark.parametrize("y_ss", [0.0, 1.0, 1e-300, 1e300])
@pytest.mark.parametrize("residual_ss", [0.0, 1.0, 1e-30])
def test_zero_total_ss_is_nan_without_a_warning(residual_ss, y_ss):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = r_squared(residual_ss, 0.0, y_ss, 50)
        numpy_result = r_squared(np.float64(residual_ss), np.float64(0.0), y_ss, 50)
    is_nan = np.isnan(result)
    numpy_is_nan = np.isnan(numpy_result)
    is_float = isinstance(result, float)
    assert is_nan
    assert numpy_is_nan
    assert is_float


@pytest.mark.parametrize("nobs", [1, 10, 100, 1000, 10**6, 10**8])
def test_limit_of_the_total_ss(nobs):
    y_ss = 123.0
    limit = (max(nobs, 100) * EPS) ** 2 * y_ss
    below = total_ss_is_zero(0.5 * limit, y_ss, nobs)
    at_limit = total_ss_is_zero(limit, y_ss, nobs)
    above = total_ss_is_zero(2 * limit, y_ss, nobs)
    assert below
    assert at_limit
    assert not above
    result = r_squared(1.0, 0.5 * limit, y_ss, nobs)
    is_nan = np.isnan(result)
    assert is_nan
    result = r_squared(1.0, 2 * limit, y_ss, nobs)
    assert_allclose(result, 1 - 1.0 / (2 * limit))


def test_total_ss_that_is_not_centered_is_zero_only_if_it_is_zero():
    # The total sum of squares of a model without a constant is the sum of squares of
    # the dependent variable, however small it is
    for scale in (1e-100, 1e-10, 1.0, 1e100):
        y = scale * np.array([1.0, -2.0, 3.0])
        y_ss = float(y @ y)
        is_zero = total_ss_is_zero(y_ss, y_ss, 3)
        zero_is_zero = total_ss_is_zero(0.0, y_ss, 3)
        assert not is_zero
        assert zero_is_zero
    all_zero = total_ss_is_zero(0.0, 0.0, 3)
    assert all_zero


def test_scale_of_the_data_does_not_matter():
    for scale in (1e-100, 1e-3, 1.0, 1e5, 1e100):
        noise = total_ss_is_zero(scale * 3e-33, scale * 10.0, 100)
        variation = total_ss_is_zero(scale * 3e-5, scale * 10.0, 100)
        assert noise
        assert not variation


@pytest.mark.parametrize("value", [0.1, 1 / 3, np.pi, 1e-10, 1e10, 123456.789, 5.0])
@pytest.mark.parametrize("nobs", [3, 25, 480, 20000, 150000])
def test_rounding_error_of_the_mean_of_a_constant(value, nobs):
    # The deviations from the mean of a variable that does not vary are rounding error
    y = np.full(nobs, value)
    deviations = y - y.mean()
    zero = total_ss_is_zero(float(deviations @ deviations), float(y @ y), nobs)
    assert zero
    # With weights, which have a mean of 1
    rg = np.random.default_rng(nobs)
    w = rg.uniform(0.2, 3, nobs)
    w /= w.mean()
    mean = (w * y).sum() / w.sum()
    deviations = np.sqrt(w) * (y - mean)
    y_ss = float((w * y**2).sum())
    zero = total_ss_is_zero(float(deviations @ deviations), y_ss, nobs)
    assert zero
    # The mean computed by a matrix product, as the models do, has more error with more
    # data
    mean = float((np.ones((1, nobs)) @ y[:, None] / nobs)[0, 0])
    deviations = y - mean
    zero = total_ss_is_zero(float(deviations @ deviations), float(y @ y), nobs)
    assert zero


@pytest.mark.parametrize("nobs", [20, 500, 20000])
@pytest.mark.parametrize("offset", [0.0, 1.0, 1e3, 1e6])
def test_data_that_varies_is_not_zero(nobs, offset):
    # A standard deviation of 1 and a mean of up to 1e6 is 1e-6 of the size of the data
    y = offset + np.random.default_rng(0).standard_normal(nobs)
    deviations = y - y.mean()
    zero = total_ss_is_zero(float(deviations @ deviations), float(y @ y), nobs)
    assert not zero


@pytest.mark.parametrize("relative_sd", [1e-4, 1e-8, 1e-11])
def test_variation_far_above_rounding_error_is_not_zero(relative_sd):
    nobs = 1000
    rg = np.random.default_rng(1)
    y = 7.0 * (1 + relative_sd * rg.standard_normal(nobs))
    deviations = y - y.mean()
    zero = total_ss_is_zero(float(deviations @ deviations), float(y @ y), nobs)
    assert not zero


def test_nan_total_ss_is_nan():
    result = r_squared(1.0, float("nan"), 1.0, 10)
    is_nan = np.isnan(result)
    assert is_nan
    result = r_squared(float("nan"), 1.0, 1.0, 10)
    is_nan = np.isnan(result)
    assert is_nan
