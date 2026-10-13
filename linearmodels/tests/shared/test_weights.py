"""
Tests of the validation and the normalization of weights

The weights have to be strictly positive and finite, with a NaN for a missing value. They
are scaled to have a mean of 1. The mean of weights of 1e308 overflows, so that the
weights were divided by infinity and became 0, and the weights that are scaled have to
be finite for any finite weights.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from linearmodels.shared.weights import check_weights, normalize_weights

EXTREME = {
    "huge": np.full(10, 1e308),
    "tiny": np.full(10, 1e-308),
    "subnormal": np.full(10, 5e-324),
    "maximum": np.full(10, np.finfo(float).max),
    "mixed 1e308 and 1": np.r_[np.full(5, 1e308), np.ones(5)],
    "mixed 1e308 and 1e-308": np.r_[np.full(5, 1e308), np.full(5, 1e-308)],
    "mixed 1e300 and 1e-300": np.r_[np.full(5, 1e300), np.full(5, 1e-300)],
    "mixed scales": np.array([1e308, 1e-308, 1.0, 1e150, 1e-150, 3.0, 7e20, 2e-20]),
    "one huge": np.r_[1e308, np.ones(9)],
    "one tiny": np.r_[1e-308, np.ones(9)],
}
MESSAGE = r"^weights must be strictly positive and finite\.$"


@pytest.mark.parametrize("bad", [0.0, -0.0, -1.0, -1e-300, -np.inf, np.inf])
@pytest.mark.parametrize("shape", [(5,), (5, 1), (2, 3)])
def test_check_rejects_weights_that_are_not_strictly_positive_and_finite(bad, shape):
    weights = np.ones(shape)
    weights.flat[2] = bad
    with pytest.raises(ValueError, match=MESSAGE):
        check_weights(weights)


def test_check_names_the_input():
    weights = np.array([1.0, np.inf])
    message = r"^weights in equation 'y1' must be strictly positive and finite\.$"
    with pytest.raises(ValueError, match=message):
        check_weights(weights, "weights in equation 'y1'")
    with pytest.raises(ValueError, match=r"^w must be strictly positive and finite\.$"):
        check_weights(weights, "w")


@pytest.mark.parametrize("name", EXTREME)
def test_check_accepts_finite_positive_weights(name):
    check_weights(EXTREME[name])
    check_weights(EXTREME[name][:, None])


def test_check_ignores_missing_values():
    # A NaN is a missing value, which is dropped with its observation
    check_weights(np.array([1.0, np.nan, 2.0]))
    check_weights(np.full(3, np.nan))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
def test_check_other_types(dtype):
    check_weights(np.arange(1, 5).astype(dtype))
    with pytest.raises(ValueError, match="strictly positive"):
        check_weights(np.arange(0, 5).astype(dtype))
    check_weights([1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        check_weights([1.0, -2.0])


@pytest.mark.parametrize("shape", [(50,), (50, 1)])
@pytest.mark.parametrize("seed", range(5))
def test_normalized_weights_are_weights_divided_by_their_mean(shape, seed):
    rg = np.random.default_rng(seed)
    weights = rg.chisquare(3, shape) + 0.01
    normalized = normalize_weights(weights)
    assert normalized.shape == shape
    assert normalized.dtype == np.float64
    assert_allclose(normalized, weights / weights.mean(), rtol=1e-14)
    assert_allclose(normalized.mean(), 1.0, rtol=1e-14)


def test_normalize_does_not_modify_its_input():
    weights = np.array([1.0, 2.0, np.nan, 4.0])
    original = weights.copy()
    normalize_weights(weights)
    assert_array_equal(weights, original)


@pytest.mark.parametrize("value", [1.0, 0.1, 3.0, 1e-5, 12345.678])
def test_equal_weights_are_exactly_one(value):
    # 0.1 is not exactly representable, and the mean of copies of it is not exactly 0.1
    normalized = normalize_weights(np.full(37, value))
    assert_array_equal(normalized, np.ones(37))


def test_normalize_ignores_missing_values():
    weights = np.array([1.0, 2.0, np.nan, 3.0])
    normalized = normalize_weights(weights)
    expected = np.array([0.5, 1.0, np.nan, 1.5])
    assert_allclose(normalized, expected)
    missing = np.isnan(normalized)
    assert_array_equal(missing, [False, False, True, False])
    assert_allclose(np.nanmean(normalized), 1.0)


def test_normalize_all_missing_and_empty():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalized = normalize_weights(np.full(4, np.nan))
        empty = normalize_weights(np.empty((0, 1)))
    all_missing = np.isnan(normalized).all()
    assert normalized.shape == (4,)
    assert all_missing
    assert empty.shape == (0, 1)


def test_the_plain_mean_of_huge_weights_overflows():
    # The case that the normalization has to handle: the sum of the weights is infinite
    weights = np.full(10, 1e308)
    with np.errstate(over="ignore"):
        plain_mean = weights.mean()
        old = weights / plain_mean
    overflow = np.isinf(plain_mean)
    assert overflow
    assert_array_equal(old, np.zeros(10))
    normalized = normalize_weights(weights)
    assert_array_equal(normalized, np.ones(10))


@pytest.mark.parametrize("name", EXTREME)
def test_normalized_weights_are_finite(name):
    weights = EXTREME[name]
    for values in (weights, weights[:, None], weights[::-1]):
        normalized = normalize_weights(values)
        all_finite = np.isfinite(normalized).all()
        assert all_finite
        non_negative = (normalized >= 0).all()
        assert non_negative
        # The largest weight is still the largest, and the mean is 1
        largest = normalized.argmax()
        expected_largest = values.argmax()
        assert largest == expected_largest
        assert_allclose(normalized.mean(), 1.0, rtol=1e-14)


def test_huge_and_tiny_weights():
    normalized = normalize_weights(np.r_[np.full(5, 1e308), np.full(5, 1e-308)])
    # The tiny weights are too small next to the large weights to be represented
    assert_array_equal(normalized, np.r_[np.full(5, 2.0), np.zeros(5)])
    normalized = normalize_weights(np.array([1e308, 5e307, 1e307]))
    expected = np.array([1.0, 0.5, 0.1]) / (1.6 / 3)
    assert_allclose(normalized, expected, rtol=1e-14)


@pytest.mark.parametrize("seed", range(5))
def test_normalization_does_not_depend_on_the_scale_of_the_weights(seed):
    rg = np.random.default_rng(seed)
    weights = 10.0 ** rg.uniform(-3, 3, 40)
    expected = normalize_weights(weights)
    for scale in (1e-300, 1e-100, 1e-3, 1e5, 1e100, 1e300):
        scaled = weights * scale
        # The scaled weights have to be normal numbers
        in_range = scaled.max() < 1e308 and scaled.min() > 1e-300
        if not in_range:
            continue
        assert_allclose(normalize_weights(scaled), expected, rtol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_random_weights_over_many_scales_are_finite(seed):
    rg = np.random.default_rng(seed)
    exponents = rg.uniform(-307, 307, 200)
    weights = 10.0**exponents
    normalized = normalize_weights(weights)
    all_finite = np.isfinite(normalized).all()
    assert all_finite
    assert_allclose(normalized.mean(), 1.0, rtol=1e-14)
    # The weights that are not too small next to the largest are scaled by one number
    keep = weights > weights.max() * 1e-200
    ratio = normalized[keep] / weights[keep]
    assert_allclose(ratio, ratio[0], rtol=1e-13)
