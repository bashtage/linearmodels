"""
has_constant is exact for a matrix of any rank

The column of ones is in the column space of x if and only if x has a constant. The test
used to compare the rank of x with the rank of x and a column of ones, and read every
matrix that did not have full column rank as having a constant. Duplicated columns, a
column of zeros, or a column that is a combination of the others, which are all allowed
if the models do not check the rank, then gave a constant to regressors that had none.
A rank that was too large, such as the number of columns of a matrix that is not full
rank, hid the constant of dummy variables that add to one.
"""

import numpy as np
import pytest

from linearmodels.shared.linalg import has_constant

NOBS = 60


def reference_rank_comparison(x):
    """
    The algorithm that has_constant used, which is exact if x has full column rank

    The column of ones is in the column space if it does not add to the rank of x,
    unless x has as many columns as rows or more.
    """
    if np.any(np.all(x == 1, axis=0)):
        return True, int(np.argmax(np.all(x == 1, axis=0)))
    is_single_valued = (np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)
    if np.any(is_single_valued):
        return True, int(np.argmax(is_single_valued))
    nobs = x.shape[0]
    aug_rank = np.linalg.matrix_rank(np.c_[np.ones((nobs, 1)), x])
    rank = np.linalg.matrix_rank(x)
    const = bool((aug_rank == rank) and nobs > x.shape[1])
    loc = -1
    if const:
        loc = int(np.argmin(x.var(0) / np.abs(x).max(0)))
    return const, loc


def dummies(rg, nobs, ngroups):
    """Dummy variables for ngroups categories, which all appear and add to one"""
    group = np.arange(nobs) % ngroups
    rg.shuffle(group)
    return (group[:, None] == np.arange(ngroups)).astype(float)


def normal(rg, nobs, ncols):
    return rg.standard_normal((nobs, ncols))


def design(kind, rg=None):
    """
    Regressors for a kind of design, and whether they have a constant

    The designs do not have full column rank, except for the last ones.
    """
    rg = np.random.default_rng(0) if rg is None else rg
    d = dummies(rg, NOBS, 3)
    z = normal(rg, NOBS, 3)
    zero = np.zeros((NOBS, 1))
    designs = {
        # Rank deficient, with a constant in the column space
        "dummies and a duplicate dummy": (np.c_[d, d[:, [0]]], True),
        "dummies and a scaled duplicate": (np.c_[d, 4.0 * d[:, [1]]], True),
        "dummies and a column of zeros": (np.c_[d, zero], True),
        "dummies, regressors and their sum": (np.c_[d, z, z.sum(1)], True),
        "dummies and a duplicate of a regressor": (np.c_[d, z, z[:, [0]]], True),
        "only dummies, many duplicates": (np.c_[d, d, d], True),
        "the last dummy twice, as one minus the others": (
            np.c_[d[:, :2], z[:, [0]], 1 - d[:, [0]] - d[:, [1]], d[:, [2]]],
            True,
        ),
        # Rank deficient, without a constant in the column space
        "regressor and a duplicate": (np.c_[z[:, [0]], z[:, [0]]], False),
        "regressors and a column of zeros": (np.c_[z, zero], False),
        "regressors and their sum": (np.c_[z, z.sum(1)], False),
        "a column of zeros": (zero, False),
        "regressors, zeros and a duplicate": (np.c_[z, zero, z[:, [1]]], False),
        "missing dummy and a duplicate": (np.c_[d[:, :2], d[:, [0]]], False),
        "missing dummy and a column of zeros": (np.c_[d[:, :2], zero], False),
        "dummy, regressor and their combination": (
            np.c_[d[:, [0]], z[:, [0]], d[:, [0]] - 2 * z[:, [0]]],
            False,
        ),
        # Full column rank
        "dummies": (d, True),
        "dummies that add to 2": (2.0 * d, True),
        "dummies and regressors": (np.c_[d, z], True),
        "one regressor": (z[:, [0]], False),
        "regressors": (z, False),
        "all but one dummy": (np.c_[d[:, :2], z], False),
    }
    return designs[kind]


RANK_DEFICIENT_CONSTANT = [
    "dummies and a duplicate dummy",
    "dummies and a scaled duplicate",
    "dummies and a column of zeros",
    "dummies, regressors and their sum",
    "dummies and a duplicate of a regressor",
    "only dummies, many duplicates",
    "the last dummy twice, as one minus the others",
]
RANK_DEFICIENT_NO_CONSTANT = [
    "regressor and a duplicate",
    "regressors and a column of zeros",
    "regressors and their sum",
    "a column of zeros",
    "regressors, zeros and a duplicate",
    "missing dummy and a duplicate",
    "missing dummy and a column of zeros",
    "dummy, regressor and their combination",
]
FULL_RANK = [
    "dummies",
    "dummies that add to 2",
    "dummies and regressors",
    "one regressor",
    "regressors",
    "all but one dummy",
]


@pytest.mark.parametrize("kind", RANK_DEFICIENT_CONSTANT + RANK_DEFICIENT_NO_CONSTANT)
def test_rank_deficient_design_is_not_full_rank(kind):
    # The cases are rank deficient, so that they test what they say they test
    x, _ = design(kind)
    rank = np.linalg.matrix_rank(x)
    assert rank < x.shape[1]


@pytest.mark.parametrize("kind", FULL_RANK)
def test_full_rank_design_is_full_rank(kind):
    x, _ = design(kind)
    rank = np.linalg.matrix_rank(x)
    assert rank == x.shape[1]


@pytest.mark.parametrize(
    "kind", RANK_DEFICIENT_CONSTANT + RANK_DEFICIENT_NO_CONSTANT + FULL_RANK
)
def test_constant_of_any_rank(kind):
    x, expected = design(kind)
    const, loc = has_constant(x)
    assert const is expected
    if expected:
        assert 0 <= loc < x.shape[1]
        # The location is that of a column that varies, not of a column of zeros
        not_zeros = np.any(x[:, loc] != 0)
        assert not_zeros
    else:
        assert loc == -1


@pytest.mark.parametrize("kind", RANK_DEFICIENT_CONSTANT + RANK_DEFICIENT_NO_CONSTANT)
def test_rank_argument_is_not_used(kind):
    # check_rank=False passes the number of columns, which is not the rank
    x, expected = design(kind)
    k = x.shape[1]
    expected_result = has_constant(x)
    for x_rank in (k, np.linalg.matrix_rank(x), 1, 0):
        result = has_constant(x, x_rank)
        assert result == expected_result
    const, _ = has_constant(x, k)
    assert const is expected


@pytest.mark.parametrize("kind", RANK_DEFICIENT_CONSTANT + RANK_DEFICIENT_NO_CONSTANT)
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_constant_does_not_depend_on_the_size_of_the_columns(kind, scale):
    x, expected = design(kind)
    rg = np.random.default_rng(1)
    sizes = np.exp(rg.uniform(-1, 1, x.shape[1])) * scale
    const, _ = has_constant(x * sizes)
    assert const is expected
    sizes[::2] = 1.0
    const, _ = has_constant(x * sizes)
    assert const is expected


@pytest.mark.parametrize("kind", RANK_DEFICIENT_CONSTANT)
def test_constant_location_of_rank_deficient_dummies(kind):
    x, _ = design(kind)
    _, loc = has_constant(x)
    # The column that varies the least relative to its size, ignoring columns of zeros.
    # Which of the columns that are tied is chosen depends on rounding
    nonzero = np.any(x != 0, axis=0)
    normed_var = np.full(x.shape[1], np.inf)
    normed_var[nonzero] = x[:, nonzero].var(0) / np.abs(x[:, nonzero]).max(0)
    smallest = normed_var.min()
    assert normed_var[loc] <= smallest * (1 + 1e-12)


def test_zero_column_is_not_the_location():
    d = dummies(np.random.default_rng(0), NOBS, 3)
    x = np.c_[np.zeros(NOBS), d]
    const, loc = has_constant(x)
    assert const
    # The normalized variance of the column of zeros is not defined
    assert loc in (1, 2, 3)


def test_explicit_constant_is_found_first():
    rg = np.random.default_rng(2)
    d = dummies(rg, NOBS, 3)
    z = normal(rg, NOBS, 2)
    ones = np.ones((NOBS, 1))
    zeros = np.zeros((NOBS, 1))
    # A constant column is found before one that is in the column space
    result = has_constant(np.c_[d, z, ones])
    assert result == (True, 5)
    result = has_constant(np.c_[d, ones])
    assert result == (True, 3)
    result = has_constant(np.c_[z, ones, d])
    assert result == (True, 2)
    result = has_constant(np.c_[z, 3.0 * ones, d])
    assert result == (True, 2)
    # Columns of ones come before other constants
    result = has_constant(np.c_[z, 3.0 * ones, ones])
    assert result == (True, 3)
    # Duplicated constants, with rank deficiency
    result = has_constant(np.c_[z, ones, ones, 2 * ones])
    assert result == (True, 2)
    result = has_constant(np.c_[zeros, z, 7 * ones])
    assert result == (True, 3)
    # The rank does not change it
    result = has_constant(np.c_[z, z, ones], x_rank=5)
    assert result == (True, 4)


@pytest.mark.parametrize("nobs", [30, 100])
@pytest.mark.parametrize("seed", range(6))
def test_full_rank_results_are_unchanged(nobs, seed):
    # The full rank cases that the rank comparison got right, which is all of them
    rg = np.random.default_rng(seed)
    ones = np.ones((nobs, 1))
    for ngroups in (2, 3, 7):
        d = dummies(rg, nobs, ngroups)
        for ncont in (0, 1, 4):
            z = normal(rg, nobs, ncont)
            candidates = [
                np.c_[d, z],
                np.c_[d[:, 1:], z],
                np.c_[z, d[:, :-1]],
                np.c_[z, ones, d[:, 1:]],
                np.c_[d * rg.uniform(1, 4, ngroups), z],
            ]
            candidates += [
                x * np.exp(rg.uniform(-3, 3, x.shape[1])) for x in candidates
            ]
            for x in candidates:
                if x.shape[1] == 0:
                    continue
                rank = np.linalg.matrix_rank(x)
                assert rank == x.shape[1]
                result = has_constant(x)
                reference = reference_rank_comparison(x)
                assert result == reference
    # Square, tall, and wide
    for shape in ((nobs, 1), (nobs, nobs - 1), (nobs, nobs), (nobs // 3, nobs)):
        x = normal(rg, *shape)
        result = has_constant(x)
        reference = reference_rank_comparison(x)
        assert result == reference
        # A saturated matrix spans the column of ones, which is not a constant
        if x.shape[1] >= x.shape[0]:
            assert result == (False, -1)


def test_matrix_that_spans_everything_has_no_constant():
    rg = np.random.default_rng(3)
    # The rank is the number of rows, so that the column of ones is in the column space
    result = has_constant(np.eye(5))
    assert result == (False, -1)
    result = has_constant(normal(rg, 6, 6))
    assert result == (False, -1)
    result = has_constant(normal(rg, 6, 9))
    assert result == (False, -1)
    # A wide matrix of lower rank has a constant if the column of ones is in its space
    wide = np.c_[dummies(rg, 12, 3), normal(rg, 12, 2) @ normal(rg, 2, 9)]
    rank = np.linalg.matrix_rank(wide)
    assert rank < 12
    const, _ = has_constant(wide)
    assert const
    wide_no_constant = normal(rg, 12, 3) @ normal(rg, 3, 14)
    rank = np.linalg.matrix_rank(wide_no_constant)
    assert rank < 12
    result = has_constant(wide_no_constant)
    assert result == (False, -1)


@pytest.mark.parametrize(
    ("variation", "expected"),
    [(1e-13, True), (1e-11, True), (1e-3, False), (1e-1, False), (1.0, False)],
)
def test_column_that_varies_by_less_than_the_tolerance_is_a_constant(
    variation, expected
):
    # A column of 1 + variation * noise is in the column space of the column of ones
    # only if the noise is small relative to the tolerance, about 1.5e-8
    z = np.random.default_rng(4).standard_normal(200)
    x = (1 + variation * z)[:, None]
    const, _ = has_constant(x)
    assert const is expected


def test_columns_are_not_modified():
    rg = np.random.default_rng(5)
    x = np.c_[dummies(rg, NOBS, 3), normal(rg, NOBS, 2)]
    original = x.copy()
    has_constant(x)
    np.testing.assert_array_equal(x, original)


def test_integer_and_bool_columns():
    rg = np.random.default_rng(6)
    d = dummies(rg, NOBS, 3)
    expected = has_constant(d)
    result = has_constant(d.astype(int))
    assert result == expected
    result = has_constant(d.astype(bool))
    assert result == expected
    const, _ = has_constant(d.astype(np.float32))
    assert const
    result = has_constant(d[:, :2].astype(int))
    assert result == (False, -1)


def test_no_columns():
    result = has_constant(np.empty((NOBS, 0)))
    assert result == (False, -1)
