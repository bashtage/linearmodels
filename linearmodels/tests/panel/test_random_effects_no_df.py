"""
RandomEffects where the regressions of the variance components have no degrees of freedom

The variance of the idiosyncratic error is the residual sum of squares of the within
regression divided by its degrees of freedom, nobs - N - rank, where rank is the rank
of the regressors once the means of the entities are removed. The variance of the
effects is estimated from the between regression, which has N - rank observations of
the means of the regressors left. The variances are not defined if there are no
degrees of freedom, e.g., if there is a single observation of each entity or fewer
entities than regressors. These models raised an error from the division, from NumPy or
from SciPy, or returned an infinite variance of the effects.
"""

import itertools

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import RandomEffects


def make_panel(counts, nvar, const, seed=0):
    rg = np.random.default_rng(seed)
    entities = np.repeat(np.arange(len(counts)), counts)
    times = np.concatenate([np.arange(count) for count in counts])
    index = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])
    x = pd.DataFrame(
        rg.standard_normal((entities.shape[0], nvar)),
        index=index,
        columns=[f"x{i}" for i in range(nvar)],
    )
    if const:
        x.insert(0, "const", 1.0)
    y = pd.Series(rg.standard_normal(entities.shape[0]), index=index, name="y")
    return y, x


def degrees_of_freedom(x):
    """The residual degrees of freedom of the within and the between regressions"""
    nobs = x.shape[0]
    means = x.groupby(level=0).mean()
    demeaned = x - x.groupby(level=0).transform("mean")
    within = nobs - means.shape[0] - np.linalg.matrix_rank(demeaned.to_numpy())
    between = means.shape[0] - np.linalg.matrix_rank(means.to_numpy())
    return within, between


def lsdv_variance(y, x):
    """Residual variance of the regression on the dummy variables of the entities"""
    codes = pd.factorize(x.index.get_level_values(0))[0]
    dummies = np.eye(codes.max() + 1)[codes]
    design = np.column_stack([x.to_numpy(), dummies])
    resid = y.to_numpy() - design @ np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
    return resid @ resid / (design.shape[0] - np.linalg.matrix_rank(design))


WITHIN = "variance of the idiosyncratic error cannot be estimated"
BETWEEN = "variance of the effects cannot be estimated"
BALANCED = [[t] * n for n in (1, 2, 3, 4) for t in (1, 2, 3)]
UNBALANCED = [[1, 2, 1], [1, 1, 2], [2, 2, 1], [1, 3, 2, 1], [3, 1, 4, 2, 2]]


def full_rank_designs():
    """The designs where the regressors are not collinear, as the model requires"""
    for counts, nvar, const in itertools.product(
        BALANCED + UNBALANCED, [1, 2, 3], [False, True]
    ):
        x = make_panel(counts, nvar, const)[1]
        if np.linalg.matrix_rank(x.to_numpy()) == x.shape[1]:
            yield counts, nvar, const


DESIGNS = list(full_rank_designs())
DESIGN_IDS = [f"{c}-{k}-{'const' if const else 'noconst'}" for c, k, const in DESIGNS]


@pytest.mark.parametrize("small_sample", [False, True])
@pytest.mark.parametrize(("counts", "nvar", "const"), DESIGNS, ids=DESIGN_IDS)
def test_variances_need_degrees_of_freedom(counts, nvar, const, small_sample):
    y, x = make_panel(counts, nvar, const)
    within, between = degrees_of_freedom(x)
    mod = RandomEffects(y, x)
    if within <= 0:
        with pytest.raises(ValueError, match=WITHIN):
            mod.fit(small_sample=small_sample)
    elif between <= 0:
        with pytest.raises(ValueError, match=BETWEEN):
            mod.fit(small_sample=small_sample)
    else:
        res = mod.fit(small_sample=small_sample)
        variances = res.variance_decomposition
        assert np.all(np.isfinite(variances))
        assert variances["Residual"] > 0
        assert variances["Effects"] >= 0
        assert np.all(np.isfinite(res.params))
        assert_allclose(variances["Residual"], lsdv_variance(y, x))


def test_each_cell_of_the_grid_is_used():
    # Some designs of the grid fail for each reason, and some are estimated
    reasons = []
    for counts, nvar, const in DESIGNS:
        within, between = degrees_of_freedom(make_panel(counts, nvar, const)[1])
        reasons.append("within" if within <= 0 else "between" if between <= 0 else "ok")
    assert {"within", "between", "ok"} == set(reasons)
    assert all(reasons.count(reason) >= 10 for reason in set(reasons))


@pytest.mark.parametrize("small_sample", [False, True])
def test_one_observation_of_each_entity(small_sample):
    y, x = make_panel([1] * 10, 1, True)
    # The 10 means leave nothing of the 10 observations, and the regressor does not vary
    with pytest.raises(ValueError, match=WITHIN) as exc_info:
        RandomEffects(y, x).fit(small_sample=small_sample)
    assert "at least 11 observations are needed" in str(exc_info.value)
    assert "10 observations of 10 entities" in str(exc_info.value)


@pytest.mark.parametrize("small_sample", [False, True])
def test_as_many_regressors_as_the_observations_of_the_entities_leave(small_sample):
    # 3 entities with 2 observations: 3 regressors use the 3 degrees of freedom
    y, x = make_panel([2, 2, 2], 3, False)
    with pytest.raises(ValueError, match=WITHIN) as exc_info:
        RandomEffects(y, x).fit(small_sample=small_sample)
    assert "rank of 3" in str(exc_info.value)
    assert "at least 7 observations are needed" in str(exc_info.value)


@pytest.mark.parametrize("small_sample", [False, True])
@pytest.mark.parametrize("counts", [[10], [3, 3], [4, 4]])
def test_not_more_entities_than_the_rank_of_the_means(counts, small_sample):
    # The constant and a regressor have a rank of 2 in the means of 1 or 2 entities
    y, x = make_panel(counts, 1, True)
    with pytest.raises(ValueError, match=BETWEEN) as exc_info:
        RandomEffects(y, x).fit(small_sample=small_sample)
    rank = min(len(counts), 2)
    assert f"{len(counts)} entities" in str(exc_info.value)
    assert f"at least {rank + 1} entities are needed" in str(exc_info.value)


@pytest.mark.parametrize("small_sample", [False, True])
def test_one_more_entity_than_the_rank_of_the_means(small_sample):
    # The smallest panel that can be estimated: 3 entities, a constant and a regressor
    y, x = make_panel([3, 3, 3], 1, True)
    assert degrees_of_freedom(x) == (9 - 3 - 1, 1)
    res = RandomEffects(y, x).fit(small_sample=small_sample)
    assert np.all(np.isfinite(res.variance_decomposition))
    assert_allclose(res.variance_decomposition["Residual"], lsdv_variance(y, x))
