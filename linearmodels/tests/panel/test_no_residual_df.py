"""
Models without residual degrees of freedom

A model that has as many parameters as observations, after the effects are removed,
raised the ZeroDivisionError of the scale of the covariance, n / (n - k), or of an
F-statistic. The covariance of such a model is not defined, and the error has to say
so. The F-statistic of the model is not defined either, which is the result if the
covariance is not debiased. The F-statistic that tests that the effects are zero is not
defined if the effects do not use any degree of freedom: a single entity with a
constant, or a single period with time effects.
"""

import numpy as np
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
)
from linearmodels.shared.exceptions import InferenceUnavailableWarning
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
)

NO_DF = "no residual degrees of freedom"


def make_panel(nentity, nperiod, nvar, seed=0):
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    nobs = nentity * nperiod
    x = pd.DataFrame(
        rg.standard_normal((nobs, nvar)),
        index=index,
        columns=[f"x{i}" for i in range(nvar)],
    )
    y = pd.Series(rg.standard_normal(nobs), index=index, name="y")
    return y, x


def saturated_pooled():
    # 2 observations, a constant and a regressor
    y, x = make_panel(1, 2, 1)
    x.insert(0, "const", 1.0)
    return y, x


def saturated_between():
    # 2 entities, a constant and a regressor in the regression of the means
    y, x = make_panel(2, 3, 1)
    x.insert(0, "const", 1.0)
    return y, x


# Each model has as many parameters as observations once the effects are removed
SATURATED = {
    "pooled": lambda: PooledOLS(*saturated_pooled()),
    # 2 entities with 2 periods, 2 regressors and 2 effects
    "entity-effects": lambda: PanelOLS(*make_panel(2, 2, 2), entity_effects=True),
    "time-effects": lambda: PanelOLS(*make_panel(2, 2, 2), time_effects=True),
    "between": lambda: BetweenOLS(*saturated_between()),
    # 1 entity with 3 periods has 2 differences, and there are 2 regressors
    "first-difference": lambda: FirstDifferenceOLS(*make_panel(1, 3, 2)),
}
# BetweenOLS does not have the kernel covariances
COVARIANCES = [
    (model, cov_type)
    for model in SATURATED
    for cov_type in (["robust"] if model == "between" else ["robust", "kernel", "ac"])
]


@pytest.mark.parametrize("model", list(SATURATED))
def test_saturated_model_raises_value_error(model):
    mod = SATURATED[model]()
    with pytest.raises(ValueError, match=NO_DF) as exc_info:
        mod.fit()
    assert exc_info.type is ValueError


@pytest.mark.parametrize(("model", "cov_type"), COVARIANCES)
def test_saturated_model_with_other_covariances(model, cov_type):
    with pytest.raises(ValueError, match=NO_DF):
        SATURATED[model]().fit(cov_type=cov_type)


@pytest.mark.parametrize("model", list(SATURATED))
def test_saturated_model_that_is_not_debiased(model):
    # The estimates and the covariance do not need the degrees of freedom. The
    # F-statistic does, and it is not defined
    res = SATURATED[model]().fit(debiased=False)
    assert np.all(np.isfinite(res.params))
    assert isinstance(res.f_statistic, InvalidTestStatistic)
    assert np.isnan(res.f_statistic.stat)
    assert NO_DF in str(res.f_statistic)
    assert str(res.summary)


def test_famamacbeth_with_as_many_observations_as_parameters():
    # One cross-section that has as many entities as regressors
    y, x = make_panel(2, 1, 2)
    with pytest.warns(InferenceUnavailableWarning):
        mod = FamaMacBeth(y, x)
    with pytest.raises(ValueError, match=NO_DF):
        mod.fit()


def single_entity(const):
    y, x = make_panel(1, 8, 2)
    if const:
        x.insert(0, "const", 1.0)
    return y, x


def single_period(const):
    y, x = make_panel(8, 1, 2)
    if const:
        x.insert(0, "const", 1.0)
    return y, x


@pytest.mark.parametrize("const", [False, True])
def test_single_entity_with_entity_effects(const):
    # The effect is the constant: the slopes are those of the regression with a constant
    y, x = single_entity(const)
    res = PanelOLS(y, x, entity_effects=True).fit()
    slopes = ["x0", "x1"]
    pooled_x = x[slopes].copy()
    pooled_x.insert(0, "const", 1.0)
    pooled = PooledOLS(y, pooled_x).fit()
    np.testing.assert_allclose(res.params[slopes], pooled.params[slopes])
    # The effects do not use any degree of freedom: there is nothing to test
    assert isinstance(res.f_pooled, InapplicableTestStatistic)
    assert "effects" in str(res.f_pooled)
    assert np.isnan(res.f_pooled.stat)
    assert str(res.summary)


@pytest.mark.parametrize("const", [False, True])
def test_single_period_with_time_effects(const):
    y, x = single_period(const)
    res = PanelOLS(y, x, time_effects=True).fit()
    assert isinstance(res.f_pooled, InapplicableTestStatistic)
    assert np.all(np.isfinite(res.params))
