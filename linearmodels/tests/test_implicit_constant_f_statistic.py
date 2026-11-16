"""Model F tests of regressors that jointly span a constant."""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.utility import AbsorbingEffectWarning
from linearmodels.shared.hypotheses import InvalidTestStatistic
from linearmodels.system import IV3SLS, SUR, IVSystemGMM


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(48)
    index = pd.MultiIndex.from_product([range(32), range(8)], names=["entity", "time"])
    nobs = len(index)
    group = (index.codes[0] + index.codes[1]) % 3 == 0
    x = rng.normal(size=nobs)
    z = rng.normal(size=nobs)
    endog = z + rng.normal(size=nobs)
    error = rng.normal(size=nobs)
    data = pd.DataFrame(
        {
            "constant": 1.0,
            "a": (~group).astype(float),
            "b": group.astype(float),
            "x": x,
            "y": 1.3 + 0.16 * group + 0.15 * x + error,
            "other": -0.7 + 0.2 * x + 0.4 * error + rng.normal(size=nobs),
            "z": z,
            "endog": endog,
            "weights": rng.uniform(0.6, 1.8, size=nobs),
        },
        index=index,
    )
    return data


@pytest.mark.parametrize(
    "model", [PooledOLS, PanelOLS, BetweenOLS, RandomEffects, FamaMacBeth]
)
@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize(
    ("cov_type", "weighted"), [("unadjusted", False), ("robust", True)]
)
@pytest.mark.parametrize("scales", [(1.0, 1.0), (-2.0, 5.0)])
def test_panel_f_statistics_with_implicit_constant(
    panel_data, model, debiased, cov_type, weighted, scales
):
    data = panel_data
    design = data[["a", "b", "x"]] * [*scales, 1.0]
    weights = data["weights"] if weighted else None
    options = {"entity_effects": True} if model is PanelOLS else {}
    fit = model(data["y"], design, weights=weights, **options).fit(
        cov_type=cov_type, debiased=debiased
    )
    explicit = model(
        data["y"], data[["constant", "b", "x"]], weights=weights, **options
    ).fit(cov_type=cov_type, debiased=debiased)
    restriction = np.array([[scales[0], -scales[1], 0], [0, 0, 1]])
    expected = fit.wald_test(restriction).stat
    if debiased:
        expected /= 2
        if model is FamaMacBeth:
            periods = fit.df_resid + 1
            expected *= (periods - 2) / (periods - 1)

    robust = fit.f_statistic_robust
    assert robust.df == 2
    assert robust.df_denom == explicit.f_statistic_robust.df_denom
    assert_allclose(robust.stat, expected, rtol=1e-9)
    assert_allclose(robust.stat, explicit.f_statistic_robust.stat, rtol=1e-9)
    assert_allclose(robust.pval, explicit.f_statistic_robust.pval, rtol=1e-9)
    assert_allclose(fit.f_statistic.stat, explicit.f_statistic.stat, rtol=1e-9)


@pytest.mark.parametrize("fit_options", [{}, {"use_lsdv": True}, {"use_lsmr": True}])
@pytest.mark.parametrize("weighted", [False, True])
def test_panel_f_statistics_when_an_implicit_constant_column_is_absorbed(
    panel_data, fit_options, weighted
):
    data = panel_data
    data["b"] = (data.index.codes[0] % 2).astype(float)
    data["a"] = 1 - data["b"]
    weights = data["weights"] if weighted else None
    with pytest.warns(AbsorbingEffectWarning, match="b"):
        fit = PanelOLS(
            data["y"],
            data[["a", "b", "x"]],
            weights=weights,
            entity_effects=True,
            drop_absorbed=True,
        ).fit(**fit_options)
    explicit = PanelOLS(
        data["y"], data[["constant", "x"]], weights=weights, entity_effects=True
    ).fit(**fit_options)
    for name in ["f_statistic", "f_statistic_robust"]:
        stat, reference = getattr(fit, name), getattr(explicit, name)
        assert stat.df == reference.df == 1
        assert stat.df_denom == reference.df_denom
        assert_allclose(stat.stat, reference.stat, rtol=1e-7)


def test_panel_f_statistic_with_implicit_constant_and_singular_covariance(panel_data):
    data = panel_data
    fit = PooledOLS(data["y"] * 0.0, data[["a", "b", "x"]]).fit()
    assert isinstance(fit.f_statistic_robust, InvalidTestStatistic)


@pytest.mark.parametrize("model", [SUR, IV3SLS, IVSystemGMM])
@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_system_equation_f_statistic_with_implicit_constant(
    panel_data, model, debiased, weighted
):
    data = panel_data
    y = data["y"]
    weights = {"weights": data["weights"]} if weighted else {}
    first = {"dependent": y, **weights}
    if model is not SUR:
        first = {
            **first,
            "dependent": y + 0.6 * data["endog"],
            "endog": data[["endog"]],
            "instruments": data[["z"]],
        }
    explicit_design = data[["constant", "b", "x"]]
    other = {"dependent": data["other"], "exog": explicit_design, **weights}
    scales = (-2.0, 5.0)
    design = data[["a", "b", "x"]] * [*scales, 1.0]
    result = model({"first": {**first, "exog": design}, "other": other}).fit(
        cov_type="robust", debiased=debiased
    )
    reference = model(
        {"first": {**first, "exog": explicit_design}, "other": other}
    ).fit(cov_type="robust", debiased=debiased)
    equation = result.equations["first"]
    restriction = np.eye(len(equation.params))[1:]
    restriction[0, :2] = [scales[0], -scales[1]]
    diff = restriction @ equation.params.to_numpy()
    covariance = restriction @ equation.cov.to_numpy() @ restriction.T
    df = restriction.shape[0]
    expected = diff @ np.linalg.solve(covariance, diff) / (df if debiased else 1)
    stat = equation.f_statistic
    explicit = reference.equations["first"].f_statistic
    assert stat.df == explicit.df == df
    assert stat.df_denom == explicit.df_denom
    assert_allclose(stat.stat, expected, rtol=1e-9)
    assert_allclose(stat.stat, explicit.stat, rtol=1e-9)
    assert_allclose(stat.pval, explicit.pval, rtol=1e-9)
    assert_allclose(
        result.equations["other"].f_statistic.stat,
        reference.equations["other"].f_statistic.stat,
        rtol=1e-9,
    )
