"""
The variance components of RandomEffects

The idiosyncratic variance is the residual variance of the within regression, and
the variance of the effects is estimated from the residuals of the between
regression. The degrees of freedom of the variances are those of the regressions:
the within regression has the coefficients of the regressors that vary within
entities and are not collinear once the means of the entities are removed, and the
between regression has the coefficients of the regressors that are not collinear
across entities. A time-invariant regressor, such as education in a wage panel, is
not in the within regression, and neither is a regressor that is collinear with the
dummy variables of the time periods within entities, such as experience. A
dummy variable of the time periods can be collinear with the constant in the between
regression. The degrees of freedom do not count them.

The references are

* the residual variance of the regression on the dummy variables of the entities,
  which is the idiosyncratic variance of Swamy and Arora;
* the estimators written out in NumPy using the ranks of the regressions;
* plm, in results/re-variance-components-reference.csv, for the wage panel of Vella
  and Verbeek. The R script that produced it is
  results/re-variance-components-reference.R. Stata's xtreg, re counts the
  coefficients in the same way: sigma_e of a model with a time-invariant regressor is
  that of xtreg, fe, which omits the regressor.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.linalg import qr

from linearmodels.datasets import wage_panel
from linearmodels.panel import PanelOLS, RandomEffects

RESULTS = os.path.join(os.path.split(os.path.abspath(__file__))[0], "results")
NPERIOD = 6
DUMMIES = [f"d{p}" for p in range(1, NPERIOD)]
DESIGNS = {
    "time-varying": ["const", "x1"],
    "no-constant": ["x1"],
    "time-invariant": ["const", "x1", "z"],
    "time-invariant-no-constant": ["x1", "z"],
    # exper is collinear with the dummies once the means of the entities are removed
    "collinear-with-dummies": ["const", "x1", "z", "exper", *DUMMIES],
}


def make_data(balance, seed=0, nentity=40):
    """
    A panel with a time-invariant regressor (z) and one that is collinear with the
    dummies of the time periods within entities (exper), as the experience in a
    wage panel. The unbalanced panels drop observations from the last three periods
    ("collinear-means") or from all periods
    """
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(NPERIOD)], names=["entity", "time"]
    )
    entity = index.codes[0]
    period = index.codes[1]
    nobs = nentity * NPERIOD
    effect = rg.standard_normal(nentity)[entity]
    x1 = 0.5 * effect + rg.standard_normal(nobs)
    z = rg.standard_normal(nentity)[entity]
    exper = period + rg.integers(0, 5, nentity)[entity]
    shock = (1 + 0.3 * np.abs(x1)) * rg.standard_normal(nobs)
    data = pd.DataFrame(
        {
            "y": 1 + 0.8 * x1 + 0.5 * z + 0.05 * exper + 1.2 * effect + shock,
            "const": 1.0,
            "x1": x1,
            "z": z,
            "exper": exper,
            "w": rg.chisquare(4, nobs) / 4 + 0.25,
        },
        index=index,
    )
    for p in range(1, NPERIOD):
        data[f"d{p}"] = (period == p).astype(float)
    if balance != "balanced":
        keep = rg.random((nentity, NPERIOD)) > 0.3
        if balance == "unbalanced-collinear-means":
            # The dummies of the first periods have the same mean in all entities
            keep[:, :3] = True
        for i in range(nentity):
            keep[i, rg.choice(NPERIOD, 3, replace=False)] = True
        data = data.loc[keep.ravel()]
    return data


@pytest.fixture(
    params=["balanced", "unbalanced", "unbalanced-collinear-means"], scope="module"
)
def data(request):
    return make_data(request.param)


def lsdv_variance(data, columns, weights):
    """
    The residual variance of the regression on the dummy variables of the entities

    The regressors of the model that are not identified by the dummy variables
    (the time-invariant ones, which they span) do not reduce the degrees of freedom.
    """
    dummies = pd.get_dummies(data.index.get_level_values(0)).astype(float)
    regressors = [c for c in columns if c != "const"]
    exog = np.column_stack([data[regressors].to_numpy(), dummies.to_numpy()])
    root_w = np.ones(data.shape[0])
    if weights is not None:
        root_w = np.sqrt((weights / weights.mean()).to_numpy())
    wexog, wy = root_w[:, None] * exog, root_w * data["y"].to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(wexog, wy, rcond=None)
    return float(np.sum((wy - wexog @ beta) ** 2)) / (data.shape[0] - rank)


def swar_reference(y, x, weights=None, small_sample=False):
    """
    The estimators of Swamy and Arora written out, with the ranks of the regressions

    Returns the idiosyncratic variance, the variance of the effects and the numbers
    of coefficients of the within and the between regressions
    """
    entity = y.index.get_level_values(0)
    weights = pd.Series(1.0, index=y.index) if weights is None else weights
    weights = weights / weights.mean()
    total = weights.groupby(entity).transform("sum")

    def weighted_means(frame):
        weighted = frame.mul(weights, axis=0)
        return weighted.groupby(entity).transform("sum").div(total, axis=0)

    root_w = np.sqrt(weights.to_numpy())
    xd = root_w[:, None] * (x - weighted_means(x)).to_numpy()
    yd = root_w * (y - weighted_means(y.to_frame()).iloc[:, 0]).to_numpy()
    beta, _, rank_within, _ = np.linalg.lstsq(xd, yd, rcond=None)
    nobs, nentity = x.shape[0], entity.nunique()
    sigma2_eps = float(np.sum((yd - xd @ beta) ** 2)) / (nobs - nentity - rank_within)
    # The regression of the means of the entities
    xb = weighted_means(x).groupby(entity).first().to_numpy()
    yb = weighted_means(y.to_frame()).iloc[:, 0].groupby(entity).first().to_numpy()
    count = y.groupby(entity).size().to_numpy().astype(float)
    rank_between = int(np.linalg.matrix_rank(xb))
    if not small_sample:
        resid = yb - xb @ np.linalg.lstsq(xb, yb, rcond=None)[0]
        tbar = nentity / np.sum(1 / count)
        sigma2_effects = float(np.sum(resid**2)) / (nentity - rank_between)
        sigma2_effects = max(0.0, sigma2_effects - sigma2_eps / tbar)
        return sigma2_eps, sigma2_effects, rank_within, rank_between
    # Baltagi and Chang (1994) for unbalanced panels: the means are weighted by the
    # number of observations. This is the regression of the data replaced by the
    # means of the entities. The trace uses columns that are not collinear
    root_t = np.sqrt(count)
    coef = np.linalg.lstsq(root_t[:, None] * xb, root_t * yb, rcond=None)[0]
    rss = float(np.sum(count * (yb - xb @ coef) ** 2))
    basis = qr(xb, pivoting=True, mode="economic")[2][:rank_between]
    xr = xb[:, basis]
    trace = np.trace(
        np.linalg.inv((count[:, None] * xr).T @ xr)
        @ ((count[:, None] * xr).T @ (count[:, None] * xr))
    )
    sigma2_effects = (rss - (nentity - rank_between) * sigma2_eps) / (nobs - trace)
    return sigma2_eps, max(0.0, float(sigma2_effects)), rank_within, rank_between


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("design", list(DESIGNS))
def test_idiosyncratic_variance_against_dummy_regression(data, design, weighted):
    columns = DESIGNS[design]
    weights = data["w"] if weighted else None
    res = RandomEffects(data["y"], data[columns], weights=weights).fit()
    expected = lsdv_variance(data, columns, weights)
    assert_allclose(res.variance_decomposition["Residual"], expected)


@pytest.mark.parametrize("design", ["time-invariant", "time-invariant-no-constant"])
def test_idiosyncratic_variance_is_that_of_the_fixed_effects(data, design):
    # Stata: sigma_e of xtreg, re with a time-invariant regressor is the sigma_e
    # of xtreg, fe, which omits the regressor
    columns = DESIGNS[design]
    re_res = RandomEffects(data["y"], data[columns]).fit()
    fe_columns = [c for c in columns if c not in ("const", "z")]
    fe_res = PanelOLS(data["y"], data[fe_columns], entity_effects=True).fit()
    assert_allclose(re_res.variance_decomposition["Residual"], fe_res.s2)


@pytest.mark.parametrize("scale", [1e3, 1e6, 1e8])
def test_variance_does_not_depend_on_the_units_of_a_regressor_that_does_not_vary(
    data, scale
):
    # Removing the means of a regressor that does not vary within entities leaves
    # rounding error, which is large next to the other regressors if its units are
    # large, and it is not a coefficient of the within regression
    columns = DESIGNS["time-invariant"]
    scaled = data.assign(z=data["z"] * np.pi * scale)
    res = RandomEffects(scaled["y"], scaled[columns]).fit()
    expected = lsdv_variance(scaled, columns, None)
    assert_allclose(res.variance_decomposition["Residual"], expected)


@pytest.mark.parametrize("small_sample", [False, True])
@pytest.mark.parametrize("design", list(DESIGNS))
def test_variance_components_against_estimators_written_out(data, design, small_sample):
    columns = DESIGNS[design]
    res = RandomEffects(data["y"], data[columns]).fit(small_sample=small_sample)
    sigma2_eps, sigma2_effects, *_ = swar_reference(
        data["y"], data[columns], small_sample=small_sample
    )
    decomposition = res.variance_decomposition
    assert_allclose(decomposition["Residual"], sigma2_eps)
    assert_allclose(decomposition["Effects"], sigma2_effects)
    rho = sigma2_effects / (sigma2_effects + sigma2_eps)
    assert_allclose(decomposition["Percent due to Effects"], rho)
    theta = res.theta["theta"]
    count = data.groupby(level=0).size().reindex(theta.index)
    expected_theta = 1 - np.sqrt(sigma2_eps / (count * sigma2_effects + sigma2_eps))
    assert_allclose(theta, expected_theta)


@pytest.mark.parametrize("design", list(DESIGNS))
def test_weighted_variance_components_against_estimators_written_out(data, design):
    columns = DESIGNS[design]
    res = RandomEffects(data["y"], data[columns], weights=data["w"]).fit()
    sigma2_eps, sigma2_effects, *_ = swar_reference(
        data["y"], data[columns], weights=data["w"]
    )
    assert_allclose(res.variance_decomposition["Residual"], sigma2_eps)
    assert_allclose(res.variance_decomposition["Effects"], sigma2_effects)


def test_the_designs_have_regressors_that_the_regressions_do_not_estimate():
    # The tests are only informative if the numbers of coefficients of the
    # regressions differ from the number of regressors. The within regression
    # has x1 and the dummies: z and the constant do not vary within entities, and
    # exper is collinear with the dummies. In the means of a balanced panel the
    # dummies are the constant 1/T. If periods 0, 1 and 2 are observed for every
    # entity, the means of their dummies are equal (1/T_i), and so the constant is
    # 3 times the mean of the dummy of period 1 plus the means of the other dummies
    columns = DESIGNS["collinear-with-dummies"]
    for balance, between in [
        ("balanced", 4),
        ("unbalanced", len(columns)),
        ("unbalanced-collinear-means", len(columns) - 2),
    ]:
        sample = make_data(balance)
        ranks = swar_reference(sample["y"], sample[columns])[2:]
        assert ranks == (1 + len(DUMMIES), between), balance


def wage_data(unbalanced):
    """The wage panel of the notebook, and its unbalanced version in the R script"""
    data = wage_panel.load()
    if unbalanced:
        year, nr = data["year"], data["nr"]
        drop = ((nr * 7 + year * 13) % 5 == 0) & ((nr + year) % 3 != 0)
        data = data.loc[~drop]
    year = pd.Categorical(data["year"])
    data = data.set_index(["nr", "year"])
    data["year"] = year
    data["const"] = 1.0
    return data


TIME_INVARIANT = ["black", "hisp", "exper", "expersq", "married", "educ", "union"]
PLM_COLUMNS = {
    "notebook": ["const", *TIME_INVARIANT, "year"],
    "year-dummies": [
        "const",
        "black",
        "hisp",
        "expersq",
        "married",
        "educ",
        "union",
        "year",
    ],
    "time-invariant": ["const", *TIME_INVARIANT],
    "time-varying": ["const", "expersq", "married", "union"],
}
PLM_CASES = [
    ("balanced", "notebook"),
    ("balanced", "time-invariant"),
    ("balanced", "time-varying"),
    ("unbalanced", "year-dummies"),
    ("unbalanced", "time-invariant"),
]


@pytest.fixture(scope="module")
def plm_reference():
    path = os.path.join(RESULTS, "re-variance-components-reference.csv")
    reference = pd.read_csv(path)
    return {
        case: group.set_index("quantity")["value"]
        for case, group in reference.groupby("case")
    }


@pytest.mark.parametrize(("balance", "model"), PLM_CASES)
def test_random_effects_against_plm(plm_reference, balance, model):
    expected = plm_reference[f"{balance}-{model}"]
    unbalanced = balance == "unbalanced"
    data = wage_data(unbalanced)
    mod = RandomEffects(data["lwage"], data[PLM_COLUMNS[model]])
    # plm uses the estimator of Baltagi and Chang (1994) for an unbalanced panel
    res = mod.fit(small_sample=unbalanced)
    assert res.nobs == expected["nobs"]
    decomposition = res.variance_decomposition
    assert_allclose(decomposition["Residual"], expected["sigma2_idios"], rtol=1e-8)
    assert_allclose(decomposition["Effects"], expected["sigma2_id"], rtol=1e-8)
    # plm reports the theta of each observation
    entities = mod.dependent.index.get_level_values(0)
    theta = res.theta["theta"].reindex(entities)
    assert_allclose(theta.mean(), expected["theta_mean"], rtol=1e-8)
    nparam = res.params.shape[0]
    coefficients = expected[[f"coef{i}" for i in range(nparam)]].to_numpy()
    std_errors = expected[[f"se{i}" for i in range(nparam)]].to_numpy()
    assert_allclose(res.params.to_numpy(), coefficients, rtol=1e-7, atol=1e-10)
    assert_allclose(res.std_errors.to_numpy(), std_errors, rtol=1e-7)
    # The estimators written out use the numbers of coefficients that plm does
    sigma2_eps, sigma2_effects, rank_within, rank_between = swar_reference(
        mod.dependent.dataframe["lwage"],
        mod.exog.dataframe,
        small_sample=unbalanced,
    )
    nentity = entities.nunique()
    assert expected["df_within"] == res.nobs - nentity - rank_within
    assert expected["df_between"] == nentity - rank_between
    assert_allclose(sigma2_eps, expected["sigma2_idios"], rtol=1e-8)
    assert_allclose(sigma2_effects, expected["sigma2_id"], rtol=1e-8)
