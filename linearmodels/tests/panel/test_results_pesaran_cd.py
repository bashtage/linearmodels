"""
Tests of PanelResults.pesaran_cd: which models the test is applicable to, and
the residuals that FirstDifferenceOLS uses.

The reference values of FirstDifferenceOLS in results/pesaran-cd-fd-reference.csv
were produced by results/pesaran-cd-fd-reference.R, which computes the statistic
from the residuals of the differenced equation using the R package plm.
"""

from itertools import combinations
import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats
from statsmodels.tools.tools import add_constant

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    NormalTestStatistic,
)

RESULTS = os.path.join(os.path.split(os.path.abspath(__file__))[0], "results")


def cd_panel(seed=0, nentity=30, nperiod=8, loading=0.0):
    """Panel with independent errors, or a common factor if loading > 0"""
    rg = np.random.default_rng(seed)
    factor = rg.standard_normal(nperiod)
    load = 1.0 + 0.5 * rg.standard_normal(nentity)
    x = rg.standard_normal((nentity, nperiod))
    effects = rg.standard_normal((nentity, 1))
    shock = loading * load[:, None] * factor + rg.standard_normal((nentity, nperiod))
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    y = 1.0 + 0.5 * x + effects + shock
    return pd.DataFrame({"y": y.ravel(), "x": x.ravel()}, index=index)


def brute_force_cd(idiosyncratic):
    """Pesaran CD using explicit loops over entity pairs"""
    resid = idiosyncratic.iloc[:, 0]
    by_entity = {key: grp.droplevel(0) for key, grp in resid.groupby(level=0)}
    total = 0.0
    npairs = 0
    for first, second in combinations(by_entity, 2):
        common = by_entity[first].index.intersection(by_entity[second].index)
        if len(common) < 2:
            continue
        rho = np.corrcoef(by_entity[first][common], by_entity[second][common])[0, 1]
        if np.isfinite(rho):
            total += np.sqrt(len(common)) * rho
            npairs += 1
    return total / np.sqrt(npairs)


def cd_models(frame):
    """Functions that create the models and if the Pesaran CD test applies"""
    y, x = frame.y, frame[["x"]]
    xc = add_constant(x)
    entity = pd.Series(frame.index.get_level_values(0), index=frame.index)
    period = pd.Series(frame.index.get_level_values(1), index=frame.index)
    nperiod = int(period.max()) + 1
    nentity = int(entity.max()) + 1
    rg = np.random.default_rng(1234)
    random_cats = pd.Series(rg.integers(0, 4, size=frame.shape[0]), index=frame.index)
    weights = pd.Series(1.0 + (np.arange(frame.shape[0]) % 4), index=frame.index)
    group_period = (entity % 3) * nperiod + period
    group_block = (entity % 3) * nperiod + period // 2
    # Groups with a single entity are not shared with other entities
    entity_regime = entity * 2 + (period >= nperiod // 2).astype(int)
    # Groups that change over time but are present in every period
    states = np.empty((nentity, nperiod), dtype=int)
    states[:, 0] = rg.integers(0, 3, size=nentity)
    for i in range(1, nperiod):
        stay = rg.random(nentity) < 0.6
        states[:, i] = np.where(stay, states[:, i - 1], rg.integers(0, 3, size=nentity))
    markov = pd.Series(states.ravel(), index=frame.index)
    # A treatment indicator that is switched on at a different date for each entity
    switch = pd.Series(rg.integers(1, nperiod, size=nentity))
    post = (period >= switch.loc[entity].to_numpy()).astype(int)
    return {
        "pooled": (lambda: PooledOLS(y, xc), True),
        "entity": (lambda: PanelOLS(y, x, entity_effects=True), True),
        "entity-weighted": (
            lambda: PanelOLS(y, x, entity_effects=True, weights=weights),
            True,
        ),
        "random": (lambda: RandomEffects(y, xc), True),
        "first-difference": (lambda: FirstDifferenceOLS(y, x), True),
        "first-difference-weighted": (
            lambda: FirstDifferenceOLS(y, x, weights=weights),
            True,
        ),
        # Time-invariant groups of entities, and groups unrelated to the periods
        "other-group": (lambda: PanelOLS(y, xc, other_effects=entity % 3), True),
        "entity-other-random": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=random_cats),
            True,
        ),
        "entity-other-markov": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=markov),
            True,
        ),
        "other-entity-regime": (
            lambda: PanelOLS(y, x, other_effects=entity_regime),
            True,
        ),
        "between": (lambda: BetweenOLS(y, xc), False),
        "fama-macbeth": (lambda: FamaMacBeth(y, xc), False),
        "time": (lambda: PanelOLS(y, x, time_effects=True), False),
        "entity-time": (
            lambda: PanelOLS(y, x, entity_effects=True, time_effects=True),
            False,
        ),
        # Time effects specified as other effects
        "other-period": (lambda: PanelOLS(y, xc, other_effects=period), False),
        "entity-other-period": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=period),
            False,
        ),
        "entity-other-group-period": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=group_period),
            False,
        ),
        "entity-other-blocks": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=period // 2),
            False,
        ),
        # Groups of entities by blocks of periods, which are not in every period
        "entity-other-group-block": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=group_block),
            False,
        ),
        "entity-other-post": (
            lambda: PanelOLS(y, x, entity_effects=True, other_effects=post),
            False,
        ),
        "other-random-period": (
            lambda: PanelOLS(
                y,
                xc,
                other_effects=pd.DataFrame({"random": random_cats, "period": period}),
            ),
            False,
        ),
    }


CD_MODELS = list(cd_models(cd_panel()))


@pytest.mark.parametrize("name", CD_MODELS)
def test_pesaran_cd_applicability(name):
    create, applicable = cd_models(cd_panel())[name]
    res = create().fit()
    cd = res.pesaran_cd
    if applicable:
        assert isinstance(cd, NormalTestStatistic)
        if name.startswith("first-difference"):
            # The residuals of the differenced equation, not the shocks in
            # levels, and the weights are not applied to them
            shocks = res.resids.to_frame()
        else:
            shocks = res.idiosyncratic
        expected = brute_force_cd(shocks)
        assert_allclose(cd.stat, expected, rtol=1e-10)
        assert_allclose(cd.pval, 2 * stats.norm.sf(abs(expected)), rtol=1e-10)
        return
    assert isinstance(cd, InapplicableTestStatistic)
    assert np.isnan(cd.stat)
    assert np.isnan(cd.pval)
    assert cd.critical_values is None
    assert "Pesaran CD test is not applicable" in str(cd)
    assert "Pesaran CD test" in str(cd)


@pytest.mark.parametrize(
    ("name", "phrase"),
    [
        ("between", "BetweenOLS"),
        ("between", "entity means"),
        ("fama-macbeth", "FamaMacBeth"),
        ("fama-macbeth", "(beta_t - beta_bar)'x_it"),
        ("fama-macbeth", "robust to cross-sectional dependence"),
        ("time", "time effects"),
        ("entity-time", "time effects"),
        ("other-period", "concentrated in some periods"),
        ("entity-other-period", "concentrated in some periods"),
        ("entity-other-group-period", "concentrated in some periods"),
        ("entity-other-blocks", "concentrated in some periods"),
        ("entity-other-group-block", "concentrated in some periods"),
        ("entity-other-post", "concentrated in some periods"),
        ("other-random-period", "concentrated in some periods"),
        ("time", "demean the residuals across entities"),
    ],
)
def test_pesaran_cd_inapplicable_reason(name, phrase):
    create, applicable = cd_models(cd_panel())[name]
    assert not applicable
    cd = create().fit().pesaran_cd
    assert isinstance(cd, InapplicableTestStatistic)
    assert phrase in str(cd)


def test_pesaran_cd_time_effects_mechanism():
    # Time effects remove the cross-sectional mean of the residuals in every
    # period, which is why the correlation of independent errors is negative
    frame = cd_panel(seed=7, nentity=40, nperiod=6)
    y, x = frame.y, frame[["x"]]
    two_way = PanelOLS(y, x, entity_effects=True, time_effects=True).fit()
    period_means = two_way.idiosyncratic.iloc[:, 0].groupby(level=1).mean()
    assert_allclose(period_means, 0.0, atol=1e-12)
    entity = PanelOLS(y, x, entity_effects=True).fit()
    period_means = entity.idiosyncratic.iloc[:, 0].groupby(level=1).mean()
    assert np.abs(period_means).max() > 1e-3
    # The correlation of the demeaned residuals is about -1/(N-1) for every pair
    wide = two_way.idiosyncratic.iloc[:, 0].unstack(level=0)
    corr = wide.corr().to_numpy()
    avg = corr[np.tril_indices(corr.shape[0], k=-1)].mean()
    assert avg < -0.5 / (frame.index.get_level_values(0).nunique() - 1)


def test_pesaran_cd_still_detects_dependence():
    # The applicable models still reject in the presence of a common factor
    frame = cd_panel(seed=11, nentity=60, nperiod=10, loading=1.0)
    y, x = frame.y, frame[["x"]]
    res = PanelOLS(y, x, entity_effects=True).fit()
    assert isinstance(res.pesaran_cd, NormalTestStatistic)
    assert res.pesaran_cd.pval < 0.01


def test_pesaran_cd_time_invariant_other_effects_unbalanced():
    # A group that is only observed in some periods does not remove a component
    # that varies over time when the group of every entity is the same in
    # every period
    frame = cd_panel(seed=5, nentity=30, nperiod=8)
    entity = frame.index.get_level_values(0)
    period = frame.index.get_level_values(1)
    # Entities 20 to 29 are only observed in the first 4 periods
    frame = frame.loc[~((entity >= 20) & (period >= 4))]
    region = pd.Series(frame.index.get_level_values(0) // 10, index=frame.index)
    assert region.groupby(level=1).nunique().min() < 3
    res = PanelOLS(frame.y, add_constant(frame[["x"]]), other_effects=region).fit()
    cd = res.pesaran_cd
    assert isinstance(cd, NormalTestStatistic)
    assert_allclose(cd.stat, brute_force_cd(res.idiosyncratic), rtol=1e-10)


def test_pesaran_cd_first_difference_closed_form():
    # With N entities that are observed in T periods there are T - 1 residuals
    # of the differenced equation for each, so that
    # CD = sqrt(2 (T - 1) / (N (N - 1))) * the sum of the N (N - 1) / 2 correlations
    nentity, nperiod = 40, 8
    frame = cd_panel(seed=3, nentity=nentity, nperiod=nperiod)
    y = frame.y.to_numpy().reshape(nentity, nperiod)
    x = frame.x.to_numpy().reshape(nentity, nperiod)
    dy, dx = np.diff(y, axis=1), np.diff(x, axis=1)
    beta = np.sum(dx * dy) / np.sum(dx * dx)
    corr = np.corrcoef(dy - beta * dx)
    pairs = corr[np.triu_indices(nentity, k=1)]
    expected = np.sqrt(2 * (nperiod - 1) / (nentity * (nentity - 1))) * pairs.sum()

    res = FirstDifferenceOLS(frame.y, frame[["x"]]).fit()
    assert_allclose(res.params.iloc[0], beta, rtol=1e-10)
    cd = res.pesaran_cd
    assert isinstance(cd, NormalTestStatistic)
    assert_allclose(cd.stat, expected, rtol=1e-10)
    assert_allclose(cd.pval, 2 * stats.norm.sf(abs(expected)), rtol=1e-10)
    # It is not the statistic of the shocks in levels
    assert not np.isclose(cd.stat, brute_force_cd(res.idiosyncratic), rtol=1e-3)


def test_pesaran_cd_first_difference_two_periods():
    # There is one difference for each entity, so that no pair of entities has
    # two overlapping residuals. The shocks in levels have two for every pair
    frame = cd_panel(seed=2, nentity=10, nperiod=2)
    res = FirstDifferenceOLS(frame.y, frame[["x"]]).fit()
    assert res.resids.shape[0] == 10
    cd = res.pesaran_cd
    assert isinstance(cd, InapplicableTestStatistic)
    assert "two or more overlapping observations" in str(cd)


FD_PANELS = pd.read_csv(os.path.join(RESULTS, "pesaran-cd-fd-panels.csv"))
FD_REFERENCE = pd.read_csv(os.path.join(RESULTS, "pesaran-cd-fd-reference.csv"))


@pytest.mark.parametrize(
    "row",
    [pytest.param(row, id=row.case) for row in FD_REFERENCE.itertuples(index=False)],
)
def test_pesaran_cd_first_difference_plm(row):
    panel = FD_PANELS.loc[FD_PANELS.case == row.case].set_index(["entity", "time"])
    res = FirstDifferenceOLS(panel.y, panel[["x"]]).fit()
    # The reference uses the same differences and the same estimate
    assert res.nobs == row.nobs
    assert_allclose(res.params.iloc[0], row.coef, rtol=1e-8)
    # Pairs of entities with at least two residuals at the same times
    seen = res.resids.unstack(level=0).notnull().to_numpy(dtype=int)
    assert np.tril(seen.T @ seen >= 2, k=-1).sum() == row.npairs

    cd = res.pesaran_cd
    assert isinstance(cd, NormalTestStatistic)
    assert_allclose(cd.stat, row.cd_stat, rtol=1e-8)
    assert_allclose(cd.pval, row.cd_pvalue, rtol=1e-8)
    # The shocks in levels give the reference of the levels, which is not the same
    assert_allclose(brute_force_cd(res.idiosyncratic), row.cd_levels, rtol=1e-8)
    assert not np.isclose(cd.stat, row.cd_levels, rtol=1e-2)
