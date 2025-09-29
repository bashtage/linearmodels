from itertools import product
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import AmbiguityError, PanelOLS
from linearmodels.panel.utility import AbsorbingEffectError
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.tests.panel._utility import datatypes, generate_data, lsdv

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = datatypes


@pytest.fixture(
    params=list(product(PERC_MISSING, TYPES)),
    ids=list(
        map(
            lambda x: str(int(100 * x[0])) + "-" + str(x[1]),
            product(PERC_MISSING, TYPES),
        )
    ),
)
def data(request):
    missing, datatype = request.param
    rng = np.random.RandomState(12345)
    return generate_data(missing, datatype, ntk=(131, 4, 3), rng=rng)


def test_panel_ols(data):
    PanelOLS(data.y, data.x).fit()
    PanelOLS(data.y, data.x, entity_effects=True).fit()
    PanelOLS(data.y, data.x, time_effects=True).fit()


def test_valid_weight_shape(data):
    # Same size
    n = np.prod(data.y.shape)
    weights = 1 + np.random.random_sample(n)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    missing = PanelData(data.y).isnull | PanelData(data.x).isnull
    expected = weights[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert w == pytest.approx(expected)

    # Per time
    if isinstance(data.x, pd.DataFrame):
        n = len(data.y.index.levels[1])
        k = len(data.y.index.levels[0])
    elif isinstance(data.x, np.ndarray):
        n = data.y.shape[0]
        k = data.y.shape[1]
    else:
        n = data.y.shape[1]
        k = data.y.shape[2]

    weights = 1 + np.random.random_sample(n)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    expected = weights[:, None] @ np.ones((1, k))
    expected = expected.T.ravel()
    expected = expected[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert w == pytest.approx(expected)

    # Per entity
    if isinstance(data.x, pd.DataFrame):
        n = len(data.y.index.levels[0])
        k = len(data.y.index.levels[1])
    elif isinstance(data.x, np.ndarray):
        n = data.y.shape[1]
        k = data.y.shape[0]
    else:
        n = data.y.shape[2]
        k = data.y.shape[1]
    weights = 1 + np.random.random_sample(n)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    expected = np.ones((k, 1)) @ weights[None, :]
    expected = expected.T.ravel()
    expected = expected[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert w == pytest.approx(expected)

    weights = 1 + np.random.random_sample(data.y.shape)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    expected = weights.T.ravel()
    expected = expected[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert w == pytest.approx(expected)


def test_weight_incorrect_shape(data):
    weights = np.ones(np.prod(data.y.shape) - 1)
    with pytest.raises(ValueError, match="Weights do not have a"):
        PanelOLS(data.y, data.x, weights=weights)

    weights = np.ones((data.y.shape[0], data.y.shape[1] - 1))
    with pytest.raises(ValueError, match="Weights do not have a supported shape"):
        PanelOLS(data.y, data.x, weights=weights)


def test_invalid_weight_values(data):
    w = PanelData(data.w)
    w.dataframe.iloc[::13, :] = 0.0
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        PanelOLS(data.y, data.x, weights=w)

    w = PanelData(data.w)
    w.dataframe.iloc[::13, :] = -0.0
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        PanelOLS(data.y, data.x, weights=w)

    w = PanelData(data.w)
    w.dataframe.iloc[::29, :] = -1.0
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        PanelOLS(data.y, data.x, weights=w)


def test_panel_lsdv(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    y, x = mod.dependent.dataframe, mod.exog.dataframe
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, entity=True)
    assert_allclose(res.params.squeeze(), expected)

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, time=True)
    assert_allclose(res.params.squeeze(), expected)

    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, entity=True, time=True)
    assert_allclose(res.params.squeeze(), expected, rtol=1e-4)

    other = y.copy()
    other.iloc[:, :] = 0
    other = other.astype(np.int64)
    skip = other.shape[0] // 3
    for i in range(skip):
        other.iloc[i::skip] = i

    mod = PanelOLS(y, x, other_effects=other)
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, general=other.iloc[:, 0].values)
    assert_allclose(res.params.squeeze(), expected, rtol=1e-4)


def test_incorrect_weight_shape(data):
    w = data.w
    if isinstance(w, pd.DataFrame):
        entities = w.index.levels[0][:4]
        w = w.loc[pd.IndexSlice[entities[0] : entities[-1]], :]
    elif isinstance(w, np.ndarray):
        w = w[:3]
        w = w[None, :, :]
    else:  # xarray
        return

    with pytest.raises(ValueError, match="Weights do not have "):
        PanelOLS(data.y, data.x, weights=w)


def test_weight_ambiguity(data):
    if isinstance(data.x, pd.DataFrame):
        t = len(data.y.index.levels[1])
        entities = data.x.index.levels[0]
        index_slice = pd.IndexSlice[entities[0] : entities[t - 1]]
        x = data.x.loc[index_slice, :]
    else:
        t = data.x.shape[1]
        x = data.x[:, :, :t]
    y = data.y
    weights = 1 + np.random.random_sample(t)
    with pytest.raises(AmbiguityError):
        PanelOLS(y, x, weights=weights)


@pytest.mark.parametrize("intercept", [True, False])
def test_absorbing_effect(data, intercept):
    x = data.x.copy()
    if isinstance(data.x, pd.DataFrame):
        nentity = len(x.index.levels[0])
        ntime = len(x.index.levels[1])
        temp = data.x.iloc[:, 0].copy()
        temp.loc[:] = 1.0
        temp.iloc[: (ntime * (nentity // 2))] = 0

        if intercept:
            x["Intercept"] = 1.0
        x["absorbed"] = temp
    else:
        intercept_vals = np.ones((1, x.shape[1], x.shape[2]))
        absorbed = np.ones((1, x.shape[1], x.shape[2]))
        absorbed[:, :, : x.shape[2] // 2] = 0
        if intercept:
            extra = [x, intercept_vals, absorbed]
        else:
            extra = [x, absorbed]
        x = np.concatenate(extra, 0)

    mod = PanelOLS(data.y, x, entity_effects=True)
    with pytest.raises(AbsorbingEffectError) as exc_info:
        mod.fit()
    var_names = mod.exog.vars

    assert var_names[3] in str(exc_info.value)
    assert (" " * (2 - intercept) + var_names[-1]) in str(exc_info.value)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_all_missing(data):
    y = PanelData(data.y)
    x = PanelData(data.x)
    missing = y.isnull | x.isnull
    y.drop(missing)
    x.drop(missing)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        PanelOLS(y.dataframe, x.dataframe).fit()
    assert len(w) == 0


def test_pickle(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    remod = pickle.loads(pickle.dumps(mod))
    res = mod.fit()
    reres = remod.fit()
    rereres = pickle.loads(pickle.dumps(res))
    assert_allclose(res.params, reres.params)
    assert_allclose(res.params, rereres.params)
    assert_allclose(res.cov, reres.cov)
    assert_allclose(res.cov, rereres.cov)
    assert isinstance(res.f_statistic_robust, WaldTestStatistic)
    assert isinstance(reres.f_statistic_robust, WaldTestStatistic)
    assert isinstance(res.f_statistic_robust, WaldTestStatistic)
