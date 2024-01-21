from itertools import product
import warnings

import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import FamaMacBeth
from linearmodels.shared.exceptions import (
    InferenceUnavailableWarning,
    MissingValueWarning,
)
from linearmodels.tests.panel._utility import (
    access_attributes,
    assert_frame_similar,
    datatypes,
    generate_data,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)

missing = [0.0, 0.20]
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perms) for perm in perms]


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing, datatype, const=const, other_effects=1, ntk=(25, 200, 5)
    )


def test_fama_macbeth(data):
    res = FamaMacBeth(data.y, data.x).fit(debiased=True)
    y = PanelData(data.y)
    x = PanelData(data.x)
    missing = y.isnull | x.isnull
    y.drop(missing)
    x.drop(missing)
    y = y.dataframe
    x = x.dataframe
    times = y.index.levels[1]
    params = []
    for t in times:
        _y = y.xs(t, level=1)
        _x = x.xs(t, level=1)
        if _x.shape[0] < _x.shape[1]:
            continue
        _x = _x.loc[_y.index]
        params.append(lstsq(_x.values, _y.values, rcond=None)[0])
    params = np.array(params).squeeze()
    all_params = params
    params = params.mean(0)
    assert_allclose(params.squeeze(), res.params)
    assert_allclose(all_params, res.all_params.dropna(how="all"))
    e_params = all_params - params[None, :]
    ntime = e_params.shape[0]
    cov = e_params.T @ e_params / ntime / (ntime - 1)
    assert_allclose(cov, np.asarray(res.cov))
    access_attributes(res)


def test_unknown_cov_type(data):
    with pytest.raises(ValueError):
        FamaMacBeth(data.y, data.x).fit(cov_type="unknown")


@pytest.mark.smoke
def test_fama_macbeth_kernel_smoke(data):
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="bartlett")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="newey-west")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="parzen")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="qs")
    FamaMacBeth(data.y, data.x).fit(cov_type="kernel", bandwidth=3)
    res = FamaMacBeth(data.y, data.x).fit(cov_type="kernel", kernel="andrews")
    access_attributes(res)


def test_fitted_effects_residuals(data):
    mod = FamaMacBeth(data.y, data.x)
    res = mod.fit()

    expected = mod.exog.values2d @ res.params.values
    expected = pd.DataFrame(expected, index=mod.exog.index, columns=["fitted_values"])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    idiosyncratic = pd.DataFrame(
        mod.dependent.values2d - expected.values, index=expected.index
    )
    expected.iloc[:, 0] = idiosyncratic.iloc[:, 0]
    expected.columns = ["idiosyncratic"]
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    expected.iloc[:, 0] = np.nan
    expected.columns = ["estimated_effects"]
    assert_allclose(res.estimated_effects, expected)
    assert_frame_similar(res.estimated_effects, expected)


@pytest.mark.filterwarnings(
    "always::linearmodels.shared.exceptions.MissingValueWarning"
)
def test_block_size_warnings():
    y = np.arange(12.0)[:, None]
    x = np.ones((12, 3))
    x[:, 1] = np.arange(12.0)
    x[:, 2] = np.arange(12.0) ** 2
    idx = pd.MultiIndex.from_product(
        [["a", "b", "c"], pd.date_range("2000-1-1", periods=4)]
    )
    y = pd.DataFrame(y, index=idx, columns=["y"])
    x = pd.DataFrame(x, index=idx, columns=["x1", "x2", "x3"])
    with pytest.warns(MissingValueWarning):
        FamaMacBeth(y.iloc[:11], x.iloc[:11])
    with pytest.warns(InferenceUnavailableWarning):
        FamaMacBeth(y.iloc[::4], x.iloc[::4])


def test_block_size_error():
    y = np.arange(12.0)[:, None]
    x = np.ones((12, 2))
    x[1::4, 1] = 2
    x[2::4, 1] = 3
    idx = pd.MultiIndex.from_product(
        [["a", "b", "c"], pd.date_range("2000-1-1", periods=4)]
    )
    y = pd.DataFrame(y, index=idx, columns=["y"])
    x = pd.DataFrame(x, index=idx, columns=["x1", "x2"])
    with pytest.raises(ValueError):
        FamaMacBeth(y, x)


def test_limited_redundancy():
    data = generate_data(
        0, datatype="numpy", const=False, other_effects=1, ntk=(25, 200, 5)
    )
    for i in range(0, data.x.shape[1], 7):
        data.x[1, i, :] = data.x[0, i, :]
    mod = FamaMacBeth(data.y, data.x)
    res = mod.fit()
    assert np.any(np.isnan(res.all_params))


@pytest.mark.parametrize("just_id", [True, False])
@pytest.mark.parametrize("use_const", [True, False])
def test_avg_r2(use_const, just_id):
    nvar = 10 if not just_id else 3 + int(use_const)
    mi = pd.MultiIndex.from_product([np.arange(nvar), np.arange(10)])
    rg = np.random.default_rng(0)
    n = len(mi)
    e = rg.standard_normal((n, 3))
    cols = ["x1", "x2", "x3"]
    x = pd.DataFrame(e, index=mi, columns=cols)
    if use_const:
        const = pd.Series(np.ones(n), index=mi, name="const")
        x = pd.concat([const, x], axis=1)
    y = pd.Series(rg.standard_normal(n), index=mi, name="y")
    res = FamaMacBeth(y, x).fit()
    z = pd.concat([y, x], axis=1)
    r2 = []
    r2_adj = []
    for _, g in z.groupby(level=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_res = OLS(g["y"], g[x.columns]).fit()
            r2.append(ols_res.rsquared)
            r2_adj.append(ols_res.rsquared_adj)
    assert_allclose(np.array(r2).mean(), res.avg_rsquared)
    assert_allclose(np.array(r2_adj).mean(), res.avg_adj_rsquared)
