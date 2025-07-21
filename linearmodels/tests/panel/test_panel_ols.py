from itertools import product

import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.datasets import wage_panel
from linearmodels.iv.model import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS, PooledOLS
from linearmodels.panel.utility import AbsorbingEffectWarning
from linearmodels.shared.exceptions import MemoryWarning
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.shared.utility import AttrDict
from linearmodels.tests.panel._utility import (
    access_attributes,
    assert_frame_similar,
    assert_results_equal,
    datatypes,
    generate_data,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning",
    "ignore:the matrix subclass:PendingDeprecationWarning",
)

perc_missing = [0.0, 0.02, 0.20]
has_const = [True, False]
perms = list(product(perc_missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perms) for perm in perms]


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing, datatype, const=const, ntk=(91, 15, 5), other_effects=2
    )


@pytest.fixture(params=["numpy", "pandas"])
def absorbed_data(request):
    datatype = request.param
    rng = np.random.RandomState(12345)
    data = generate_data(0, datatype, ntk=(131, 4, 3), rng=rng)
    x = data.x
    if isinstance(data.x, np.ndarray):
        absorbed = np.arange(x.shape[2])
        absorbed = np.tile(absorbed, (1, x.shape[1], 1))
        data.x = np.concatenate([data.x, absorbed])
    elif isinstance(data.x, pd.DataFrame):
        codes = data.x.index.codes
        absorbed = np.array(codes[0]).astype(np.double)
        data.x["x_absorbed"] = absorbed
    return data


@pytest.fixture(params=perms, ids=ids)
def large_data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing, datatype, const=const, ntk=(51, 71, 5), other_effects=2
    )


singleton_ids = [i for i, p in zip(ids, perms) if p[1] == "pandas" and not p[-1]]
singleton_perms = [p for p in perms if p[1] == "pandas" and not p[-1]]


@pytest.fixture(params=singleton_perms, ids=singleton_ids)
def singleton_data(request):
    missing, datatype, const = request.param
    return generate_data(
        missing,
        datatype,
        const=const,
        ntk=(91, 15, 5),
        other_effects=2,
        num_cats=[5 * 91, 15],
    )


const_perms = list(product(perc_missing, datatypes))
const_ids = ["-".join(str(val) for val in perm) for perm in const_perms]


@pytest.fixture(params=const_perms, ids=const_ids)
def const_data(request):
    missing, datatype = request.param
    data = generate_data(missing, datatype, ntk=(91, 7, 1))
    y = PanelData(data.y).dataframe
    x = y.copy()
    x.iloc[:, :] = 1
    x.columns = ["Const"]
    return AttrDict(y=y, x=x, w=PanelData(data.w).dataframe)


@pytest.fixture(params=[True, False])
def entity_eff(request):
    return request.param


@pytest.fixture(params=[True, False])
def time_eff(request):
    return request.param


lsdv_perms = [
    p
    for p in product([True, False], [True, False], [True, False], [0, 1, 2])
    if sum(p[1:]) <= 2
]
lsdv_ids = []
for p in lsdv_perms:
    str_id = "weighted" if p[0] else "unweighted"
    str_id += "-entity_effects" if p[1] else ""
    str_id += "-time_effects" if p[2] else ""
    str_id += f"-{p[3]}_other_effects" if p[3] else ""
    lsdv_ids.append(str_id)


@pytest.fixture(params=lsdv_perms, ids=lsdv_ids)
def lsdv_config(request):
    weights, entity_effects, time_effects, other_effects = request.param
    return AttrDict(
        weights=weights,
        entity_effects=entity_effects,
        time_effects=time_effects,
        other_effects=other_effects,
    )


def test_const_data_only(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x)
    res = mod.fit(debiased=False)
    res2 = IV2SLS(y, x, None, None).fit()
    assert_allclose(res.params, res2.params)


def test_const_data_only_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, weights=const_data.w)
    res = mod.fit(debiased=False)
    res2 = IV2SLS(y, x, None, None, weights=const_data.w).fit()
    assert_allclose(res.params, res2.params)


def test_const_data_entity(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True)
    res = mod.fit(debiased=False)

    x = mod.exog.dataframe
    d = mod.dependent.dummies("entity", drop_first=True)
    d.iloc[:, :] = d.values - x.values @ lstsq(x.values, d.values, rcond=None)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_time(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, time_effects=True)
    res = mod.fit(debiased=False)

    x = mod.exog.dataframe
    d = mod.dependent.dummies("time", drop_first=True)
    d.iloc[:, :] = d.values - x.values @ lstsq(x.values, d.values, rcond=None)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


@pytest.mark.parametrize("entity", [True, False])
def test_const_data_single_effect_weights(const_data, entity):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(
        y, x, entity_effects=entity, time_effects=not entity, weights=const_data.w
    )
    res = mod.fit(debiased=False)

    y = mod.dependent.dataframe
    w = mod.weights.dataframe
    x = mod.exog.dataframe
    dummy_type = "entity" if entity else "time"
    d = mod.dependent.dummies(dummy_type, drop_first=True)
    d_columns = list(d.columns)

    root_w = np.sqrt(w.values)
    z = np.ones_like(x)
    wd = root_w * d.values
    wz = root_w
    d = d - z @ lstsq(wz, wd, rcond=None)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + d_columns)

    res2 = IV2SLS(y, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_both(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    res = mod.fit(debiased=False)

    x = mod.exog.dataframe
    d1 = mod.dependent.dummies("entity", drop_first=True)
    d1.columns = [f"d.entity.{i}" for i in d1]
    d2 = mod.dependent.dummies("time", drop_first=True)
    d2.columns = [f"d.time.{i}" for i in d2]
    d = np.c_[d1.values, d2.values]
    d = pd.DataFrame(d, index=x.index, columns=list(d1.columns) + list(d2.columns))
    d.iloc[:, :] = d.values - x.values @ lstsq(x.values, d.values, rcond=None)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_both_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True, weights=const_data.w)
    res = mod.fit(debiased=False)

    w = mod.weights.dataframe
    x = mod.exog.dataframe

    d1 = mod.dependent.dummies("entity", drop_first=True)
    d1.columns = [f"d.entity.{i}" for i in d1]
    d2 = mod.dependent.dummies("time", drop_first=True)
    d2.columns = [f"d.time.{i}" for i in d2]
    d = np.c_[d1.values, d2.values]
    root_w = np.sqrt(w.values)
    z = np.ones_like(x)
    wd = root_w * d
    wz = root_w
    d = d - z @ lstsq(wz, wd, rcond=None)[0]
    d = pd.DataFrame(d, index=x.index, columns=list(d1.columns) + list(d2.columns))

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_panel_no_effects(data):
    panel = PanelOLS(data.y, data.x)
    assert panel._collect_effects().shape[1] == 0
    res = panel.fit()
    res2 = PooledOLS(data.y, data.x).fit()
    assert_results_equal(res, res2)


def test_panel_no_effects_weighted(data):
    res = PanelOLS(data.y, data.x, weights=data.w).fit()
    res2 = PooledOLS(data.y, data.x, weights=data.w).fit()
    assert_results_equal(res, res2)


def test_panel_entity_lsdv(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    if mod.has_constant:
        d = mod.dependent.dummies("entity", drop_first=True)
        z = np.ones_like(y)
        d_demean = d.values - z @ lstsq(z, d.values, rcond=None)[0]
    else:
        d = mod.dependent.dummies("entity", drop_first=False)
        d_demean = d.values

    xd = np.c_[x.values, d_demean]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted", debiased=False)
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(data.vc1)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(data.vc2)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_entity_fwl(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    if mod.has_constant:
        d = mod.dependent.dummies("entity", drop_first=True)
        z = np.ones_like(y)
        d_demean = d.values - z @ lstsq(z, d.values, rcond=None)[0]
    else:
        d = mod.dependent.dummies("entity", drop_first=False)
        d_demean = d.values

    x = x - d_demean @ lstsq(d_demean, x, rcond=None)[0]
    y = y - d_demean @ lstsq(d_demean, y, rcond=None)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_df=False)


def test_panel_time_lsdv(large_data):
    mod = PanelOLS(large_data.y, large_data.x, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies("time", drop_first=mod.has_constant)
    d_cols = list(d.columns)
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ lstsq(z, d, rcond=None)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + d_cols)

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = large_data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = large_data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_time_fwl(data):
    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies("time", drop_first=mod.has_constant)
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ lstsq(z, d, rcond=None)[0]

    x = x - d @ lstsq(d, x, rcond=None)[0]
    y = y - d @ lstsq(d, y, rcond=None)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_df=False)


def test_panel_both_lsdv(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d1 = mod.dependent.dummies("entity", drop_first=mod.has_constant)
    d2 = mod.dependent.dummies("time", drop_first=True)
    d = np.c_[d1.values, d2.values]

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ lstsq(z, d, rcond=None)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(
        xd, index=x.index, columns=list(x.columns) + list(d1.columns) + list(d2.columns)
    )

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_both_fwl(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d1 = mod.dependent.dummies("entity", drop_first=mod.has_constant)
    d2 = mod.dependent.dummies("time", drop_first=True)
    d = np.c_[d1.values, d2.values]

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ lstsq(z, d, rcond=None)[0]

    x = x - d @ lstsq(d, x, rcond=None)[0]
    y = y - d @ lstsq(d, y, rcond=None)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_df=False)


def test_panel_entity_lsdv_weighted(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, weights=data.w)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    d = mod.dependent.dummies("entity", drop_first=mod.has_constant)
    d_cols = d.columns
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        root_w = np.sqrt(w.values)
        wd = root_w * d
        wz = root_w * z
        d = d - z @ lstsq(wz, wd, rcond=None)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d_cols))

    ols_mod = IV2SLS(y, xd, None, None, weights=w)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_time_lsdv_weighted(large_data):
    mod = PanelOLS(large_data.y, large_data.x, time_effects=True, weights=large_data.w)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    d = mod.dependent.dummies("time", drop_first=mod.has_constant)
    d_cols = d.columns
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        root_w = np.sqrt(w.values)
        wd = root_w * d
        wz = root_w * z
        d = d - z @ lstsq(wz, wd, rcond=None)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d_cols))

    ols_mod = IV2SLS(y, xd, None, None, weights=w)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = large_data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = large_data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_both_lsdv_weighted(data):
    mod = PanelOLS(
        data.y, data.x, entity_effects=True, time_effects=True, weights=data.w
    )
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    d1 = mod.dependent.dummies("entity", drop_first=mod.has_constant)
    d2 = mod.dependent.dummies("time", drop_first=True)
    d = np.c_[d1.values, d2.values]

    if mod.has_constant:
        z = np.ones_like(y)
        root_w = np.sqrt(w.values)
        wd = root_w * d
        wz = root_w * z
        d = d - z @ lstsq(wz, wd, rcond=None)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(
        xd, index=x.index, columns=list(x.columns) + list(d1.columns) + list(d2.columns)
    )

    ols_mod = IV2SLS(y, xd, None, None, weights=w)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_entity_other_equivalence(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit()
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    cats = pd.DataFrame(mod.dependent.entity_ids, index=mod.dependent.index)

    mod2 = PanelOLS(y, x, other_effects=cats)
    res2 = mod2.fit()
    assert_results_equal(res, res2)
    assert "Model includes 1 other effect" in res2.summary.as_text()


def test_panel_time_other_equivalence(data):
    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit()
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    cats = pd.DataFrame(mod.dependent.time_ids, index=mod.dependent.index)

    mod2 = PanelOLS(y, x, other_effects=cats)
    res2 = mod2.fit()
    assert_results_equal(res, res2)
    assert "Model includes 1 other effect" in res2.summary.as_text()


def test_panel_entity_time_other_equivalence(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit()
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    c1 = mod.dependent.entity_ids
    c2 = mod.dependent.time_ids
    cats = np.c_[c1, c2]
    cats = pd.DataFrame(cats, index=mod.dependent.index)

    mod2 = PanelOLS(y, x, other_effects=cats)
    res2 = mod2.fit()
    assert_results_equal(res, res2)
    assert "Model includes 2 other effects" in res2.summary.as_text()


def test_panel_other_lsdv(data):
    mod = PanelOLS(data.y, data.x, other_effects=data.c)
    assert "Num Other Effects: 2" in str(mod)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    c = mod._other_effect_cats.dataframe.copy()
    d = []
    d_columns = []
    for i, col in enumerate(c):
        s = c[col].copy()
        dummies = pd.get_dummies(
            s.astype(np.int64), drop_first=(mod.has_constant or i > 0)
        )
        dummies.columns = [s.name + "_val_" + str(c) for c in dummies.columns]
        d_columns.extend(list(dummies.columns))
        d.append(dummies.values)
    d = np.column_stack(d)

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ lstsq(z, d, rcond=None)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d_columns))

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_fit=False)

    res3 = mod.fit(
        cov_type="unadjusted", auto_df=False, count_effects=False, debiased=False
    )
    assert_results_equal(res, res3)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(
        cov_type="clustered",
        clusters=clusters,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_time=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.time_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(
        cov_type="clustered",
        cluster_entity=True,
        auto_df=False,
        count_effects=False,
        debiased=False,
    )
    clusters = pd.DataFrame(
        mod.dependent.entity_ids, index=mod.dependent.index, columns=["var.clust"]
    )
    res2 = ols_mod.fit(cov_type="clustered", clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_other_fwl(data):
    mod = PanelOLS(data.y, data.x, other_effects=data.c)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    c = mod._other_effect_cats.dataframe
    d = []
    d_columns = []
    for i, col in enumerate(c):
        s = c[col].copy()
        dummies = pd.get_dummies(
            s.astype(np.int64), drop_first=(mod.has_constant or i > 0)
        )
        dummies.columns = [s.name + "_val_" + str(c) for c in dummies.columns]
        d_columns.extend(list(dummies.columns))
        d.append(dummies.values)
    d = np.column_stack(d)

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ lstsq(z, d, rcond=None)[0]

    x = x - d @ lstsq(d, x, rcond=None)[0]
    y = y - d @ lstsq(d, y, rcond=None)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type="unadjusted")
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type="robust", auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type="robust")
    assert_results_equal(res, res2, test_df=False)


def test_panel_other_incorrect_size(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    cats = pd.DataFrame(mod.dependent.entity_ids, index=mod.dependent.index)
    cats = PanelData(cats)
    cats = cats.dataframe.iloc[: cats.dataframe.shape[0] // 2, :]

    with pytest.raises(ValueError):
        PanelOLS(y, x, other_effects=cats)


def test_results_access(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit()
    access_attributes(res)

    mod = PanelOLS(data.y, data.x, other_effects=data.c)
    res = mod.fit()
    access_attributes(res)

    mod = PanelOLS(data.y, data.x, time_effects=True, entity_effects=True)
    res = mod.fit()
    access_attributes(res)

    mod = PanelOLS(data.y, data.x)
    res = mod.fit()
    access_attributes(res)

    const = PanelData(data.y).copy()
    const.dataframe.iloc[:, :] = 1
    const.dataframe.columns = ["const"]
    mod = PanelOLS(data.y, const)
    res = mod.fit()
    access_attributes(res)


def test_alt_rsquared(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_within)


def test_alt_rsquared_weighted(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, weights=data.w)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_within)


def test_too_many_effects(data):
    with pytest.raises(ValueError):
        PanelOLS(
            data.y, data.x, entity_effects=True, time_effects=True, other_effects=data.c
        )


def test_cov_equiv_cluster(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True, debiased=False)

    y = PanelData(data.y)
    clusters = pd.DataFrame(y.entity_ids, index=y.index)
    res2 = mod.fit(cov_type="clustered", clusters=clusters, debiased=False)
    assert_results_equal(res, res2)

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_time=True, debiased=False)
    y = PanelData(data.y)
    clusters = pd.DataFrame(y.time_ids, index=y.index)
    res2 = mod.fit(cov_type="clustered", clusters=clusters, debiased=False)
    assert_results_equal(res, res2)

    res = mod.fit(cov_type="clustered", debiased=False)
    res2 = mod.fit(cov_type="clustered", clusters=None, debiased=False)
    assert_results_equal(res, res2)


@pytest.mark.smoke
def test_cluster_smoke(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    mod.fit(cov_type="clustered", cluster_time=True, debiased=False)
    mod.fit(cov_type="clustered", cluster_entity=True, debiased=False)
    c2 = PanelData(data.vc2)
    c1 = PanelData(data.vc1)

    mod.fit(cov_type="clustered", clusters=c2, debiased=False)
    mod.fit(cov_type="clustered", cluster_entity=True, clusters=c1, debiased=False)
    mod.fit(cov_type="clustered", cluster_time=True, clusters=c1, debiased=False)
    with pytest.raises(ValueError):
        mod.fit(cov_type="clustered", cluster_time=True, clusters=c2, debiased=False)
    with pytest.raises(ValueError):
        mod.fit(cov_type="clustered", cluster_entity=True, clusters=c2, debiased=False)
    with pytest.raises(ValueError):
        mod.fit(
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
            clusters=c1,
            debiased=False,
        )
    with pytest.raises(ValueError):
        clusters = c1.dataframe.iloc[: c1.dataframe.shape[0] // 2]
        mod.fit(cov_type="clustered", clusters=clusters, debiased=False)


def test_f_pooled(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(debiased=False)

    if mod.has_constant:
        mod2 = PooledOLS(data.y, data.x)
    else:
        exog = mod.exog.dataframe.copy()
        exog["Intercept"] = 1.0
        mod2 = PooledOLS(mod.dependent.dataframe, exog)

    res2 = mod2.fit(debiased=False)

    eps = res.resids.values
    eps2 = res2.resids.values
    v1 = res.df_model - res2.df_model
    v2 = res.df_resid
    f_pool = (eps2.T @ eps2 - eps.T @ eps) / v1
    f_pool /= (eps.T @ eps) / v2
    f_pool = float(f_pool)
    assert_allclose(res.f_pooled.stat, f_pool)
    assert res.f_pooled.df == v1
    assert res.f_pooled.df_denom == v2

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(debiased=False)
    eps = res.resids.values
    eps2 = res2.resids.values
    v1 = res.df_model - res2.df_model
    v2 = res.df_resid
    f_pool = (eps2.T @ eps2 - eps.T @ eps) / v1
    f_pool /= (eps.T @ eps) / v2
    f_pool = float(f_pool)
    assert_allclose(res.f_pooled.stat, f_pool)
    assert res.f_pooled.df == v1
    assert res.f_pooled.df_denom == v2

    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(debiased=False)
    eps = res.resids.values
    eps2 = res2.resids.values
    v1 = res.df_model - res2.df_model
    v2 = res.df_resid
    f_pool = (eps2.T @ eps2 - eps.T @ eps) / v1
    f_pool /= (eps.T @ eps) / v2
    f_pool = float(f_pool)
    assert_allclose(res.f_pooled.stat, f_pool)
    assert res.f_pooled.df == v1
    assert res.f_pooled.df_denom == v2


def test_entity_other(data):
    y = PanelData(data.y)
    x = PanelData(data.x)
    c = PanelData(data.c).copy()
    missing = y.isnull | x.isnull | c.isnull
    y.drop(missing)
    x.drop(missing)
    c.drop(missing)
    c_entity = c.dataframe.copy()
    c_entity.iloc[:, 1] = y.entity_ids.squeeze()
    c_entity = c_entity.astype(np.int64)

    mod = PanelOLS(y, x, other_effects=c_entity)
    res = mod.fit(debiased=False)
    c_only = PanelData(c.dataframe.iloc[:, [0]].astype(np.int64))
    mod2 = PanelOLS(y, x, other_effects=c_only, entity_effects=True)
    res2 = mod2.fit(debiased=False)
    assert_results_equal(res, res2)


@pytest.mark.smoke
def test_other_weighted_smoke(data):
    mod = PanelOLS(data.y, data.x, weights=data.w, other_effects=data.c)
    mod.fit(debiased=False)


@pytest.mark.slow
def test_methods_equivalent(data, lsdv_config):
    other_effects = None
    if lsdv_config.other_effects == 1:
        other_effects = PanelData(data.c).dataframe.iloc[:, [0]]
    elif lsdv_config.other_effects == 2:
        other_effects = data.c
    weights = data.w if lsdv_config.weights else None
    mod = PanelOLS(
        data.y,
        data.x,
        weights=weights,
        entity_effects=lsdv_config.entity_effects,
        time_effects=lsdv_config.time_effects,
        other_effects=other_effects,
    )
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    res3 = mod.fit(use_lsmr=True)
    assert_results_equal(res1, res2)
    assert_results_equal(res2, res3, strict=False)


def test_rsquared_inclusive_equivalence(data):
    mod = PanelOLS(data.y, data.x)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_inclusive)

    mod = PanelOLS(data.y, data.x, weights=data.w)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_inclusive)


def test_panel_effects_sanity(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)

    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)

    mod = PanelOLS(data.y, data.x, weights=data.w, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)

    mod = PanelOLS(
        data.y, data.x, weights=data.w, entity_effects=True, time_effects=True
    )
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)


def test_fitted_effects_residuals(data, entity_eff, time_eff):
    mod = PanelOLS(data.y, data.x, entity_effects=entity_eff, time_effects=time_eff)
    res = mod.fit()

    expected = mod.exog.values2d @ res.params.values
    expected = pd.DataFrame(expected, index=mod.exog.index, columns=["fitted_values"])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    expected.iloc[:, 0] = res.resids
    expected.columns = ["idiosyncratic"]
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    fitted_error = res.fitted_values + res.idiosyncratic.values
    estimated_effects = mod.dependent.values2d - fitted_error
    expected.iloc[:, 0] = estimated_effects.iloc[:, 0]
    expected.columns = ["estimated_effects"]
    assert_allclose(res.estimated_effects, expected, atol=1e-8)
    assert_frame_similar(res.estimated_effects, expected)


@pytest.mark.parametrize("weighted", [True, False])
def test_low_memory(data, weighted):
    if weighted:
        mod = PanelOLS(
            data.y, data.x, weights=data.w, entity_effects=True, time_effects=True
        )
    else:
        mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit()

    low_mem = mod.fit(low_memory=True)
    assert_allclose(res.params, low_mem.params)


def test_low_memory_auto():
    x = np.random.standard_normal((1000, 1000))
    e = np.random.standard_normal((1000, 1000))
    eff = np.arange(1000)[:, None]
    y = x + e + eff + eff.T
    y = y.ravel()
    x = np.reshape(x, (1000000, 1))
    mi = pd.MultiIndex.from_product([np.arange(1000), np.arange(1000)])
    y = pd.Series(y, index=mi)
    x = pd.DataFrame(x, index=mi)
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    with pytest.warns(MemoryWarning):
        mod.fit()


@pytest.mark.filterwarnings("ignore::linearmodels.shared.exceptions.SingletonWarning")
def test_singleton_removal():
    entities = []
    for i in range(6):
        entities.extend([f"entity.{j}" for j in range(6 - i)])
    nobs = len(entities)
    times = np.arange(nobs) % 6
    index = pd.MultiIndex.from_arrays((entities, times))
    cols = [f"x{i}" for i in range(3)]
    x = pd.DataFrame(np.random.randn(nobs, 3), index=index, columns=cols)
    y = pd.DataFrame(np.random.randn(nobs, 1), index=index)
    mod = PanelOLS(y, x, singletons=False, entity_effects=True, time_effects=True)
    res = mod.fit()

    mod = PanelOLS(y, x, singletons=True, entity_effects=True, time_effects=True)
    res_with = mod.fit()
    assert_allclose(res.params, res_with.params)


@pytest.mark.filterwarnings("ignore::linearmodels.shared.exceptions.SingletonWarning")
def test_masked_singleton_removal():
    nobs = 8
    entities = ["A", "B", "C", "D"] * 2
    times = [0, 1, 1, 1, 1, 2, 2, 2]
    index = pd.MultiIndex.from_arrays((entities, times))
    x = pd.DataFrame(np.random.randn(nobs, 1), index=index, columns=["x"])
    y = pd.DataFrame(np.random.randn(nobs, 1), index=index)
    mod = PanelOLS(y, x, singletons=False, entity_effects=True, time_effects=True)
    res = mod.fit()
    assert res.nobs == 6


def test_singleton_removal_other_effects(data):
    mod_keep = PanelOLS(
        data.y, data.x, weights=data.w, other_effects=data.c, singletons=True
    )
    res_keep = mod_keep.fit()

    mod = PanelOLS(
        data.y, data.x, weights=data.w, other_effects=data.c, singletons=False
    )
    res = mod.fit(cov_type="clustered", clusters=data.vc1)

    assert res.nobs <= res_keep.nobs


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::linearmodels.shared.exceptions.SingletonWarning")
@pytest.mark.parametrize("other_effects", [1, 2])
def test_singleton_removal_mixed(singleton_data, other_effects):
    if other_effects == 1:
        other_effects = PanelData(singleton_data.c).dataframe.iloc[:, [0]]
    else:
        other_effects = singleton_data.c
    mod = PanelOLS(singleton_data.y, singleton_data.x, other_effects=other_effects)
    res_keep = mod.fit(use_lsmr=True)

    mod = PanelOLS(
        singleton_data.y,
        singleton_data.x,
        other_effects=other_effects,
        singletons=False,
    )
    res = mod.fit(cov_type="clustered", clusters=singleton_data.vc2, use_lsmr=True)
    assert_allclose(res_keep.params, res.params)
    assert res.nobs <= res_keep.nobs


def test_repeated_measures_weight():
    # Issue reported by email
    rs = np.random.RandomState(0)
    w = rs.chisquare(5, 300) / 5
    idx1 = ["a"] * 100 + ["b"] * 100 + ["c"] * 100
    idx2 = np.arange(300) % 25
    mi = pd.MultiIndex.from_arrays([idx1, idx2])
    df = pd.DataFrame(rs.standard_normal((300, 2)), index=mi, columns=["y", "x"])
    w = pd.Series(w, index=mi, name="weight")
    df["weight"] = w
    mod = PanelOLS.from_formula(
        "y ~ x + EntityEffects + TimeEffects", df, weights=df["weight"]
    )
    res = mod.fit()
    mod = PanelOLS.from_formula("y ~ x + EntityEffects + TimeEffects", df)
    res_un = mod.fit()
    assert res.params.iloc[0] != res_un.params.iloc[0]


def test_absorbed(absorbed_data):
    mod = PanelOLS(
        absorbed_data.y, absorbed_data.x, drop_absorbed=True, entity_effects=True
    )
    if isinstance(absorbed_data.y, pd.DataFrame):
        match = "x_absorbed"
    else:
        match = "Exog.3"
    with pytest.warns(AbsorbingEffectWarning, match=match):
        res = mod.fit()
    if isinstance(absorbed_data.x, np.ndarray):
        x = absorbed_data.x[:-1]
    else:
        x = absorbed_data.x.iloc[:, :-1]
    mod = PanelOLS(absorbed_data.y, x, drop_absorbed=False, entity_effects=True)
    res_no = mod.fit()
    assert_allclose(res.params, res_no.params)
    assert_results_equal(res, res_no)


def test_absorbed_option(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, drop_absorbed=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)
    mod = PanelOLS(data.y, data.x, entity_effects=True, drop_absorbed=False)
    res_false = mod.fit(auto_df=False, count_effects=False, debiased=False)
    assert_results_equal(res, res_false)


def test_fully_absorbed():
    x = np.arange(10)
    x = np.repeat(x, (2,))[:, None]
    y = x @ np.array([1]) + np.random.standard_normal(x.shape[0])
    mi = pd.MultiIndex.from_product([np.arange(10), [1, 2]])
    x = pd.DataFrame(x, index=mi, columns=["x"])
    y = pd.Series(y, index=mi, name="y")
    with pytest.raises(ValueError, match="All columns in exog have been fully"):
        PanelOLS(y, x, drop_absorbed=True, entity_effects=True).fit()


def test_zero_endog():
    x = np.arange(10)
    x = np.repeat(x, (2,))[:, None]
    y = x @ np.array([0])
    mi = pd.MultiIndex.from_product([np.arange(10), [1, 2]])
    x = pd.DataFrame(x, index=mi, columns=["x"])
    y = pd.Series(y, index=mi, name="y")
    PanelOLS(y, x).fit()


def test_f_after_drop():
    rg = np.random.default_rng(918273645)
    y = pd.Series(rg.standard_normal(1000))
    a1 = np.arange(1000) % 10
    a2 = np.arange(1000) // 100

    x = pd.DataFrame(
        {"x": rg.standard_normal(1000), "a1": a1, "a2": a2, "c": np.ones(1000)}
    )
    mi = pd.MultiIndex.from_product([list(range(100)), list(range(10))])
    y.index = mi
    x.index = mi
    mod = PanelOLS(y, x, drop_absorbed=True, entity_effects=True, time_effects=True)
    with pytest.warns(AbsorbingEffectWarning):
        res = mod.fit()
    assert isinstance(res.f_statistic, WaldTestStatistic)
    assert isinstance(res.f_statistic_robust, WaldTestStatistic)
    assert res.f_statistic.stat > 0
    assert res.f_statistic_robust.stat > 0


def test_predict_incorrect(data):
    mod = PanelOLS(data.y, data.x)
    res = mod.fit()
    with pytest.raises(ValueError, match="exog does not have the correct"):
        mod.predict(res.params.iloc[:-1], exog=data.x)
    exog = np.asarray(data.x)
    if exog.ndim == 3:
        exog = exog[:-1]
    else:
        exog = exog[:, :-1]
    with pytest.raises(ValueError, match="exog does not have the correct"):
        mod.predict(res.params, exog=exog)


@pytest.mark.parametrize(
    "cov_config",
    [
        ("clustered", "cluster"),
        ("unadjusted", "bandwidth"),
        ("kernel", "bw"),
        ("robust", "clusters"),
    ],
)
def test_unknown_covconfig_kwargs(data, cov_config):
    # GH342
    c, fig = cov_config
    mod = PanelOLS(data.y, data.x)
    if c == "clustered":
        cov = "ClusteredCovariance"
    elif c == "kernel":
        cov = "DriscollKraay"
    elif c == "robust":
        cov = "HeteroskedasticCovariance"
    else:
        cov = "HomoskedasticCovariance"
    with pytest.raises(ValueError, match=f"Covariance estimator {cov}"):
        mod.fit(cov_type=c, **{fig: data.vc1})


# Reported by email
def test_corr_squared(data):
    mod = PanelOLS(data.y, data.x)
    res = mod.fit()
    has_const = np.any(np.all(mod._x == 1.0, 0))
    if has_const:
        assert_allclose(res.rsquared_overall, res.corr_squared_overall)


def test_quoted_var_names():
    mi = pd.MultiIndex.from_product([np.arange(20), np.arange(5)])
    d = pd.DataFrame({"var a": np.arange(100.0)}, index=mi)
    res = PanelOLS.from_formula("`var a` ~ 1", data=d).fit()
    assert_allclose(res.params, d.mean().iloc[0])


def test_entity_into():
    # GH 534
    rg = np.random.default_rng(12345)
    mi = pd.MultiIndex.from_product([np.arange(20), np.arange(5)])
    df = pd.DataFrame(
        {
            "a": rg.standard_normal(100),
            "b": rg.standard_normal(100),
        },
        index=mi,
    )
    res = PanelOLS.from_formula("a ~ 1 + b", data=df).fit()
    ei = res.entity_info
    assert ei["total"] == 20
    assert ei["min"] == 5
    ti = res.time_info
    assert ti["total"] == 5
    assert ti["min"] == 20

    df2 = df.drop([2, 4, 6, 8], level=0)
    df2 = df2.drop(0, level=1)
    res = PanelOLS.from_formula("a ~ 1 + b", data=df2).fit()
    ei = res.entity_info
    assert ei["min"] == 4
    assert ei["total"] == 16
    ti = res.time_info
    assert ti["total"] == 4
    assert ti["min"] == 16


@pytest.mark.parametrize("path", ["use_lsdv", "low_memory", ""])
def test_absorbed_with_weights(path):
    data = wage_panel.load().copy()
    year = pd.Categorical(data.year)
    data = data.set_index(["nr", "year"])
    data["year"] = year
    # and random number between 0 and 1 for weights
    data["rand"] = np.random.rand(data.shape[0])
    data["absorbe"] = data.groupby("nr")["union"].transform("mean")

    fit_options = {}
    if path:
        fit_options[path] = True

    with pytest.warns(AbsorbingEffectWarning, match="Variables have been"):
        PanelOLS.from_formula(
            "lwage ~1+absorbe + married + EntityEffects",
            data=data,
            weights=data["rand"],
            drop_absorbed=True,
        ).fit(**fit_options)
