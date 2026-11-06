import linearmodels.compat.statsmodels
from linearmodels.compat.statsmodels import Summary

import copy
import warnings

import numpy as np
from numpy.linalg import pinv
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
import scipy.linalg
from statsmodels.tools.tools import add_constant

from linearmodels.datasets import card
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML
from linearmodels.iv.model import _OLS
from linearmodels.iv.results import compare
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.shared.utility import AttrDict
from linearmodels.tests._utility import method_kwargs


def gen_data(n=1000):
    q, k, p = 2, 5, 3
    rs = np.random.RandomState(12345)
    clusters = rs.randint(0, 10, n)

    rho = 0.5
    r = scipy.linalg.toeplitz([1, *(rho + np.linspace(0.1, -0.1, 8)).tolist()])
    r[-1, 2:] = 0
    r[2:, -1] = 0
    r[-1, -1] = 1
    v = rs.multivariate_normal(np.zeros(r.shape[0]), r, n)
    x = v[:, :k]
    z = v[:, k : k + p]
    e = v[:, [-1]]
    params = np.arange(1, k + 1) / k
    params = params[:, None]
    y = x @ params + e
    exog_instr = np.column_stack((x[:, q:], z))
    xhat = exog_instr @ np.linalg.pinv(exog_instr) @ x
    nobs, nvar = x.shape
    s2 = e.T @ e / nobs
    s2_debiased = e.T @ e / (nobs - nvar)
    v = xhat.T @ xhat / nobs
    vinv = np.linalg.inv(v)
    kappa = 0.99
    vk = (x.T @ x * (1 - kappa) + kappa * xhat.T @ xhat) / nobs
    return AttrDict(
        nobs=nobs,
        e=e,
        x=x,
        y=y,
        z=z,
        xhat=xhat,
        params=params,
        s2=s2,
        s2_debiased=s2_debiased,
        clusters=clusters,
        nvar=nvar,
        v=v,
        vinv=vinv,
        vk=vk,
        kappa=kappa,
        dep=y,
        exog=x[:, q:],
        endog=x[:, :q],
        instr=z,
    )


@pytest.fixture(scope="module")
def data():
    return gen_data()


@pytest.fixture(scope="module")
def small_data():
    return gen_data(n=250)


def get_all(v):
    attr = [d for d in dir(v) if not d.startswith("_")]
    for a in attr:
        val = getattr(v, a)
        if a in ("conf_int", "durbin", "wu_hausman", "c_stat"):
            val()


def test_rank_deficient_exog_exception(data):
    exog = data.exog.copy()
    exog[:, :2] = 1
    with pytest.raises(ValueError, match=r"regressors \[exog endog\] do not have"):
        IV2SLS(data.dep, exog, data.endog, data.instr, method="pinv")


def test_rank_deficient_endog_exception(data):
    endog = data.endog.copy()
    endog[:, :2] = 1
    with pytest.raises(ValueError, match=r"regressors \[exog endog\] do not have"):
        IV2SLS(data.dep, data.exog, endog, data.instr, method="pinv")
    with pytest.raises(ValueError, match=r"regressors \[exog endog\] do not have"):
        IV2SLS(data.dep, data.exog, data.exog, data.instr, method="pinv")


def test_invalid_weights_exception(data):
    weights = np.zeros_like(data.dep)
    with pytest.raises(ValueError, match=r"weights must be strictly positive."):
        IV2SLS(
            data.dep, data.exog, data.endog, data.instr, weights=weights, method="pinv"
        )


def test_rank_deficient_instr_exception(data):
    instr = data.instr.copy()
    instr[:, :2] = 1
    with pytest.raises(ValueError, match=r"instruments \[exog instruments\]"):
        IV2SLS(data.dep, data.exog, data.endog, instr, method="pinv")
    with pytest.raises(ValueError, match=r"instruments \[exog instruments\]"):
        IV2SLS(data.dep, data.exog, data.endog, data.exog, method="pinv")


def test_kappa_error_exception(data):
    with pytest.raises(ValueError, match=r"kappa must be None or a scalar"):
        IVLIML(
            data.dep,
            data.exog,
            data.endog,
            data.instr,
            kappa=np.array([1]),
            method="pinv",
        )


def test_fuller_error_exception(data):
    with pytest.raises(ValueError, match=r"fuller must be None or a scalar"):
        IVLIML(
            data.dep,
            data.exog,
            data.endog,
            data.instr,
            fuller=np.array([1]),
            method="pinv",
        )


def test_kappa_fuller_warning_exception(data):
    with warnings.catch_warnings(record=True) as w:
        IVLIML(
            data.dep,
            data.exog,
            data.endog,
            data.instr,
            kappa=0.99,
            fuller=1,
            method="pinv",
        )
    assert len(w) == 1


def test_string_cat_exception(data):
    instr = data.instr.copy()
    n = data.instr.shape[0]
    cat = pd.Series(["a"] * (n // 2) + ["b"] * (n // 2))
    instr = pd.DataFrame(instr)
    instr["cat"] = cat
    res = IV2SLS(data.dep, data.exog, data.endog, instr, method="pinv").fit(
        cov_type="unadjusted"
    )
    instr["cat"] = cat.astype("category")
    res_cat = IV2SLS(data.dep, data.exog, data.endog, instr, method="pinv").fit(
        cov_type="unadjusted"
    )
    assert_series_equal(res.params, res_cat.params)


def test_no_regressors_exception(data):
    with pytest.raises(ValueError, match=r"Model must contain at least one"):
        IV2SLS(data.dep, None, None, None, method="pinv")


def test_too_few_instruments_exception(data):
    with pytest.raises(ValueError, match=r"The number of instruments"):
        IV2SLS(data.dep, data.exog, data.endog, None, method="pinv")


def test_2sls_direct(data):
    mod = IV2SLS(
        data.dep, add_constant(data.exog), data.endog, data.instr, method="pinv"
    )
    res = mod.fit()
    x = np.c_[add_constant(data.exog), data.endog]
    z = np.c_[add_constant(data.exog), data.instr]
    y = data.y
    xhat = z @ pinv(z) @ x
    params = pinv(xhat) @ y
    assert_allclose(res.params, params.ravel())
    # This is just a quick smoke check of results
    get_all(res)


def test_2sls_direct_small(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv")
    res = mod.fit()
    res2 = mod.fit(debiased=True)
    assert np.all(res.tstats != res2.tstats)
    get_all(res2)
    fs = res.first_stage
    assert isinstance(fs.diagnostics, pd.DataFrame)
    # Fetch again to test cache
    get_all(res2)


def test_liml_direct(data):
    mod = IVLIML(data.dep, data.exog, data.endog, data.instr, method="pinv")
    nobs = data.dep.shape[0]
    ninstr = data.exog.shape[1] + data.instr.shape[1]
    res = mod.fit()
    get_all(res)
    mod2 = IVLIML(
        data.dep, data.exog, data.endog, data.instr, kappa=res.kappa, method="pinv"
    )
    res2 = mod2.fit()
    assert_allclose(res.params, res2.params)
    mod3 = IVLIML(data.dep, data.exog, data.endog, data.instr, fuller=1, method="pinv")
    res3 = mod3.fit()
    assert_allclose(res3.kappa, res.kappa - 1 / (nobs - ninstr))


def test_2sls_ols_equiv(data):
    mod = IV2SLS(data.dep, data.exog, None, None, method="pinv")
    res = mod.fit()
    params = pinv(data.exog) @ data.dep
    assert_allclose(res.params, params.ravel())


def test_gmm_iter(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit(iter_limit=100)
    assert res.iterations > 2
    # This is just a quick smoke check of results
    get_all(res)


def test_gmm_cue(small_data):
    mod = IVGMMCUE(small_data.dep, small_data.exog, small_data.endog, small_data.instr)
    res = mod.fit(display=False)
    assert res.iterations > 2
    # CUE centers by default
    mod2 = IVGMM(
        small_data.dep,
        small_data.exog,
        small_data.endog,
        small_data.instr,
        center=True,
    )
    res2 = mod2.fit()
    assert res.j_stat.stat <= res2.j_stat.stat

    mod = IVGMMCUE(
        small_data.dep,
        small_data.exog,
        small_data.endog,
        small_data.instr,
        center=False,
    )
    res = mod.fit(display=False)
    mod2 = IVGMM(
        small_data.dep,
        small_data.exog,
        small_data.endog,
        small_data.instr,
        center=False,
    )
    res2 = mod2.fit()
    assert res.j_stat.stat <= res2.j_stat.stat


def test_gmm_cue_starting_vals(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    sv = mod.fit().params
    mod = IVGMMCUE(data.dep, data.exog, data.endog, data.instr)
    mod.fit(starting=sv, display=False)

    with pytest.raises(ValueError, match=r"starting does not have the correct"):
        mod.fit(starting=sv[:-1], display=True)


def test_2sls_just_identified(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr[:, :2], method="pinv")
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    assert isinstance(fs.diagnostics, pd.DataFrame)
    # Fetch again to test cache
    get_all(fs)
    get_all(res)

    mod = IV2SLS(
        data.dep, data.exog, data.endog[:, :1], data.instr[:, :1], method="pinv"
    )
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    assert isinstance(fs.diagnostics, pd.DataFrame)
    get_all(fs)

    mod = IV2SLS(data.dep, None, data.endog[:, :1], data.instr[:, :1], method="pinv")
    res = mod.fit()
    get_all(res)
    fs = res.first_stage
    assert isinstance(fs.diagnostics, pd.DataFrame)
    get_all(fs)


@pytest.mark.smoke
def test_durbin_smoke(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv")
    res = mod.fit()
    res.durbin()
    res.durbin([mod.endog.cols[1]])


@pytest.mark.smoke
def test_wu_hausman_smoke(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv")
    res = mod.fit()
    res.wu_hausman()
    res.wu_hausman([mod.endog.cols[1]])


@pytest.mark.smoke
def test_wooldridge_smoke(data):
    mod = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv")
    res = mod.fit()
    assert isinstance(res.wooldridge_regression, WaldTestStatistic)
    assert isinstance(res.wooldridge_score, WaldTestStatistic)


@pytest.mark.smoke
def test_model_summary_smoke(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit()
    res.__repr__()
    res.__str__()
    res._repr_html_()
    assert isinstance(res.summary, linearmodels.compat.statsmodels.Summary)

    res = _OLS(data.dep, data.exog).fit()
    res.__repr__()
    res.__str__()
    res._repr_html_()
    assert isinstance(res.summary, linearmodels.compat.statsmodels.Summary)


def test_model_missing(data):

    data2 = AttrDict()
    for key in data:
        data2[key] = copy.deepcopy(data[key])
    data = data2
    data.dep[::7, :] = np.nan
    data.exog[::13, :] = np.nan
    data.endog[::23, :] = np.nan
    data.instr[::29, :] = np.nan
    with warnings.catch_warnings(record=True) as w:
        mod = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv")
    assert len(w) == 1
    res = mod.fit()

    var_names = [data.dep, data.exog, data.endog, data.instr]
    missing = [np.any(np.isnan(var_name), 1) for var_name in var_names]
    missing = np.any(np.c_[missing], 0)
    not_missing = missing.shape[0] - missing.sum()
    assert res.nobs == not_missing
    assert_equal(mod.isnull, missing)
    assert_equal(mod.notnull, ~missing)


@pytest.mark.parametrize("stars", [False, True])
def test_compare(data, stars):
    res1 = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit()
    res2 = IV2SLS(
        data.dep, data.exog, data.endog, data.instr[:, :-1], method="pinv"
    ).fit()
    res3 = IVGMM(data.dep, data.exog[:, :2], data.endog, data.instr).fit()
    res4 = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit()
    c = compare([res1, res2, res3, res4], stars=stars)
    assert len(c.rsquared) == 4
    assert isinstance(str(c.summary), str)
    if stars:
        total = 1 * (c.pvalues < 0.10) + (c.pvalues < 0.05) + (c.pvalues < 0.01)
        total_stars = np.asarray(total).sum()
        count = sum(char == "*" for char in str(c.summary))
        print(c.pvalues)
        print(total)
        print(c.summary)
        assert count == total_stars

    c = compare({"Model A": res1, "Model B": res2, "Model C": res3, "Model D": res4})
    assert isinstance(str(c.summary), str)
    res = {"Model A": res1, "Model B": res2, "Model C": res3, "Model D": res4}
    c = compare(res, stars=stars)
    assert isinstance(str(c.summary), str)
    assert isinstance(c.pvalues, pd.DataFrame)

    res1 = IV2SLS(data.dep, data.exog[:, :1], None, None, method="pinv").fit()
    res2 = IV2SLS(data.dep, data.exog[:, :2], None, None, method="pinv").fit()
    c = compare({"Model A": res1, "Model B": res2}, stars=stars)
    assert isinstance(str(c.summary), str)


def test_compare_single(data):
    res1 = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit()
    c = compare([res1])
    assert len(c.rsquared) == 1
    assert isinstance(c.summary, Summary)
    c = compare({"Model A": res1})
    assert isinstance(c.summary, Summary)
    res = {"Model A": res1}
    c = compare(res)
    assert isinstance(c.summary, Summary)
    assert isinstance(c.pvalues, pd.DataFrame)


def test_compare_single_single_parameter(data):
    res1 = IV2SLS(data.dep, data.exog[:, :1], None, None, method="pinv").fit()
    c = compare([res1])
    assert len(c.rsquared) == 1
    assert isinstance(c.summary, Summary)


def test_first_stage_summary(data):
    res1 = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit()
    assert isinstance(res1.first_stage.summary, Summary)


# The default bandwidth of the kernel covariance warns, which is tested in
# test_gmm_kernel_bandwidth.py, so the bandwidth is given
def test_gmm_str(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    assert isinstance(str(mod.fit(cov_type="unadjusted")), str)
    assert isinstance(str(mod.fit(cov_type="robust")), str)
    assert isinstance(str(mod.fit(cov_type="clustered", clusters=data.clusters)), str)
    assert isinstance(str(mod.fit(cov_type="kernel", bandwidth=5)), str)


def test_gmm_cue_optimization_options(small_data):
    mod = IVGMMCUE(small_data.dep, small_data.exog, small_data.endog, small_data.instr)
    res_none = mod.fit(display=False)
    opt_options = {"method": "BFGS", "options": {"disp": False}}
    res_bfgs = mod.fit(display=False, opt_options=opt_options)
    opt_options = {"method": "L-BFGS-B"}
    res_lbfgsb = mod.fit(display=False, opt_options=opt_options)
    assert res_none.iterations > 2
    assert res_bfgs.iterations > 2
    assert res_lbfgsb.iterations >= 1

    # CUE centers by default
    mod2 = IVGMM(
        small_data.dep,
        small_data.exog,
        small_data.endog,
        small_data.instr,
        center=True,
    )
    res2 = mod2.fit()
    assert res_none.j_stat.stat <= res2.j_stat.stat
    assert res_bfgs.j_stat.stat <= res2.j_stat.stat
    assert res_lbfgsb.j_stat.stat <= res2.j_stat.stat


def test_weighted_r2():
    # GH 274
    data = card.load()
    data = add_constant(data)

    dep = ["wage"]
    endog = ["educ"]
    exog = [
        "const",
        "exper",
        "expersq",
        "black",
        "smsa",
        "south",
        "smsa66",
        "reg662",
        "reg663",
        "reg664",
        "reg665",
        "reg666",
        "reg667",
        "reg668",
        "reg669",
    ]
    instr = ["nearc4"]
    data = data[dep + exog + endog + instr].dropna()

    rs = np.random.RandomState(12345)
    weights = rs.random([len(data)]) ** 3
    res = IV2SLS(
        data.educ, data[instr + exog], None, None, weights=weights, method="pinv"
    ).fit()

    res_2sls = IV2SLS(
        np.log(data.wage),
        data[exog],
        data[endog],
        data[instr],
        weights=weights,
        method="pinv",
    ).fit()
    fs_educ = res_2sls.first_stage.individual["educ"]
    assert_allclose(fs_educ.rsquared, res.rsquared)
    index = res.params.index
    assert_allclose(fs_educ.params[index], res.params[index])


def test_initial_weight(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit(iter_limit=1)
    z = np.concatenate([data.exog, data.instr], 1)
    ze = z + np.random.standard_normal(size=z.shape)
    w0 = ze.T @ ze / ze.shape[0]
    res0 = mod.fit(initial_weight=w0, iter_limit=1)
    assert np.any(res0.params != res.params)


def test_initial_weight_error(data):
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    z = np.concatenate([data.exog, data.instr], 1)
    ze = z + np.random.standard_normal(size=z.shape)
    w0 = ze.T @ ze / ze.shape[0]
    with pytest.raises(ValueError, match=r"initial_weight must"):
        mod.fit(initial_weight=w0[:-1, :-1])
    with pytest.raises(ValueError, match=r"initial_weight must"):
        mod.fit(initial_weight=w0[:-1])


def test_first_state_f(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit(
        cov_type="unadjusted"
    )
    diag = res.first_stage.diagnostics
    endog = np.asarray(data.endog)
    rhs = np.c_[np.asarray(data.exog), np.asarray(data.instr)]
    nz = data.instr.shape[1]
    null_rest = np.zeros((nz, rhs.shape[1]))
    null_rest[:, -nz:] = np.eye(nz)
    null_value = np.zeros(nz)
    for i in range(endog.shape[1]):
        indiv = _OLS(endog[:, i], rhs).fit(cov_type="unadjusted")
        stat = indiv.wald_test(null_rest, null_value)
        assert_allclose(diag.iloc[i].loc["f.stat"], stat.stat / nz)


# The weighted copies are made in _IVModelBase, which every IV estimator uses
UNIT_WEIGHT_MODELS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]


@pytest.mark.parametrize("model", UNIT_WEIGHT_MODELS)
def test_weights_all_one_not_materialized(data, model):
    # GH 607: weighted copies of the data are not stored when all of the
    # weights are 1, which is the case when weights is None or constant
    mod = model(data.dep, data.exog, data.endog, data.instr, **method_kwargs(model))
    assert mod._wy is mod._y
    assert mod._wx is mod._x
    assert mod._wz is mod._z

    weights = np.full(data.nobs, 2.5)
    mod = model(
        data.dep,
        data.exog,
        data.endog,
        data.instr,
        weights=weights,
        **method_kwargs(model),
    )
    assert mod._wy is mod._y
    assert mod._wx is mod._x
    assert mod._wz is mod._z


def test_weights_nonconstant_materialized(data):
    rng = np.random.default_rng(12345)
    weights = 0.5 + rng.random(data.nobs)
    mod = IV2SLS(
        data.dep, data.exog, data.endog, data.instr, weights=weights, method="pinv"
    )
    assert mod._wz is not mod._z
    root_w = np.sqrt(mod.weights.ndarray)
    assert_allclose(mod._wy, mod._y * root_w)
    assert_allclose(mod._wx, mod._x * root_w)
    assert_allclose(mod._wz, mod._z * root_w)


@pytest.mark.parametrize("model", UNIT_WEIGHT_MODELS)
def test_weights_all_one_results_unchanged(data, model):
    # Estimation with no weights or constant weights is identical to
    # explicitly weighting by the square root of the normalized weights
    res = model(
        data.dep, data.exog, data.endog, data.instr, **method_kwargs(model)
    ).fit()
    weights = np.full(data.nobs, 3.0)
    res_const = model(
        data.dep,
        data.exog,
        data.endog,
        data.instr,
        weights=weights,
        **method_kwargs(model),
    ).fit()
    assert_allclose(res_const.params, res.params, rtol=1e-12)
    assert_allclose(res_const.cov, res.cov, rtol=1e-12)


def test_weights_float32_inputs():
    # GH 607: float32 data with float32 weights produces the same estimates
    # as the float64 data, up to the precision of the float32 inputs, and
    # constant float32 weights do not materialize weighted copies. IVData
    # stores the data as float64, so the arrays of the model are float64
    rng = np.random.default_rng(12345)
    n = 500
    dep = rng.standard_normal((n, 1))
    exog = rng.standard_normal((n, 2))
    endog = rng.standard_normal((n, 1))
    instr = rng.standard_normal((n, 2))
    weights = 0.5 + rng.random(n)

    res64 = IV2SLS(dep, exog, endog, instr, weights=weights, method="pinv").fit()
    res32 = IV2SLS(
        dep.astype(np.float32),
        exog.astype(np.float32),
        endog.astype(np.float32),
        instr.astype(np.float32),
        weights=weights.astype(np.float32),
        method="pinv",
    ).fit()
    assert_allclose(res32.params, res64.params, rtol=1e-5, atol=1e-6)

    mod32 = IV2SLS(
        dep.astype(np.float32),
        exog.astype(np.float32),
        endog.astype(np.float32),
        instr.astype(np.float32),
        weights=np.ones(n, dtype=np.float32),
        method="pinv",
    )
    assert mod32._wz is mod32._z
    assert mod32._z.dtype == np.float64


def test_weights_float32_all_one_fit():
    # GH 607: float32 data with all-one float32 weights are fitted through
    # the unmaterialized path, and the estimates agree with the estimates
    # of the same data stored as float64
    rng = np.random.default_rng(12345)
    n = 500
    dep = rng.standard_normal((n, 1)).astype(np.float32)
    exog = rng.standard_normal((n, 2)).astype(np.float32)
    endog = rng.standard_normal((n, 1)).astype(np.float32)
    instr = rng.standard_normal((n, 2)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)

    mod = IV2SLS(dep, exog, endog, instr, weights=weights, method="pinv")
    assert mod._wy is mod._y
    assert mod._wx is mod._x
    assert mod._wz is mod._z
    res = mod.fit()
    res64 = IV2SLS(
        dep.astype(np.float64),
        exog.astype(np.float64),
        endog.astype(np.float64),
        instr.astype(np.float64),
        method="pinv",
    ).fit()
    assert_allclose(res.params, res64.params, rtol=1e-6, atol=1e-8)
    assert_allclose(res.cov, res64.cov, rtol=1e-5, atol=1e-9)


def test_weights_integer_inputs_fit():
    # GH 607: integer data with constant integer weights are fitted through
    # the unmaterialized path, and the estimates are identical to the
    # estimates of the same data stored as float64
    rng = np.random.default_rng(12345)
    n = 500
    dep = np.round(10 * rng.standard_normal((n, 1))).astype(np.int64)
    exog = np.round(10 * rng.standard_normal((n, 2))).astype(np.int64)
    endog = np.round(10 * rng.standard_normal((n, 1))).astype(np.int64)
    instr = np.round(10 * rng.standard_normal((n, 2))).astype(np.int64)
    weights = np.full(n, 2)

    mod = IV2SLS(dep, exog, endog, instr, weights=weights, method="pinv")
    assert mod._wy is mod._y
    assert mod._wx is mod._x
    assert mod._wz is mod._z
    res = mod.fit()
    res64 = IV2SLS(
        dep.astype(np.float64),
        exog.astype(np.float64),
        endog.astype(np.float64),
        instr.astype(np.float64),
        weights=np.full(n, 2.0),
        method="pinv",
    ).fit()
    assert_allclose(res.params, res64.params, rtol=1e-12)
    assert_allclose(res.cov, res64.cov, rtol=1e-12)


def test_weights_all_one_matches_materialized_reference(data):
    # GH 607: the estimates from the unmaterialized path agree with a
    # reference computed directly from explicitly materialized weighted
    # data, which is what the model stored before the copies were skipped
    y = np.asarray(data.dep, dtype=np.float64)
    x = np.c_[
        np.asarray(data.exog, dtype=np.float64),
        np.asarray(data.endog, dtype=np.float64),
    ]
    z = np.c_[
        np.asarray(data.exog, dtype=np.float64),
        np.asarray(data.instr, dtype=np.float64),
    ]
    n = y.shape[0]
    root_w = np.ones((n, 1))
    wy, wx, wz = y * root_w, x * root_w, z * root_w
    pz = wz @ pinv(wz.T @ wz) @ wz.T
    x_pz_x_inv = pinv(wx.T @ pz @ wx)
    params_ref = x_pz_x_inv @ (wx.T @ pz @ wy)
    resids_ref = wy - wx @ params_ref
    s2 = float(np.squeeze(resids_ref.T @ resids_ref)) / n
    cov_ref = s2 * x_pz_x_inv

    res = IV2SLS(data.dep, data.exog, data.endog, data.instr, method="pinv").fit(
        cov_type="unadjusted", debiased=False
    )
    assert_allclose(res.params, params_ref.ravel(), rtol=1e-8, atol=1e-10)
    assert_allclose(
        np.asarray(res.resids).ravel(), resids_ref.ravel(), rtol=1e-8, atol=1e-10
    )
    assert_allclose(res.cov, cov_ref, rtol=1e-8, atol=1e-12)


def test_weights_single_non_one_materialized(data):
    # GH 607: a single weight that is not 1 takes the materializing path,
    # and the stored data is scaled by the square root of the weights
    weights = np.ones(data.nobs)
    weights[0] = 2.0
    mod = IV2SLS(
        data.dep, data.exog, data.endog, data.instr, weights=weights, method="pinv"
    )
    assert mod._wy is not mod._y
    assert mod._wx is not mod._x
    assert mod._wz is not mod._z
    root_w = np.sqrt(mod.weights.ndarray)
    assert_allclose(mod._wy, mod._y * root_w)
    assert_allclose(mod._wx, mod._x * root_w)
    assert_allclose(mod._wz, mod._z * root_w)


def test_weights_nan_weight_drops_observation():
    # GH 607: a NaN weight is a missing value, so the observation is
    # dropped, as it was before the weighted copies were skipped, and the
    # remaining all-one weights use the unmaterialized path
    rng = np.random.default_rng(12345)
    n = 200
    dep = rng.standard_normal((n, 1))
    exog = np.column_stack([np.ones(n), rng.standard_normal(n)])
    endog = rng.standard_normal((n, 1))
    instr = rng.standard_normal((n, 2))
    weights = np.ones(n)
    weights[3] = np.nan

    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = IV2SLS(dep, exog, endog, instr, weights=weights, method="pinv")
    res = mod.fit()
    keep = np.ones(n, dtype=bool)
    keep[3] = False
    res_ref = IV2SLS(
        dep[keep], exog[keep], endog[keep], instr[keep], method="pinv"
    ).fit()
    assert res.nobs == n - 1
    assert_allclose(res.params, res_ref.params, rtol=1e-12)
    assert mod._wy is mod._y
    assert mod._wx is mod._x
    assert mod._wz is mod._z
