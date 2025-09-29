import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy import stats
from statsmodels.tools.tools import add_constant

from linearmodels.asset_pricing.model import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv.data import IVData
from linearmodels.iv.model import _OLS
from linearmodels.tests.asset_pricing._utility import generate_data, get_all

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)


@pytest.fixture(params=["numpy", "pandas"])
def data(request):
    return generate_data(nportfolio=10, output=request.param)


@pytest.mark.smoke
def test_linear_model_gmm_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="robust", disp=5)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_smoke_iterate(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="robust", disp=5, steps=20)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_smoke_risk_free(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type="robust", disp=10)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_kernel_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="kernel", disp=10)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_kernel_bandwidth_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    res = mod.fit(cov_type="kernel", bandwidth=10, disp=10)
    get_all(res)


@pytest.mark.smoke
def test_linear_model_gmm_cue_smoke(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors, risk_free=True)
    res = mod.fit(cov_type="robust", disp=10, use_cue=True)
    get_all(res)


def test_linear_model_time_series(data):
    mod = TradedFactorModel(data.portfolios, data.factors)
    mod.fit()
    res = mod.fit()
    get_all(res)
    all_params = []
    all_tstats = []
    nobs, nport = data.portfolios.shape
    nf = data.factors.shape[1]
    e = np.zeros((nobs, (nport * (nf + 1))))
    x = np.zeros((nobs, (nport * (nf + 1))))
    factors = add_constant(data.factors)
    loc = 0
    for i in range(data.portfolios.shape[1]):
        if isinstance(data.portfolios, pd.DataFrame):
            p = data.portfolios.iloc[:, i : (i + 1)]
        else:
            p = data.portfolios[:, i : (i + 1)]
        ols_res = _OLS(p, factors).fit(cov_type="robust", debiased=True)
        all_params.extend(list(ols_res.params))
        all_tstats.extend(list(ols_res.tstats))
        x[:, loc : (loc + nf + 1)] = factors
        e[:, loc : (loc + nf + 1)] = ols_res.resids.values[:, None]
        loc += nf + 1
        cov = res.cov.values[
            (nf + 1) * i : (nf + 1) * (i + 1), (nf + 1) * i : (nf + 1) * (i + 1)
        ]

        assert_allclose(cov, np.asarray(ols_res.cov))
    assert_allclose(np.asarray(res.params).ravel(), np.array(all_params))
    assert_allclose(np.asarray(res.tstats).ravel(), np.array(all_tstats))
    assert_allclose(res.risk_premia, np.asarray(factors).mean(0)[1:])

    xpxi_direct = np.eye((nf + 1) * nport + nf)
    f = np.asarray(factors)
    fpfi = np.linalg.inv(f.T @ f / nobs)
    nfp1 = nf + 1
    for i in range(nport):
        st, en = i * nfp1, (i + 1) * nfp1
        xpxi_direct[st:en, st:en] = fpfi
    f = np.asarray(factors)[:, 1:]
    xe = np.c_[x * e, f - f.mean(0)[None, :]]
    xeex_direct = xe.T @ xe / nobs
    cov = xpxi_direct @ xeex_direct @ xpxi_direct / (nobs - nfp1)
    assert_allclose(cov, np.asarray(res.cov))

    alphas = np.array(all_params)[0::nfp1][:, None]
    alpha_cov = cov[0 : (nfp1 * nport) : nfp1, 0 : (nfp1 * nport) : nfp1]
    stat_direct = float(np.squeeze(alphas.T @ np.linalg.inv(alpha_cov) @ alphas))
    assert_allclose(res.j_statistic.stat, stat_direct)
    assert_allclose(1.0 - stats.chi2.cdf(stat_direct, nport), res.j_statistic.pval)


@pytest.mark.smoke
def test_linear_model_time_series_kernel_smoke(data):
    mod = TradedFactorModel(data.portfolios, data.factors)
    mod.fit(cov_type="kernel")


def test_linear_model_time_series_error(data):
    mod = TradedFactorModel(data.portfolios, data.factors)
    with pytest.raises(ValueError, match="Unknown cov_type"):
        mod.fit(cov_type="unknown")


def test_errors(data):
    p = data.portfolios.copy()
    f = data.factors.copy()
    if isinstance(p, pd.DataFrame):
        p2 = p.copy()
        p3 = p.copy().iloc[:-1]
        p4 = p.copy()
        p5 = p.copy().iloc[: f.shape[1] - 1, :1]
        p4 = p4.iloc[:, : (f.shape[1] - 1)]
        p2["dupe"] = p.iloc[:, 0]
        p["const"] = 1.0

        f5 = f.copy()
        f5 = f5.iloc[: p5.shape[0]]
        f2 = f.copy()
        f2["dupe"] = f.iloc[:, 0]
        f["const"] = 1.0
    else:
        p2 = np.c_[p, p[:, [0]]]
        p3 = p.copy()[:-1]
        p4 = p.copy()
        p5 = p.copy()[: f.shape[1] - 1, :1]
        p4 = p4[:, : (f.shape[1] - 1)]
        p = np.c_[np.ones((p.shape[0], 1)), p]

        f5 = f.copy()
        f5 = f5[: p5.shape[0]]
        f2 = np.c_[f, f[:, [0]]]
        f = np.c_[np.ones((f.shape[0], 1)), f]

    with pytest.raises(ValueError, match="portfolios must not contain"):
        TradedFactorModel(p, data.factors)
    with pytest.raises(ValueError, match="portfolios must not contain"):
        TradedFactorModel(p2, data.factors)
    with pytest.raises(ValueError, match="The number of observations"):
        TradedFactorModel(p3, data.factors)
    with pytest.raises(ValueError, match="factors must not contain"):
        TradedFactorModel(data.portfolios, f)
    with pytest.raises(ValueError, match="factors must not contain"):
        TradedFactorModel(data.portfolios, f2)
    with pytest.raises(ValueError, match="Model cannot be estimated"):
        TradedFactorModel(p5, f5)
    with pytest.raises(ValueError, match="The number of test portfolio"):
        LinearFactorModel(p4, data.factors)


def test_drop_missing(data):
    p = data.portfolios
    if isinstance(p, pd.DataFrame):
        p.iloc[::33] = np.nan
    else:
        p[::33] = np.nan

    res = TradedFactorModel(p, data.factors).fit()

    p = IVData(p)
    f = IVData(data.factors)
    isnull = p.isnull | f.isnull
    p.drop(isnull)
    f.drop(isnull)

    res2 = TradedFactorModel(p, f).fit()
    assert_equal(np.asarray(res.params), np.asarray(res2.params))


def test_unknown_kernel(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    with pytest.raises(ValueError, match="Unknown weight"):
        mod.fit(cov_type="unknown")
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    with pytest.raises(ValueError, match="Unknown weight"):
        mod.fit(cov_type="unknown")


def test_all_missing():
    p = np.nan * np.ones((1000, 10))
    f = np.nan * np.ones((1000, 3))
    with pytest.raises(ValueError, match="All observations contain missing data"):
        TradedFactorModel(p, f)


def test_repr(data):
    mod = LinearFactorModelGMM(data.portfolios, data.factors)
    assert "LinearFactorModelGMM" in mod.__repr__()
    assert str(data.portfolios.shape[1]) + " test portfolios" in mod.__repr__()
    assert str(data.factors.shape[1]) + " factors" in mod.__repr__()
    mod = LinearFactorModel(data.portfolios, data.factors, risk_free=True)
    assert "LinearFactorModel" in mod.__repr__()
    assert "Estimated risk-free" in mod.__repr__()
    assert "True" in mod.__repr__()
    mod = TradedFactorModel(data.portfolios, data.factors)
    assert "TradedFactorModel" in mod.__repr__()
    assert str(hex(id(mod))) in mod.__repr__()
