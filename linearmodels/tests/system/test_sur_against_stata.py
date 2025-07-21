import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.shared.utility import AttrDict
from linearmodels.system import SUR
from linearmodels.tests.system.results.generate_data import (
    basic_data,
    common_data,
    missing_data,
)
from linearmodels.tests.system.results.parse_stata_results import stata_results

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)


@pytest.fixture(scope="module", params=list(stata_results.keys()))
def model_data(request) -> AttrDict:
    key = request.param
    dgp, model_type = key.split("-")
    if dgp == "basic":
        data = basic_data
    elif dgp == "common":
        data = common_data
        for i, data_key in enumerate(data):
            if i == 0:
                exog = data[data_key]["exog"]
            else:
                data[data_key]["exog"] = exog
    else:  # dgp == 'missing'
        data = missing_data
    cov_kwds: dict[str, str | bool] = {"cov_type": "unadjusted"}
    if model_type == "ss":
        cov_kwds["debiased"] = True
    stata_result = stata_results[key]
    rekeyed_data = {}
    for data_key in data:
        temp = data[data_key]
        new_key = temp["dependent"].columns[0]
        rekeyed_data[new_key] = temp
    constraint = None
    if model_type == "constrained":
        cols = []
        widths = []
        for new_key in rekeyed_data:
            exog = rekeyed_data[new_key]["exog"]
            cols.extend([new_key + "_" + col for col in exog.columns])
            widths.append(exog.shape[1])
        r = pd.DataFrame(columns=cols, index=["r0", "r1"], dtype=np.float64)
        r.iloc[:, :] = 0.0
        r.iloc[:, 0] = -1.0
        r.iloc[0, widths[0]] = 1.0
        r.iloc[1, widths[0] + widths[1]] = 1.0
        constraint = r

    mod = SUR(rekeyed_data)
    if constraint is not None:
        mod.add_constraints(constraint)

    if model_type != "ss":
        res = mod.fit(cov_type="unadjusted")
    else:
        res = mod.fit(cov_type="unadjusted", debiased=True)

    return AttrDict(
        data=rekeyed_data,
        cov_kwds=cov_kwds,
        model_type=model_type,
        stata_result=stata_result,
        key=key,
        constraint=constraint,
        mod=mod,
        res=res,
    )


def test_params(model_data: AttrDict) -> None:
    res = model_data.res
    stata_params = model_data.stata_result.params
    assert_allclose(res.params, stata_params.param)


def test_cov(model_data: AttrDict) -> None:
    res = model_data.res
    stata_cov = model_data.stata_result.variance
    sigma_stata = np.diag(stata_cov)[:, None]
    corr_stata = stata_cov.values / np.sqrt(sigma_stata @ sigma_stata.T)
    sigma = np.diag(res.cov)[:, None]
    corr = res.cov.values / np.sqrt(sigma @ sigma.T)
    assert_allclose(sigma, sigma_stata)
    assert_allclose(corr, corr_stata, rtol=1, atol=1e-6)


def test_tstats(model_data: AttrDict) -> None:
    res = model_data.res
    stata_params = model_data.stata_result.params
    assert_allclose(res.tstats, stata_params.tstat)


def test_pvals(model_data: AttrDict) -> None:
    res = model_data.res
    stata_params = model_data.stata_result.params
    assert_allclose(res.pvalues, stata_params.pval, atol=1e-6)


def test_sigma(model_data: AttrDict) -> None:
    res = model_data.res
    stata_sigma = model_data.stata_result.sigma
    assert_allclose(res.sigma, stata_sigma)


def test_f_stat(model_data: AttrDict) -> None:
    res = model_data.res
    stata_stats = model_data.stata_result.stats
    for i, key in enumerate(res.equations):
        eq = res.equations[key]
        stat = eq.f_statistic.stat
        stata_stat = stata_stats.loc[f"F_{i + 1}"].squeeze()
        if np.isnan(stata_stat):
            stata_stat = stata_stats.loc[f"chi2_{i + 1}"].squeeze()
        assert_allclose(stat, stata_stat)
        pval = eq.f_statistic.pval
        stata_pval = stata_stats.loc[f"p_{i + 1}"]
        assert_allclose(pval, stata_pval, atol=1e-6)


def test_r2(model_data: AttrDict) -> None:
    res = model_data.res
    stata_stats = model_data.stata_result.stats
    for i, key in enumerate(res.equations):
        eq = res.equations[key]
        stat = eq.rsquared
        stata_stat = stata_stats.loc[f"r2_{i + 1}"].squeeze()
        assert_allclose(stat, stata_stat)


def test_sum_of_squares(model_data: AttrDict) -> None:
    res = model_data.res
    stata_stats = model_data.stata_result.stats
    for i, key in enumerate(res.equations):
        eq = res.equations[key]
        stat = eq.resid_ss
        stata_stat = stata_stats.loc[f"rss_{i + 1}"].squeeze()
        assert_allclose(stat, stata_stat)
        stata_stat = stata_stats.loc[f"mss_{i + 1}"].squeeze()
        stat = eq.model_ss
        assert_allclose(stat, stata_stat)


def test_df_model(model_data: AttrDict) -> None:
    res = model_data.res
    stata_stats = model_data.stata_result.stats
    for i, key in enumerate(res.equations):
        eq = res.equations[key]
        stat = eq.df_model
        stata_stat = stata_stats.loc[f"df_m{i + 1}"].squeeze()
        assert_allclose(stat, stata_stat + 1)
