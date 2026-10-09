"""
Tests of the centered J statistic, IVGMMResults.centered_j_stat

The reference values in results/centered-j-reference.csv were produced by
results/centered-j-reference.R, a base R implementation of the definition in
Hall (2000) and Hansen and Lee (2021, Theorem 1 and Example 1). Before it
writes the reference values the script checks that it reproduces results from
Stata that are used elsewhere in the tests: ivregress gmm on the housing and
simulated data, including the Hansen J statistic of overidentified models for
the robust, unadjusted, clustered and kernel weight matrices with and without
centering. It also checks the properties of the statistic that follow from
theory in every scenario where they apply. Stata does not have a centered J
statistic, so the values of the statistic are not from Stata.

The values were also checked against the R packages gmm and AER by
results/centered-j-package-check.R, which is not part of the tests. At the
estimates in the reference the statistics are identical to numerical precision
for all of the models that the packages can estimate.

The scenarios include misspecified models, where the J statistic that uses an
uncentered covariance is bounded by the sample size and the centered statistic
is not, and every estimator of the covariance of the moment conditions with and
without centering and small-sample adjustments, weights, two-step, iterated and
continuously updating GMM.

References
----------
Hall, A. R. (2000). Covariance matrix estimation and the power of the
overidentifying restrictions test. Econometrica, 68(6), 1517-1528.

Hansen, B. E. and Lee, S. (2021). Inference for iterated GMM under
misspecification. Econometrica, 89(3), 1419-1447.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from linearmodels.iv import IVGMM, IVGMMCUE
from linearmodels.iv.gmm import KernelWeightMatrix
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic

CWD = os.path.split(os.path.abspath(__file__))[0]
RESULTS = os.path.join(CWD, "results")

SIMULATED = pd.read_stata(os.path.join(RESULTS, "simulated-data.dta"))
SIMULATED["const"] = 1.0
# Same clusters as the R script, as many as there are moments in some models
SIMULATED["block6"] = np.arange(SIMULATED.shape[0]) // 100
HOUSING = pd.read_csv(os.path.join(RESULTS, "housing.csv"), index_col=0)
HOUSING["const"] = 1.0
# Same names as the columns created by model.matrix in the R script
REGIONS = pd.get_dummies(HOUSING.region, drop_first=True).add_prefix("region")
HOUSING = pd.concat([HOUSING, REGIONS.astype(float)], axis=1)
# The relationship between y and x is misspecified, see the generating script
MISSPECIFIED = pd.read_csv(os.path.join(RESULTS, "misspecified-data.csv"))
MISSPECIFIED["const"] = 1.0
DATASETS = {"sim": SIMULATED, "housing": HOUSING, "mis": MISSPECIFIED}

REFERENCE = pd.read_csv(
    os.path.join(RESULTS, "centered-j-reference.csv"),
    keep_default_na=False,
    na_values={"centered_j": ["NA"]},
)
RTOL = {"two_step": 1e-6, "iterated": 1e-6, "cue": 1e-6}
# The CUE objective is flat near its minimum in some scenarios, where the
# estimates are not as precise as the statistics
RTOL_PARAMS = {"two_step": 1e-8, "iterated": 1e-6, "cue": 1e-4}
# The iterated estimator converges to the same estimates as in R, which are
# found with a tolerance of 1e-13. A tolerance of 1e-14 here is reached within
# 50 iterations in all scenarios, whereas 1e-20 requires 1500 in one
ITERATED_TOL = 1e-14
# The default optimizer of IVGMMCUE, BFGS with numerical derivatives, does not
# find the minimum in some of the scenarios, e.g., it stops at J=1.66 in
# sim_e2_i4 where the minimum is 1.29 whatever the tolerance. These settings
# find the minima that R finds
CUE_OPTIONS = {
    "method": "Nelder-Mead",
    "options": {"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000, "maxfev": 40000},
}


def build_model(row, model=IVGMM):
    """Create the model described by a row of the reference table"""
    data = DATASETS[row.dataset]
    weights = data.weights if row.weighted else None
    # The way that the covariance of the moment conditions is estimated
    config = {
        "weight_type": row.weight_type,
        "center": bool(row.center),
        "debiased": bool(row.debiased),
    }
    if row.weight_type == "kernel":
        config.update({"kernel": "bartlett", "bandwidth": int(row.bandwidth)})
    elif row.weight_type == "clustered":
        config["clusters"] = pd.factorize(data[row.clusters])[0]
    return model(
        data[row.dependent],
        data[row.exog.split()],
        data[row.endog.split()],
        data[row.instruments.split()],
        weights=weights,
        **config,
    )


def fit_model(row):
    """Estimate the model of a row of the reference table"""
    if row.method == "cue":
        return build_model(row, IVGMMCUE).fit(opt_options=CUE_OPTIONS)
    if row.method == "iterated":
        res = build_model(row).fit(iter_limit=1000, tol=ITERATED_TOL)
        assert res.iterations < 1000
        return res
    return build_model(row).fit()


def reference_rows():
    return [
        pytest.param(row, id=f"{row.id}-{row.method}")
        for row in REFERENCE.itertuples(index=False)
    ]


@pytest.mark.parametrize("row", reference_rows())
def test_centered_j_against_r(row):
    res = fit_model(row)
    rtol = RTOL[row.method]
    # The estimates, so that the statistics are those of the same estimator
    params = np.array(row.params.split(), dtype=float)
    assert_allclose(np.asarray(res.params), params, rtol=RTOL_PARAMS[row.method])
    # The J statistic is the one of the same model and estimator
    assert_allclose(res.j_stat.stat, row.j, rtol=rtol)
    assert res.j_stat.df == row.df
    centered = res.centered_j_stat
    if np.isnan(row.centered_j):
        assert isinstance(centered, InvalidTestStatistic)
        assert np.isnan(centered.stat)
        assert np.isnan(centered.pval)
        return
    assert type(centered) is WaldTestStatistic
    assert centered.df == row.df
    assert_allclose(centered.stat, row.centered_j, rtol=rtol)
    assert_allclose(
        centered.pval, stats.chi2.sf(row.centered_j, row.df), rtol=max(rtol, 1e-5)
    )
    assert centered.dist_name == f"chi2({row.df})"


def test_reference_covers_the_cases_that_matter():
    # Guard against the table losing the cases that distinguish the statistics
    assert set(REFERENCE.method) == {"two_step", "iterated", "cue"}
    assert set(REFERENCE.weight_type) == {"robust", "unadjusted", "kernel", "clustered"}
    assert set(REFERENCE.dataset) == {"sim", "housing", "mis"}
    for column in ("weighted", "center", "debiased"):
        assert REFERENCE[column].any()
        assert not REFERENCE[column].all()
    defined = REFERENCE.dropna(subset=["centered_j"])
    # Strongly different, in both directions, and strongly rejecting
    assert (defined.centered_j > 1.5 * defined.j).sum() >= 10
    assert (defined.centered_j < 0.95 * defined.j).sum() >= 3
    pvals = stats.chi2.sf(defined.centered_j, defined.df)
    assert (pvals < 1e-6).sum() >= 10
    assert (pvals > 0.1).sum() >= 3
    # J is bounded by the number of observations, and the new statistic is not
    # Unadjusted models are excluded since their statistic is not
    # affected by the mean of the moment conditions
    iterated = defined[
        (defined.method == "iterated") & (defined.weight_type != "unadjusted")
    ]
    assert (iterated.j < iterated.nobs).all()
    assert REFERENCE.centered_j.isna().sum() == 1


# ----------------------------------------------------------------------------
# Independent calculations that do not use any of the weight matrix classes
# ----------------------------------------------------------------------------


def moments(res, data, dep, exog, endog, instruments, weighted=False):
    """Moment conditions at the estimates, computed from the data"""
    sw = np.ones(data.shape[0])
    if weighted:
        sw = np.sqrt(data.weights / data.weights.mean()).to_numpy()
    y = data[dep].to_numpy() * sw
    x = data[exog + endog].to_numpy() * sw[:, None]
    z = data[exog + instruments].to_numpy() * sw[:, None]
    e = y - x @ np.asarray(res.params)
    return z * e[:, None]


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("iter_limit", [1, 2, 3, 100])
def test_centered_j_matches_covariance_of_the_moments(iter_limit, weighted):
    data = MISSPECIFIED
    spec = ("y", ["const"], ["x"], ["z1", "z2", "zb"])
    weights = data.weights if weighted else None
    res = IVGMM(data.y, data[spec[1]], data[spec[2]], data[spec[3]], weights=weights)
    res = res.fit(iter_limit=iter_limit)
    g = moments(res, data, *spec, weighted=weighted)
    gbar = g.mean(0)
    # np.cov centers the moments and does not adjust for the number of terms
    s_c = np.cov(g, rowvar=False, bias=True)
    expected = g.shape[0] * gbar @ np.linalg.solve(s_c, gbar)
    assert_allclose(res.centered_j_stat.stat, expected, rtol=1e-10)


@pytest.mark.parametrize("weighted", [False, True])
def test_centered_j_cluster_formula(weighted):
    data = MISSPECIFIED
    spec = ("y", ["const"], ["x"], ["z1", "z2", "zb"])
    weights = data.weights if weighted else None
    clusters = data.group.to_numpy()
    mod = IVGMM(
        data.y,
        data[spec[1]],
        data[spec[2]],
        data[spec[3]],
        weights=weights,
        weight_type="clustered",
        clusters=clusters,
    )
    res = mod.fit()
    g = moments(res, data, *spec, weighted=weighted)
    gbar = g.mean(0)
    sums = pd.DataFrame(g - gbar).groupby(clusters).sum().to_numpy()
    s_c = sums.T @ sums / g.shape[0]
    expected = g.shape[0] * gbar @ np.linalg.solve(s_c, gbar)
    assert_allclose(res.centered_j_stat.stat, expected, rtol=1e-10)
    # and this differs from using the heteroskedasticity robust estimator
    robust = IVGMM(data.y, data[spec[1]], data[spec[2]], data[spec[3]]).fit()
    assert abs(res.centered_j_stat.stat - robust.centered_j_stat.stat) > 1


@pytest.mark.parametrize("weight_type", ["robust", "unadjusted", "kernel", "clustered"])
@pytest.mark.parametrize(
    ("cov_type", "cov_config"),
    [
        ("robust", {}),
        ("unadjusted", {}),
        ("kernel", {"kernel": "parzen", "bandwidth": 2}),
        ("clustered", {"clusters": "group"}),
    ],
)
def test_centered_j_uses_the_weight_estimator_not_the_covariance_estimator(
    weight_type, cov_type, cov_config
):
    # The statistic tests the moment conditions with the covariance estimator
    # of the weight matrix, as j_stat does, and not with the estimator used for
    # the covariance of the parameters
    data = MISSPECIFIED
    config = {"weight_type": weight_type}
    if weight_type == "kernel":
        config.update({"kernel": "bartlett", "bandwidth": 3})
    elif weight_type == "clustered":
        config["clusters"] = data.group.to_numpy()
    cov_config = {
        key: data[value].to_numpy() if key == "clusters" else value
        for key, value in cov_config.items()
    }
    mod = IVGMM(data.y, data.const, data.x, data[["z1", "z2", "zb"]], **config)
    base = mod.fit()
    res = mod.fit(cov_type=cov_type, **cov_config)
    assert_allclose(res.centered_j_stat.stat, base.centered_j_stat.stat, rtol=1e-12)
    assert_allclose(res.j_stat.stat, base.j_stat.stat, rtol=1e-12)


def test_centered_j_does_not_change_the_model():
    data = MISSPECIFIED
    mod = IVGMM(
        data.y,
        data.const,
        data.x,
        data[["z1", "z2", "zb"]],
        weight_type="kernel",
        bandwidth=3,
    )
    first = mod.fit()
    config = first.weight_config
    assert config["center"] is False
    second = mod.fit()
    assert second.weight_config == config
    assert second.weight_config["bandwidth"] == 3
    assert_allclose(second.centered_j_stat.stat, first.centered_j_stat.stat)
    assert_allclose(second.j_stat.stat, first.j_stat.stat)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_centered_j_keeps_the_options_of_the_weight_estimator(kernel):
    # A bandwidth that is chosen using the data, optimal_bw, is chosen for the
    # centered estimator too. This is not reported by the configuration of
    # the estimator, which has the bandwidth that it selected
    data = SIMULATED
    dep, exog = "y_kernel", ["const", "x3"]
    endog, instr = ["x1", "x2"], ["z1", "z2", "x4", "x5"]
    results = {}
    for optimal_bw in (True, False):
        mod = IVGMM(
            data[dep],
            data[exog],
            data[endog],
            data[instr],
            weight_type="kernel",
            kernel=kernel,
            bandwidth=None,
            optimal_bw=optimal_bw,
        )
        results[optimal_bw] = res = mod.fit()
        g = moments(res, data, dep, exog, endog, instr)
        estimator = KernelWeightMatrix(
            kernel=kernel, bandwidth=None, center=True, optimal_bw=optimal_bw
        )
        x = data[exog + endog].to_numpy()
        z = data[exog + instr].to_numpy()
        eps = (data[dep].to_numpy() - x @ np.asarray(res.params))[:, None]
        s_c = estimator.weight_matrix(x, z, eps)
        gbar = g.mean(0)
        expected = g.shape[0] * gbar @ np.linalg.solve(s_c, gbar)
        assert_allclose(res.centered_j_stat.stat, expected, rtol=1e-10)
    # The option is relevant for these data
    chosen, default = results[True], results[False]
    assert abs(chosen.centered_j_stat.stat - default.centered_j_stat.stat) > 1e-3


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("data", [MISSPECIFIED, SIMULATED], ids=["mis", "sim"])
def test_iterated_gmm_identity(data, weighted):
    # With iterated GMM and the robust estimator, S = S_c + gbar gbar', and
    # so, by the Sherman-Morrison formula, J_c = J / (1 - J / n). J can then
    # not exceed n, and J_c is not bounded
    if data is MISSPECIFIED:
        spec = ("y", ["const"], ["x"], ["z1", "z2", "zb"])
    else:
        spec = ("y_robust", ["const", "x3"], ["x1", "x2"], ["z1", "z2", "x4", "x5"])
    weights = data.weights if weighted else None
    dep, exog, endog, instr = spec
    mod = IVGMM(data[dep], data[exog], data[endog], data[instr], weights=weights)
    res = mod.fit(iter_limit=1000, tol=ITERATED_TOL)
    nobs = data.shape[0]
    j = res.j_stat.stat
    assert j < nobs
    assert res.centered_j_stat.stat > j
    assert_allclose(res.centered_j_stat.stat, j / (1 - j / nobs), rtol=1e-8)


def test_both_statistics_are_proportional_to_the_sample_size():
    # With a fixed violation of a moment condition both J / n and J_c / n have
    # limits, which are a / (1 + a) and a, and so both statistics diverge at
    # rate n. J < n always, but this is a bound on the statistic and not on how
    # it grows with the sample size
    rng = np.random.default_rng(20260628)
    nobs_values = (1000, 4000, 16000)
    ratios = []
    for nobs in nobs_values:
        z = rng.standard_normal((nobs, 3))
        v = rng.standard_normal(nobs)
        u = 0.5 * v + rng.standard_normal(nobs)
        x = z @ np.array([0.8, 0.5, 0.4]) + v
        y = 1 + x + 0.5 * z[:, 2] + u
        res = IVGMM(y, np.ones((nobs, 1)), x, z).fit(iter_limit=1000, tol=ITERATED_TOL)
        j, centered = res.j_stat.stat, res.centered_j_stat.stat
        assert j < nobs
        ratios.append((j / nobs, centered / nobs))
    ratios = np.array(ratios)
    # Not changing with the sample size, within sampling error
    assert np.all(ratios.max(0) < 1.3 * ratios.min(0))
    # a / (1 + a) and a
    assert_allclose(ratios[:, 0], ratios[:, 1] / (1 + ratios[:, 1]), rtol=1e-6)
    # which is a large statistic in the largest sample
    assert ratios[-1, 1] * nobs_values[-1] > 100


@pytest.mark.parametrize("debiased", [False, True])
def test_unadjusted_is_not_changed_by_centering(debiased):
    # The estimator of the covariance is already centered
    data = SIMULATED
    mod = IVGMM(
        data.y_unadjusted,
        data[["const", "x3"]],
        data[["x1", "x2"]],
        data[["z1", "z2", "x4", "x5"]],
        weight_type="unadjusted",
        debiased=debiased,
    )
    res = mod.fit(iter_limit=100)
    assert_allclose(res.centered_j_stat.stat, res.j_stat.stat, rtol=1e-8)


@pytest.mark.parametrize("center", [True, False])
def test_cue(center):
    data = MISSPECIFIED
    mod = IVGMMCUE(data.y, data.const, data.x, data[["z1", "z2", "zb"]], center=center)
    res = mod.fit()
    j = res.j_stat.stat
    nobs = data.shape[0]
    if center:
        # The weight matrix of the estimator is the one of the statistic
        assert_allclose(res.centered_j_stat.stat, j, rtol=1e-10)
    else:
        # The weight and the moments are evaluated at the same parameters,
        # and so this holds exactly and not only at a minimum
        assert_allclose(res.centered_j_stat.stat, j / (1 - j / nobs), rtol=1e-10)
    # The default of the CUE estimator is to center
    default = IVGMMCUE(data.y, data.const, data.x, data[["z1", "z2", "zb"]]).fit()
    assert_allclose(default.centered_j_stat.stat, default.j_stat.stat, rtol=1e-10)


def test_result_types():
    data = MISSPECIFIED
    exog, endog, instr = data.const, data.x, data[["z1", "z2", "zb"]]
    for res in (
        IVGMM(data.y, exog, endog, instr),
        IVGMMCUE(data.y, exog, endog, instr),
    ):
        centered = res.fit().centered_j_stat
        assert type(centered) is WaldTestStatistic
        assert centered.df == 2
        assert 0 <= centered.pval <= 1
        assert centered.dist_name == "chi2(2)"
        assert "Expected moment conditions are equal to 0" in str(centered)
        assert "Centered J-test" in str(centered)
        assert set(centered.critical_values) == {"10%", "5%", "1%"}


def test_exactly_identified():
    # As for j_stat, there is nothing to test. The statistic is 0 up to
    # numerical error and has no degrees of freedom
    data = MISSPECIFIED
    res = IVGMM(data.y, data.const, data.x, data[["z1"]]).fit()
    assert res.centered_j_stat.df == res.j_stat.df == 0
    assert_allclose(res.centered_j_stat.stat, 0, atol=1e-8)
    assert_allclose(res.j_stat.stat, 0, atol=1e-8)
    assert "chi2(0)" in str(res.centered_j_stat)
    assert "R-squared" in str(res.summary)


@pytest.mark.parametrize("nclusters", [4, 5, 20])
def test_too_few_clusters(nclusters):
    # The centered cluster sums add to zero, so that the covariance has at
    # most nclusters - 1 non-zero eigenvalues and the model has 4 moments
    data = MISSPECIFIED
    clusters = np.arange(data.shape[0]) % nclusters
    mod = IVGMM(
        data.y,
        data.const,
        data.x,
        data[["z1", "z2", "zb"]],
        weight_type="clustered",
        clusters=clusters,
    )
    res = mod.fit()
    centered = res.centered_j_stat
    assert np.isfinite(res.j_stat.stat)
    if nclusters > 4:
        assert type(centered) is WaldTestStatistic
        assert np.isfinite(centered.stat)
        return
    assert isinstance(centered, InvalidTestStatistic)
    assert np.isnan(centered.stat)
    assert np.isnan(centered.pval)
    assert "singular" in str(centered)
    assert "Centered J-test" in str(centered)


def test_summary_is_not_changed():
    # The statistic is an attribute and does not replace anything in the
    # header of the summary, which has the R-squared and F-statistic of the
    # model as for other IV models
    data = MISSPECIFIED
    res = IVGMM(data.y, data.const, data.x, data[["z1", "z2", "zb"]]).fit()
    summary = str(res.summary)
    for label in ("R-squared:", "Adj. R-squared:", "F-statistic:", "P-value (F-stat)"):
        assert label in summary
    assert "Centered" not in summary
    assert "Endogenous: x" in summary
    assert "Instruments: z1, z2, zb" in summary
    assert max(len(line) for line in summary.split("\n")) <= 78


@pytest.mark.slow
def test_size_and_power_of_the_test():
    # The statistic has the right size, and is not less powerful than J
    rng = np.random.default_rng(20260628)
    nobs, reps = 250, 1000
    reject = {violation: np.zeros((reps, 2), dtype=bool) for violation in (0.0, 0.3)}
    critical = stats.chi2.ppf(0.95, 2)
    for rep in range(reps):
        z = rng.standard_normal((nobs, 3))
        v = rng.standard_normal(nobs)
        u = (1 + 0.5 * np.abs(z[:, 0])) * (0.5 * v + rng.standard_normal(nobs))
        x = z @ np.array([0.6, 0.5, 0.4]) + v
        for violation, rejected in reject.items():
            y = 1 + x + violation * z[:, 2] + u
            res = IVGMM(y, np.ones((nobs, 1)), x, z).fit(iter_limit=100)
            rejected[rep] = [
                res.j_stat.stat > critical,
                res.centered_j_stat.stat > critical,
            ]
    size = reject[0.0].mean(0)
    power = reject[0.3].mean(0)
    # The standard error of a rejection frequency is about 0.007
    assert 0.03 < size[1] < 0.075
    assert 0.03 < size[0] < 0.075
    assert power[1] >= power[0] - 0.01
    assert power[1] > 0.5
