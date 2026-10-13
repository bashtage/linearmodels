"""
Tests of the J and C statistics of IVGMM and IVGMMCUE with a singular covariance

The weight matrix of a GMM estimator is the inverse of an estimate of the
covariance of the moment conditions. That estimate is singular if there are too
few clusters. The covariance of K moment conditions that is estimated from G
clusters is a sum of G outer products of K-vectors, and so it has rank at most
G. The cluster sums of centered moment conditions add to zero, and so its rank
is at most G - 1 if the moment conditions are centered. The covariance of a
kernel or robust estimator has rank at most the number of observations, and one
less if the moment conditions are centered.

The J statistic and the C statistic are functions of the inverse. When the
covariance is singular the inverse is numerical noise, and the statistics are
very large, negative or arbitrary numbers. They have to be invalid, as the
centered J statistic is. All of the expected values below are computed from the
definitions, without any of the weight matrix classes.

The objective function of IVGMMCUE is a function of the inverse of the
covariance, and so it is not defined either. IVGMMCUE.fit has to raise a
ValueError that says why, before it optimizes the function. So does IVGMM.fit,
since the estimates of two-step GMM are numerical noise when the weight matrix
is the inverse of a singular covariance. Whether NumPy raises a LinAlgError when
it inverts a covariance that is singular depends on rounding, and so on the
platform: it does so if the matrix is exactly singular, and otherwise returns
the inverse of the noise. Both fits raise the same ValueError on every platform,
and a model that cannot be estimated has no statistics to test. The C statistic
estimates another model, which has more moment conditions, and is invalid if
that model cannot be estimated.

The objective function of IVGMMCUE is also constant, and not only undefined, if
the moment conditions are not centered and there are as many clusters as moment
conditions. IVGMMCUE.fit raises a ValueError then too, so that there have to be
more clusters than moment conditions with every value of center.
"""

import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE
import linearmodels.iv.model
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def one_endog_data(nobs=120):
    """2 exog, 1 endog and 2 instruments (4 moments), the model of the bug report"""
    rng = np.random.default_rng(3)
    z = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["z0", "z1"])
    x = pd.DataFrame({"x": z.sum(axis=1) + rng.standard_normal(nobs)})
    exog = pd.DataFrame({"const": 1.0, "w": rng.standard_normal(nobs)})
    y = pd.Series(1 + 0.5 * x.x + rng.standard_normal(nobs), name="y")
    return y, exog, x, z


def two_endog_data(nobs=120):
    """1 exog, 2 endog and 3 instruments (4 moments), so that variables can be tested"""
    rng = np.random.default_rng(5)
    z = pd.DataFrame(rng.standard_normal((nobs, 3)), columns=["z0", "z1", "z2"])
    x = pd.DataFrame(
        {
            "x1": z.z0 + z.z1 + rng.standard_normal(nobs),
            "x2": z.z1 - z.z2 + rng.standard_normal(nobs),
        }
    )
    exog = pd.DataFrame({"const": np.ones(nobs)})
    y = pd.Series(1 + 0.5 * x.x1 - 0.3 * x.x2 + rng.standard_normal(nobs), name="y")
    return y, exog, x, z


def small_sample(nobs):
    """A constant, 1 endogenous variable and 3 instruments: 4 moments"""
    rng = np.random.default_rng(11)
    z = pd.DataFrame(rng.standard_normal((nobs, 3)), columns=["z0", "z1", "z2"])
    x = pd.DataFrame({"x": z.sum(axis=1) + rng.standard_normal(nobs)})
    exog = pd.DataFrame({"const": np.ones(nobs)})
    y = pd.Series(1 + 0.5 * x.x + rng.standard_normal(nobs), name="y")
    return y, exog, x, z


ONE_ENDOG = one_endog_data()
TWO_ENDOG = two_endog_data()


# ----------------------------------------------------------------------------
# Definitions
# ----------------------------------------------------------------------------


def gmm_params(x, y, z, w):
    """GMM estimator that minimizes the quadratic form in the moments"""
    xpz = x.T @ z
    return np.linalg.solve(xpz @ w @ xpz.T, xpz @ w @ (z.T @ y))


def j_value(x, y, z, w, params):
    gbar = (z * (y - x @ params)[:, None]).mean(0)
    return y.shape[0] * gbar @ w @ gbar


def clustered_cov(clusters, center):
    """Sum of the outer products of the cluster sums of the moment conditions"""

    def cov(g):
        if center:
            g = g - g.mean(0)
        sums = np.array([g[clusters == c].sum(0) for c in np.unique(clusters)])
        return sums.T @ sums / g.shape[0]

    return cov


def bartlett_cov(bandwidth, center):
    """Heteroskedasticity and autocorrelation robust, with a Bartlett kernel"""

    def cov(g):
        if center:
            g = g - g.mean(0)
        s = g.T @ g
        for lag in range(1, bandwidth + 1):
            op = g[lag:].T @ g[:-lag]
            s += (1 - lag / (bandwidth + 1)) * (op + op.T)
        return s / g.shape[0]

    return cov


def reference_j(x, y, z, cov):
    """J statistic from two step GMM, using the covariance of the first step"""
    first = gmm_params(x, y, z, np.linalg.inv(z.T @ z / y.shape[0]))
    w = np.linalg.inv(cov(z * (y - x @ first)[:, None]))
    return j_value(x, y, z, w, gmm_params(x, y, z, w))


def reference_c(x, y, z_e, original, cov):
    """
    C statistic, J_e - J_c

    z_e contains the moments of the model that treats the tested variables as
    exogenous, and original are the columns of the moments of the model.
    """
    first = gmm_params(x, y, z_e, np.linalg.inv(z_e.T @ z_e / y.shape[0]))
    s_e = cov(z_e * (y - x @ first)[:, None])
    w_e = np.linalg.inv(s_e)
    j_e = j_value(x, y, z_e, w_e, gmm_params(x, y, z_e, w_e))
    z = z_e[:, original]
    w_c = np.linalg.inv(s_e[np.ix_(original, original)])
    return j_e - j_value(x, y, z, w_c, gmm_params(x, y, z, w_c))


def arrays(data, tested=None, sqrt_weights=None):
    """x, y, the model's moments and the moments with the tested variables added"""
    y, exog, endog, instr = (np.asarray(v, dtype=float) for v in data)
    y = y.reshape(-1)
    names = list(data[2].columns)
    tested = names if tested is None else tested
    index = [names.index(name) for name in tested]
    x = np.column_stack([exog, endog])
    z = np.column_stack([exog, instr])
    z_e = np.column_stack([exog, endog[:, index], instr])
    ninstr = instr.shape[1]
    original = [*range(exog.shape[1]), *range(z_e.shape[1] - ninstr, z_e.shape[1])]
    if sqrt_weights is not None:
        x, z, z_e = (a * sqrt_weights[:, None] for a in (x, z, z_e))
        y = y * sqrt_weights
    return x, y, z, z_e, original


def nonsingular(nmoments, nclusters, center):
    """Rank condition of the covariance that is estimated from clusters"""
    return nclusters - center >= nmoments


def fit(data, nclusters, center, model=IVGMM, weights=None, **kwargs):
    """Fit the model of the data with the clusters arange(nobs) % nclusters"""
    y, exog, endog, instr = data
    clusters = np.arange(y.shape[0]) % nclusters
    mod = model(
        y,
        exog,
        endog,
        instr,
        weights=weights,
        weight_type="clustered",
        clusters=clusters,
        center=center,
    )
    return mod.fit(cov_type="clustered", clusters=clusters, **kwargs)


def check_too_few_clusters(data, nclusters, center, **kwargs):
    """The covariance of the moments of the model is singular and fit raises"""
    nmoments = data[1].shape[1] + data[3].shape[1]
    assert not nonsingular(nmoments, nclusters, center)
    with pytest.raises(ValueError, match="too few clusters") as exc_info:
        fit(data, nclusters, center, **kwargs)
    # LinAlgError is a ValueError in recent versions of NumPy
    assert exc_info.type is ValueError
    message = str(exc_info.value)
    assert f"({nclusters} clusters)" in message
    assert f"{nmoments} moment conditions" in message
    assert "singular" in message
    assert "Use more clusters than moment conditions" in message


def check_invalid(stat, name):
    assert type(stat) is InvalidTestStatistic
    assert np.isnan(stat.stat)
    assert np.isnan(stat.pval)
    assert stat.critical_values is None
    text = str(stat)
    assert text.startswith("Invalid test statistic")
    assert "singular" in text
    assert name in text


def check_valid(stat, expected, df):
    assert type(stat) is WaldTestStatistic
    assert stat.df == df
    assert np.isfinite(stat.stat)
    assert_allclose(stat.stat, expected, rtol=1e-7, atol=1e-10)
    assert_allclose(stat.pval, stats.chi2.sf(expected, df), rtol=1e-6, atol=1e-12)


J_NAME = "J-test of overidentifying restrictions"

# ----------------------------------------------------------------------------
# The model that the problem was reported with
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("nclusters", [3, 4])
def test_c_stat_with_too_few_clusters_is_invalid(nclusters):
    # With 3 and 4 clusters the statistic was -3.7e15 and -7.4e16. The model has 4
    # moments and the model for the statistic has 5, and so the covariance of the
    # moments that it estimates is singular unless there are at least 5 clusters
    if nclusters == 3:
        # The covariance of the model's own 4 moments is singular too, so that
        # the model is not estimated
        check_too_few_clusters(ONE_ENDOG, nclusters, center=False)
        return
    check_invalid(fit(ONE_ENDOG, nclusters, center=False).c_stat(), "C-statistic")


def test_c_stat_with_enough_clusters_is_unchanged():
    res = fit(ONE_ENDOG, 5, center=False)
    x, y, _, z_e, original = arrays(ONE_ENDOG)
    cov = clustered_cov(np.arange(120) % 5, center=False)
    check_valid(res.c_stat(), reference_c(x, y, z_e, original, cov), df=1)
    # The value that the statistic has always had when it was valid
    assert_allclose(res.c_stat().stat, 0.8759754606839069, rtol=1e-6)


# ----------------------------------------------------------------------------
# The C statistic
# ----------------------------------------------------------------------------

C_STAT_MODELS = [
    pytest.param(ONE_ENDOG, None, id="one-endog"),
    pytest.param(TWO_ENDOG, None, id="two-endog-all"),
    pytest.param(TWO_ENDOG, ["x1"], id="two-endog-x1"),
    pytest.param(TWO_ENDOG, ["x2"], id="two-endog-x2"),
]


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("nclusters", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize(("data", "tested"), C_STAT_MODELS)
def test_c_stat_few_clusters(data, tested, nclusters, center):
    if not nonsingular(data[1].shape[1] + data[3].shape[1], nclusters, center):
        check_too_few_clusters(data, nclusters, center)
        return
    c_stat = fit(data, nclusters, center).c_stat(tested)
    ntested = data[2].shape[1] if tested is None else len(tested)
    nmoments = data[1].shape[1] + data[3].shape[1] + ntested
    if not nonsingular(nmoments, nclusters, center):
        check_invalid(c_stat, "C-statistic")
        return
    x, y, _, z_e, original = arrays(data, tested)
    cov = clustered_cov(np.arange(y.shape[0]) % nclusters, center)
    check_valid(c_stat, reference_c(x, y, z_e, original, cov), df=ntested)
    assert c_stat.stat >= 0


@pytest.mark.parametrize("nclusters", [5, 6])
def test_c_stat_continuously_updating(nclusters):
    # The covariance is centered unless center=False, so the model with 4 moments
    # can be estimated with 5 clusters but the C statistic needs 6
    res = fit(ONE_ENDOG, nclusters, True, model=IVGMMCUE)
    assert type(res.j_stat) is WaldTestStatistic
    assert np.isfinite(res.j_stat.stat)
    c_stat = res.c_stat()
    if nclusters == 5:
        check_invalid(c_stat, "C-statistic")
        return
    x, y, _, z_e, original = arrays(ONE_ENDOG)
    cov = clustered_cov(np.arange(120) % nclusters, center=True)
    check_valid(c_stat, reference_c(x, y, z_e, original, cov), df=1)


def test_c_stat_when_inverting_the_covariance_raises(monkeypatch):
    # NumPy raises a LinAlgError if a matrix that it inverts is exactly singular,
    # which happened in c_stat for the covariance of 2 centered clusters (it has
    # rank 1) and was not caught. It depends on rounding, so raise it here, for
    # the matrices of the model that treats the endogenous variable as exogenous,
    # which has 5 moments and the model has 4
    res = fit(ONE_ENDOG, 6, center=False)
    inv = linearmodels.iv.model.inv

    def raising_inv(a):
        if a.shape[0] == 5:
            raise LinAlgError("Singular matrix")
        return inv(a)

    with monkeypatch.context() as patch:
        patch.setattr(linearmodels.iv.model, "inv", raising_inv)
        check_invalid(res.c_stat(), "C-statistic")
    assert type(res.c_stat()) is WaldTestStatistic


# ----------------------------------------------------------------------------
# The J statistic
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("nclusters", [2, 3, 4, 5, 6, 8])
def test_j_stat_few_clusters(nclusters, center):
    if not nonsingular(4, nclusters, center):
        # The statistic was between -3e15 and 2e15, or a plausible number in a
        # few cases, and there was a LinAlgError in some
        check_too_few_clusters(ONE_ENDOG, nclusters, center)
        return
    j_stat = fit(ONE_ENDOG, nclusters, center).j_stat
    x, y, z, _, _ = arrays(ONE_ENDOG)
    cov = clustered_cov(np.arange(120) % nclusters, center)
    check_valid(j_stat, reference_j(x, y, z, cov), df=1)
    assert j_stat.stat >= 0


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("nclusters", [2, 3, 4, 5])
def test_j_and_centered_j_stat_are_checked_separately(nclusters, center):
    # The statistic that is not centered is valid with as many clusters as moments,
    # whereas the centered statistic always centers the moments
    if not nonsingular(4, nclusters, center):
        check_too_few_clusters(ONE_ENDOG, nclusters, center)
        return
    res = fit(ONE_ENDOG, nclusters, center)
    assert type(res.j_stat) is WaldTestStatistic
    assert (type(res.centered_j_stat) is WaldTestStatistic) == (nclusters >= 5)


def test_j_stat_without_a_covariance_estimate_is_unchanged():
    # With one step the weight matrix is the inverse of the second moment of the
    # instruments, which is not a covariance estimate that depends on the clusters
    res = fit(ONE_ENDOG, 2, center=False, iter_limit=1)
    x, y, z, _, _ = arrays(ONE_ENDOG)
    w = np.linalg.inv(z.T @ z / 120)
    expected = j_value(x, y, z, w, gmm_params(x, y, z, w))
    check_valid(res.j_stat, expected, df=1)


@pytest.mark.parametrize("iter_limit", [2, 3, 100])
def test_j_stat_iterated(iter_limit):
    check_too_few_clusters(ONE_ENDOG, 3, center=False, iter_limit=iter_limit)
    res = fit(ONE_ENDOG, 20, center=False, iter_limit=iter_limit)
    assert type(res.j_stat) is WaldTestStatistic
    assert np.isfinite(res.j_stat.stat)


def test_summary_and_attributes_with_an_invalid_statistic():
    # The model can be estimated with 4 clusters, but not its C statistic or its
    # centered J statistic
    res = fit(ONE_ENDOG, 4, center=False)
    check_invalid(res.c_stat(), "C-statistic")
    check_invalid(
        res.centered_j_stat, "Centered J-test of overidentifying restrictions"
    )
    assert "Parameter Estimates" in str(res.summary)
    assert np.all(np.isfinite(res.params))
    assert res.weight_type == "clustered"


# ----------------------------------------------------------------------------
# Kernel and robust covariances, which are singular if there are too few observations
# ----------------------------------------------------------------------------


def bandwidth_options(weight_type):
    """
    The kernel weight matrix estimates the bandwidth, which is the default and warns.
    That is tested in test_gmm_kernel_bandwidth.py. optimal_bw=True estimates it
    without the warning.
    """
    return {"optimal_bw": True} if weight_type == "kernel" else {}


@pytest.mark.parametrize("weight_type", ["kernel", "robust"])
@pytest.mark.parametrize("nobs", [4, 5, 6, 8, 20])
def test_small_samples_j_stat(weight_type, nobs):
    # The centered moment conditions of nobs observations have rank at most
    # nobs - 1, and so the covariance of the 4 moments is singular if nobs = 4
    data = small_sample(nobs)
    options = bandwidth_options(weight_type)
    mod = IVGMM(*data, weight_type=weight_type, center=True, **options)
    if nobs == 4:
        with pytest.raises(ValueError, match="too few observations") as exc_info:
            mod.fit()
        assert exc_info.type is ValueError
        assert "4 moment conditions" in str(exc_info.value)
        return
    j_stat = mod.fit().j_stat
    x, y, z, _, _ = arrays(data)
    bandwidth = mod._weight.bandwidth if weight_type == "kernel" else 0
    expected = reference_j(x, y, z, bartlett_cov(bandwidth, center=True))
    check_valid(j_stat, expected, df=2)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("weight_type", ["kernel", "robust"])
@pytest.mark.parametrize("nobs", [5, 6, 8, 20])
def test_small_samples_c_stat(weight_type, nobs, center):
    # The C statistic needs 5 moments, so that nobs = 5 gives a singular
    # covariance if the moments are centered
    data = small_sample(nobs)
    options = {"bandwidth": nobs - 2} if weight_type == "kernel" else {}
    mod = IVGMM(*data, weight_type=weight_type, center=center, **options)
    c_stat = mod.fit().c_stat()
    if nobs == 5 and center:
        check_invalid(c_stat, "C-statistic")
        return
    x, y, _, z_e, original = arrays(data)
    bandwidth = nobs - 2 if weight_type == "kernel" else 0
    expected = reference_c(x, y, z_e, original, bartlett_cov(bandwidth, center))
    check_valid(c_stat, expected, df=1)


# ----------------------------------------------------------------------------
# Weights and the units of the data
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("nclusters", [3, 4, 5, 6, 10])
def test_weights(nclusters):
    rng = np.random.default_rng(1234)
    weights = np.exp(0.5 * rng.standard_normal(120))
    if nclusters == 3:
        check_too_few_clusters(ONE_ENDOG, nclusters, False, weights=weights)
        return
    res = fit(ONE_ENDOG, nclusters, center=False, weights=weights)
    sw = np.sqrt(weights / weights.mean())
    x, y, z, z_e, original = arrays(ONE_ENDOG, sqrt_weights=sw)
    cov = clustered_cov(np.arange(120) % nclusters, center=False)
    check_valid(res.j_stat, reference_j(x, y, z, cov), df=1)
    if nclusters >= 5:
        check_valid(res.c_stat(), reference_c(x, y, z_e, original, cov), df=1)
    else:
        check_invalid(res.c_stat(), "C-statistic")


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1e4, 1e8])
@pytest.mark.parametrize("nclusters", [4, 12])
def test_units_of_the_instruments_do_not_matter(nclusters, scale):
    # The statistics do not depend on the units of the instruments. A covariance
    # with variances that differ by 1e16 must not be mistaken for a singular one,
    # which is why the rank is that of the correlation matrix
    y, exog, endog, instr = ONE_ENDOG
    scaled = (y, exog, endog, instr * [scale, 1.0])
    res = fit(ONE_ENDOG, nclusters, center=False)
    res_scaled = fit(scaled, nclusters, center=False)
    assert type(res_scaled.j_stat) is WaldTestStatistic
    assert_allclose(res_scaled.j_stat.stat, res.j_stat.stat, rtol=1e-5)
    if nclusters == 12:
        assert type(res_scaled.c_stat()) is WaldTestStatistic
        assert_allclose(res_scaled.c_stat().stat, res.c_stat().stat, rtol=1e-5)
    else:
        check_invalid(res_scaled.c_stat(), "C-statistic")


# ----------------------------------------------------------------------------
# IVGMMCUE, which inverts the covariance in the objective that it minimizes
# ----------------------------------------------------------------------------


def cue_model(data, nclusters, **config):
    y, exog, endog, instr = data
    clusters = np.arange(y.shape[0]) % nclusters
    mod = IVGMMCUE(
        y,
        exog,
        endog,
        instr,
        weight_type="clustered",
        clusters=clusters,
        **config,
    )
    return mod, clusters


def cue_criterion(data, params, nclusters, center):
    """The objective of continuously updating GMM at params, from the definition"""
    x, y, z, _, _ = arrays(data)
    g = z * (y - x @ params)[:, None]
    gbar = g.mean(0)
    cov = clustered_cov(np.arange(y.shape[0]) % nclusters, center)(g)
    return y.shape[0] * gbar @ np.linalg.solve(cov, gbar)


# The model has 4 moment conditions. Centered cluster sums add to zero, so that the
# covariance is singular unless there are more clusters than moment conditions. If
# the moment conditions are not centered the covariance is singular with fewer
# clusters than moment conditions, and the objective function is constant with
# as many (see test_cue_uncentered_with_as_many_clusters_as_moments)
CUE_SINGULAR = [
    pytest.param(2, {}, id="default-2"),
    pytest.param(3, {}, id="default-3"),
    pytest.param(4, {}, id="default-4"),
    pytest.param(4, {"center": True}, id="center-4"),
    pytest.param(2, {"center": False}, id="uncentered-2"),
    pytest.param(3, {"center": False}, id="uncentered-3"),
]


@pytest.mark.parametrize(("nclusters", "config"), CUE_SINGULAR)
def test_cue_too_few_clusters(nclusters, config):
    # This was a LinAlgError (Singular matrix) from the objective function, which
    # is a ValueError in recent versions of NumPy and says nothing about the cause
    mod, clusters = cue_model(ONE_ENDOG, nclusters, **config)
    with pytest.raises(ValueError, match="too few clusters") as exc_info:
        mod.fit(cov_type="clustered", clusters=clusters)
    assert exc_info.type is ValueError
    message = str(exc_info.value)
    assert f"{nclusters} clusters" in message
    assert "4 moment conditions" in message
    assert "more clusters" in message
    assert "singular" in message


@pytest.mark.parametrize("debiased", [False, True])
def test_cue_uncentered_with_as_many_clusters_as_moments(debiased):
    # The covariance is not singular, but the objective function is
    # 1'Xi(Xi'Xi)^{-1}Xi'1 = 4 for all parameters, with Xi the matrix of the 4
    # cluster sums of the moment conditions, which is square. It is divided by
    # the small sample adjustment of the clustered covariance if it is debiased.
    # The model fitted without an error, and the estimates were the starting
    # values
    mod, clusters = cue_model(ONE_ENDOG, 4, center=False, debiased=debiased)
    nobs, nvar = 120, 3
    scale = (nobs - 1) / (nobs - nvar) * 4 / 3 if debiased else 1.0
    rng = np.random.default_rng(0)
    for params in rng.standard_normal((5, 3)) * 3 + [1.0, 0.0, 0.5]:
        value = mod.j(params, mod._wx, mod._wy, mod._wz)
        assert_allclose(value, 4.0 / scale, rtol=1e-9)
        if not debiased:
            assert_allclose(cue_criterion(ONE_ENDOG, params, 4, False), 4.0, rtol=1e-9)
    with pytest.raises(ValueError, match="as many clusters") as exc_info:
        mod.fit(cov_type="clustered", clusters=clusters)
    assert exc_info.type is ValueError
    message = str(exc_info.value)
    assert "as many clusters (4 clusters) as there are moment conditions" in message
    assert "4 moment conditions" in message
    assert "not centered" in message
    assert "the same for every value of the parameters" in message
    assert "Use more clusters than moment conditions" in message


def test_cue_uncentered_with_as_many_clusters_as_moments_with_starting_values():
    mod, clusters = cue_model(ONE_ENDOG, 4, center=False)
    starting = pd.Series([1.0, 0.0, 0.5], index=["const", "w", "x"])
    with pytest.raises(ValueError, match="as many clusters"):
        mod.fit(starting=starting, cov_type="clustered", clusters=clusters)


def test_cue_uncentered_with_as_many_clusters_as_moments_is_found_first(monkeypatch):
    def minimize(*args, **kwargs):
        raise AssertionError("the objective is constant, do not optimize it")

    monkeypatch.setattr(linearmodels.iv.model, "minimize", minimize)
    mod, clusters = cue_model(ONE_ENDOG, 4, center=False)
    with pytest.raises(ValueError, match="as many clusters"):
        mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("nclusters", [2, 3, 4, 5, 6, 8])
def test_cue_and_ivgmm_clusters(nclusters, center):
    # IVGMMCUE needs more clusters than the 4 moment conditions, with every
    # value of center. IVGMM needs a nonsingular covariance, which is one cluster
    # less if the moment conditions are not centered
    if nclusters > 4:
        mod, clusters = cue_model(ONE_ENDOG, nclusters, center=center)
        res = mod.fit(cov_type="clustered", clusters=clusters)
        assert np.all(np.isfinite(res.params))
        assert np.all(np.isfinite(res.std_errors))
    else:
        mod, clusters = cue_model(ONE_ENDOG, nclusters, center=center)
        with pytest.raises(ValueError, match="clusters") as exc_info:
            mod.fit(cov_type="clustered", clusters=clusters)
        assert exc_info.type is ValueError
    if nonsingular(4, nclusters, center):
        assert np.all(np.isfinite(fit(ONE_ENDOG, nclusters, center).params))
    else:
        check_too_few_clusters(ONE_ENDOG, nclusters, center)


def test_cue_too_few_clusters_with_starting_values():
    mod, clusters = cue_model(ONE_ENDOG, 3)
    starting = pd.Series([1.0, 0.0, 0.5], index=["const", "w", "x"])
    with pytest.raises(ValueError, match="too few clusters") as exc_info:
        mod.fit(starting=starting, cov_type="clustered", clusters=clusters)
    assert exc_info.type is ValueError


def test_cue_too_few_clusters_is_found_before_the_optimization(monkeypatch):
    def minimize(*args, **kwargs):
        raise AssertionError("the objective is not defined, do not optimize it")

    monkeypatch.setattr(linearmodels.iv.model, "minimize", minimize)
    mod, clusters = cue_model(ONE_ENDOG, 3)
    with pytest.raises(ValueError, match="too few clusters"):
        mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize(
    ("nclusters", "center"),
    [(5, True), (6, True), (12, True), (5, False), (6, False), (12, False)],
)
def test_cue_with_enough_clusters_is_not_changed(nclusters, center):
    # These have more clusters than the 4 moment conditions
    mod, clusters = cue_model(ONE_ENDOG, nclusters, center=center)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    assert np.all(np.isfinite(res.params))
    assert np.all(np.isfinite(res.std_errors))
    params = res.params[["const", "w", "x"]].to_numpy()
    expected = cue_criterion(ONE_ENDOG, params, nclusters, center)
    assert_allclose(res.j_stat.stat, expected, rtol=1e-6)
    # The default is to center the moment conditions
    if center:
        default, _ = cue_model(ONE_ENDOG, nclusters)
        res_default = default.fit(cov_type="clustered", clusters=clusters)
        assert_allclose(res_default.params, res.params)


@pytest.mark.parametrize("weight_type", ["kernel", "robust"])
@pytest.mark.parametrize("nobs", [4, 5, 8])
def test_cue_too_few_observations(weight_type, nobs):
    # The rank of a covariance of centered moment conditions is at most nobs - 1
    data = small_sample(nobs)
    mod = IVGMMCUE(*data, weight_type=weight_type, **bandwidth_options(weight_type))
    if nobs > 4:
        res = mod.fit()
        assert np.isfinite(res.j_stat.stat)
        return
    with pytest.raises(ValueError, match="too few observations") as exc_info:
        mod.fit()
    assert exc_info.type is ValueError
    assert "4 moment conditions" in str(exc_info.value)


# ----------------------------------------------------------------------------
# IVGMM does not estimate a model with a singular covariance, and estimates the
# other models as it did
# ----------------------------------------------------------------------------

IVGMM_CONSEQUENCE = (
    "The weight matrix of the GMM estimator is the inverse of the covariance, so "
    "it is not defined."
)
CUE_CONSEQUENCE = (
    "The objective function of the continuously updating GMM estimator uses the "
    "inverse of the covariance, so it is not defined."
)


def exactly_identified_data(nobs=120):
    """2 exog, 1 endog and 1 instrument: 3 moments and 3 parameters"""
    rng = np.random.default_rng(3)
    z = pd.DataFrame(rng.standard_normal((nobs, 1)), columns=["z0"])
    x = pd.DataFrame({"x": z.sum(axis=1) + rng.standard_normal(nobs)})
    exog = pd.DataFrame({"const": 1.0, "w": rng.standard_normal(nobs)})
    y = pd.Series(1 + 0.5 * x.x + rng.standard_normal(nobs), name="y")
    return y, exog, x, z


EXACTLY_IDENTIFIED = exactly_identified_data()


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("nclusters", [2, 3])
def test_ivgmm_message_has_the_form_of_the_cue_message(nclusters, center):
    # The same message, except for the sentence that says why the inverse is needed
    with pytest.raises(ValueError, match="too few clusters") as ivgmm:
        fit(ONE_ENDOG, nclusters, center)
    with pytest.raises(ValueError, match="too few clusters") as cue:
        fit(ONE_ENDOG, nclusters, center, model=IVGMMCUE)
    message = str(ivgmm.value)
    assert IVGMM_CONSEQUENCE in message
    assert CUE_CONSEQUENCE in str(cue.value)
    assert message.replace(IVGMM_CONSEQUENCE, CUE_CONSEQUENCE) == str(cue.value)
    rank = nclusters - 1 if center else nclusters
    assert f"Its rank is at most {rank}," in message


def test_ivgmm_message_with_too_few_observations_has_the_form_of_the_cue_message():
    data = small_sample(4)
    with pytest.raises(ValueError, match="too few observations") as ivgmm:
        IVGMM(*data, center=True).fit()
    with pytest.raises(ValueError, match="too few observations") as cue:
        IVGMMCUE(*data).fit()
    message = str(ivgmm.value)
    assert "(4) for the number of moment conditions" in message
    assert message.replace(IVGMM_CONSEQUENCE, CUE_CONSEQUENCE) == str(cue.value)


@pytest.mark.parametrize("nclusters", [2, 3])
def test_ivgmm_with_the_report_of_the_bug(nclusters):
    # The estimate of the coefficient on x was 1.697 with 3 clusters when it is
    # 0.5, the standard errors were NaN, and it was a LinAlgError on some
    # platforms. There is nothing to estimate
    check_too_few_clusters(ONE_ENDOG, nclusters, center=False)


@pytest.mark.parametrize("nclusters", [2, 3])
@pytest.mark.parametrize("center", [False, True])
def test_ivgmm_exactly_identified_with_too_few_clusters(nclusters, center):
    # The weight matrix does not matter for the estimates of an exactly identified
    # model, but the noise in it did: the estimates were 10 away from the IV
    # estimates with 3 clusters, and the standard errors were not numbers. The
    # moment conditions at the first step estimates of an exactly identified
    # model add to zero, so that 3 clusters are not enough for its 3 moment
    # conditions even if they are not centered
    with pytest.raises(ValueError, match="too few clusters") as exc_info:
        fit(EXACTLY_IDENTIFIED, nclusters, center)
    assert exc_info.type is ValueError
    assert "3 moment conditions" in str(exc_info.value)
    assert f"({nclusters} clusters)" in str(exc_info.value)


@pytest.mark.parametrize("nclusters", [4, 5, 12])
@pytest.mark.parametrize("center", [False, True])
def test_ivgmm_exactly_identified_with_enough_clusters(nclusters, center):
    res = fit(EXACTLY_IDENTIFIED, nclusters, center)
    iv = IV2SLS(*EXACTLY_IDENTIFIED).fit()
    assert_allclose(res.params, iv.params, rtol=1e-8)
    assert np.all(np.isfinite(res.std_errors))


# params, std_errors, J, centered J and C statistics of valid models, which are
# the same as before the models with a singular covariance were not estimated.
# nan is an invalid statistic: 4 clusters are not enough for the centered
# covariance of the 4 moments, or for the 5 moments of the C statistic
REFERENCE = {
    "one_endog_4_uncentered": (
        ONE_ENDOG,
        4,
        False,
        {},
        None,
        [1.1060294133733184, -0.09579152673627506, 0.5112972864437724],
        [0.12262127936353251, 0.05938494098579946, 0.014495052398286558],
        0.7641884453729715,
        np.nan,
        np.nan,
    ),
    "one_endog_5_uncentered": (
        ONE_ENDOG,
        5,
        False,
        {},
        None,
        [1.1686397759103972, 0.02314636167513907, 0.4759766716861976],
        [0.02478733188229867, 0.10956469961382122, 0.043802173882083066],
        0.2854469335292345,
        0.2766732068378109,
        0.8759754606839069,
    ),
    "one_endog_5_centered": (
        ONE_ENDOG,
        5,
        True,
        {},
        None,
        [1.170136490148785, 0.022642982069993402, 0.47318217390609696],
        [0.02477994933455665, 0.10910357791629804, 0.04336862649295782],
        0.30272957956427843,
        0.31356562792598913,
        np.nan,
    ),
    "one_endog_12_centered": (
        ONE_ENDOG,
        12,
        True,
        {},
        None,
        [1.1416369745714467, -0.030890652538546615, 0.5142191592177071],
        [0.08948919724364655, 0.0900933799632663, 0.03672622060720279],
        0.5092695078373961,
        0.511039165481266,
        0.3666652130379392,
    ),
    "one_endog_12_iterated": (
        ONE_ENDOG,
        12,
        False,
        {"iter_limit": 5},
        None,
        [1.138693223505445, -0.03130105409944977, 0.514937395372159],
        [0.08941806134708769, 0.09013853082731596, 0.036698020920592794],
        0.48939723718769423,
        0.5102753816307153,
        0.3276841075791217,
    ),
    "two_endog_6_uncentered": (
        TWO_ENDOG,
        6,
        False,
        {},
        ["x1"],
        [0.9880803474466319, 0.4146283703294095, -0.3076604194012811],
        [0.024254582559327085, 0.07723707544855464, 0.018531865035777875],
        0.7736276096934414,
        1.0570377720126551,
        0.12392028655991782,
    ),
    "two_endog_12_centered": (
        TWO_ENDOG,
        12,
        True,
        {},
        None,
        [1.017400423332445, 0.3946521531745319, -0.32124858101494413],
        [0.05108132199782605, 0.06125247442528457, 0.03134643143047266],
        0.4445148871191505,
        0.4406830165857778,
        5.354334781473466,
    ),
}


def check_stat(stat, expected, name, df):
    if np.isnan(expected):
        check_invalid(stat, name)
    else:
        check_valid(stat, expected, df)


@pytest.mark.parametrize("name", REFERENCE)
def test_ivgmm_with_a_nonsingular_covariance_is_unchanged(name):
    data, nclusters, center, kwargs, tested, params, std_errors, j, cj, c = REFERENCE[
        name
    ]
    res = fit(data, nclusters, center, **kwargs)
    assert_allclose(res.params, params, rtol=1e-8)
    assert_allclose(res.std_errors, std_errors, rtol=1e-8)
    check_stat(res.j_stat, j, J_NAME, df=1)
    check_stat(
        res.centered_j_stat, cj, "Centered J-test of overidentifying restrictions", 1
    )
    ntested = data[2].shape[1] if tested is None else len(tested)
    check_stat(res.c_stat(tested), c, "C-statistic", df=ntested)


@pytest.mark.parametrize("name", REFERENCE)
def test_ivgmm_with_a_nonsingular_covariance_is_the_definition(name):
    data, nclusters, center, kwargs, _, params, _, j, _, _ = REFERENCE[name]
    x, y, z, _, _ = arrays(data)
    cov = clustered_cov(np.arange(y.shape[0]) % nclusters, center)
    # Two steps unless iter_limit is given, which the definition does not do
    if "iter_limit" not in kwargs:
        assert_allclose(reference_params(x, y, z, cov), params, rtol=1e-8)
        assert_allclose(reference_j(x, y, z, cov), j, rtol=1e-8)


def reference_params(x, y, z, cov):
    """Two-step GMM estimates, from the definition"""
    first = gmm_params(x, y, z, np.linalg.inv(z.T @ z / y.shape[0]))
    w = np.linalg.inv(cov(z * (y - x @ first)[:, None]))
    return gmm_params(x, y, z, w)


def test_ivgmm_one_step_does_not_use_the_covariance():
    # The weight matrix is (Z'Z/n)^{-1}, so there is nothing to be singular
    res = fit(ONE_ENDOG, 2, center=False, iter_limit=1)
    x, y, z, _, _ = arrays(ONE_ENDOG)
    expected = gmm_params(x, y, z, np.linalg.inv(z.T @ z / y.shape[0]))
    assert_allclose(res.params, expected, rtol=1e-8)
    assert res.iterations == 1


@pytest.mark.parametrize("iter_limit", [2, 5])
def test_ivgmm_with_an_initial_weight_still_tests_the_covariance(iter_limit):
    initial_weight = np.eye(4)
    check_too_few_clusters(
        ONE_ENDOG, 3, False, initial_weight=initial_weight, iter_limit=iter_limit
    )


def test_ivgmm_is_not_estimated_if_the_covariance_is_singular_in_a_later_step(
    monkeypatch,
):
    # The covariance is estimated at every step and not only at the first
    calls = []
    original = linearmodels.iv.model._is_singular

    def second_call_is_singular(cov):
        calls.append(1)
        return len(calls) == 2 or original(cov)

    monkeypatch.setattr(linearmodels.iv.model, "_is_singular", second_call_is_singular)
    with pytest.raises(ValueError, match="too few clusters"):
        fit(ONE_ENDOG, 12, center=False, iter_limit=5, tol=0.0)
    assert len(calls) == 2


def test_c_stat_is_invalid_if_the_model_of_the_statistic_cannot_be_estimated():
    # It has 5 moment conditions and needs 5 clusters, whereas the model that was
    # estimated has 4. The model of the statistic raises, as fit does, and the
    # statistic does not
    res = fit(ONE_ENDOG, 4, center=False)
    c_stat = res.c_stat()
    check_invalid(c_stat, "C-statistic")
    assert "singular" in str(c_stat)
