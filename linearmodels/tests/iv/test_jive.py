"""
Tests of IVJIVE, the jackknife instrumental variables estimator

Angrist, Imbens and Krueger (1999), Journal of Applied Econometrics

The reference values in results/jive-reference.csv were produced by
results/jive-reference.R. It computes JIVE from its definition by estimating
the first stage n times, leaving out one observation each time, and not with
the leverage shortcut that IVJIVE uses. Its covariance estimators are checked
against Stata results for 2SLS before it writes the values. There are no results
from Stata or R for JIVE itself in this repository.

The other tests do not need R: they compare with an explicit implementation of
the definition in numpy, for many randomly generated models, and check the
properties that the estimator has to have.
"""

import os

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from scipy import stats

from linearmodels.datasets import card
from linearmodels.iv import IV2SLS, IVJIVE, compare
from linearmodels.iv.results import IVResults
from linearmodels.shared.exceptions import MissingValueWarning

CWD = os.path.split(os.path.abspath(__file__))[0]
RESULTS = os.path.join(CWD, "results")

SIMULATED = pd.read_stata(os.path.join(RESULTS, "simulated-data.dta"))
SIMULATED["const"] = 1.0
# Dummy variables for cluster_id modulo 25, which are instruments with a high
# leverage. The names are the same as in the R script
GROUPS = pd.get_dummies(SIMULATED.cluster_id % 25, drop_first=True).astype(float)
GROUPS.columns = [f"g{i}" for i in range(1, GROUPS.shape[1] + 1)]
SIMULATED = pd.concat([SIMULATED, GROUPS], axis=1)

HOUSING = pd.read_csv(os.path.join(RESULTS, "housing.csv"), index_col=0)
HOUSING["const"] = 1.0
REGIONS = pd.get_dummies(HOUSING.region, drop_first=True).add_prefix("region")
HOUSING = pd.concat([HOUSING, REGIONS.astype(float)], axis=1)

CARD = card.load()
CARD["const"] = 1.0

DATASETS = {"sim": SIMULATED, "housing": HOUSING, "card": CARD}

REFERENCE = pd.read_csv(
    os.path.join(RESULTS, "jive-reference.csv"), keep_default_na=False
)
KEYS = [
    "id",
    "dataset",
    "dep",
    "exog",
    "endog",
    "instruments",
    "weights",
    "cov_type",
    "debiased",
    "kernel",
    "bandwidth",
    "clusters",
]


def reference_cases():
    cases = []
    for key, group in REFERENCE.groupby(KEYS, sort=False):
        row = dict(zip(KEYS, key, strict=True))
        # groupby keys are np.bool_ in some pandas versions
        row["debiased"] = bool(row["debiased"])
        kernel = f"-{row['kernel']}" if row["kernel"] else ""
        small = "small" if row["debiased"] else "asy"
        weighted = "-w" if row["weights"] else ""
        name = f"{row['id']}{weighted}-{row['cov_type']}{kernel}-{small}"
        cases.append(pytest.param(row, group, id=name))
    return cases


def reference_model(row):
    data = DATASETS[row["dataset"]]
    exog = data[row["exog"].split()] if row["exog"] else None
    weights = data[row["weights"]] if row["weights"] else None
    return IVJIVE(
        data[row["dep"]],
        exog,
        data[row["endog"].split()],
        data[row["instruments"].split()],
        weights=weights,
    )


def cov_options(row):
    data = DATASETS[row["dataset"]]
    options = {}
    if row["cov_type"] == "kernel":
        options["kernel"] = row["kernel"]
        options["bandwidth"] = float(row["bandwidth"])
    elif row["cov_type"] == "clustered":
        options["clusters"] = pd.factorize(data[row["clusters"]])[0]
    return options


@pytest.mark.parametrize(("row", "group"), reference_cases())
def test_against_r(row, group):
    res = reference_model(row).fit(
        cov_type=row["cov_type"], debiased=row["debiased"], **cov_options(row)
    )
    assert list(res.params.index) == list(group.term)
    assert_allclose(res.params.to_numpy(), group.coef.to_numpy(), rtol=1e-8)
    assert_allclose(res.std_errors.to_numpy(), group.se.to_numpy(), rtol=1e-7)
    assert_allclose(res.rsquared, group.r2.iloc[0], rtol=1e-8)
    assert_allclose(res.f_statistic.stat, group.fstat.iloc[0], rtol=1e-7)
    assert res.cov_type == row["cov_type"]
    assert res.debiased is row["debiased"]


def test_reference_covers_everything():
    # Guard against the reference table losing cases
    assert set(REFERENCE.cov_type) == {"unadjusted", "robust", "kernel", "clustered"}
    assert set(REFERENCE.kernel) == {"", "bartlett", "parzen", "qs"}
    assert set(REFERENCE.dataset) == {"sim", "housing", "card"}
    assert REFERENCE.weights.ne("").any()
    assert REFERENCE.debiased.any()
    assert not REFERENCE.debiased.all()
    assert {
        "one_endog",
        "two_endog",
        "three_endog",
        "just_identified",
        "multi_exog",
        "no_exog",
        "many_dummies",
        "weighted",
        "weighted_two_endog",
        "housing",
        "card",
    } <= set(REFERENCE.id)
    # A model with and without a constant
    assert (REFERENCE.exog == "").any()
    assert REFERENCE.exog.str.contains("const").any()


def leave_one_out_predictions(x, z, w=None):
    """Re-estimate the first stage, leaving out each observation in turn"""
    nobs = x.shape[0]
    w = np.ones(nobs) if w is None else w
    out = np.empty_like(x)
    for i in range(nobs):
        keep = np.arange(nobs) != i
        sw = np.sqrt(w[keep])[:, None]
        coef = np.linalg.lstsq(z[keep] * sw, x[keep] * sw, rcond=None)[0]
        out[i] = z[i] @ coef
    return out


def brute_force_jive(y, x, z, w=None):
    nobs = x.shape[0]
    w = np.ones(nobs) if w is None else w / w.mean()
    x_loo = leave_one_out_predictions(x, z, w)
    sw = np.sqrt(w)[:, None]
    xs, xs_loo, ys = x * sw, x_loo * sw, y[:, None] * sw
    return np.linalg.solve(xs_loo.T @ xs, xs_loo.T @ ys).squeeze()


def brute_force_cov(y, x, z, w, cov_type, debiased, clusters=None, bandwidth=3):
    """JIVE parameters and covariance, written out from the definition"""
    nobs, nvar = x.shape
    w = np.ones(nobs) if w is None else w / w.mean()
    x_loo = leave_one_out_predictions(x, z, w)
    sw = np.sqrt(w)[:, None]
    xs, xs_loo, ys = x * sw, x_loo * sw, y[:, None] * sw
    params = np.linalg.solve(xs_loo.T @ xs, xs_loo.T @ ys)
    eps = (ys - xs @ params).squeeze()
    scores = xs_loo * eps[:, None]
    if cov_type == "unadjusted":
        s = eps @ eps / nobs * xs_loo.T @ xs_loo / nobs
    elif cov_type == "robust":
        s = scores.T @ scores / nobs
    elif cov_type == "kernel":
        s = scores.T @ scores
        for lag in range(1, bandwidth + 1):
            gamma = scores[lag:].T @ scores[:-lag]
            s = s + (1 - lag / (bandwidth + 1)) * (gamma + gamma.T)
        s = s / nobs
    else:
        s = np.zeros((nvar, nvar))
        for cluster in np.unique(clusters):
            total = scores[clusters == cluster].sum(0)[:, None]
            s = s + total @ total.T
        s = s / nobs
    if debiased:
        if cov_type == "clustered":
            ngroups = len(np.unique(clusters))
            s = s * (nobs - 1) / (nobs - nvar) * ngroups / (ngroups - 1)
        else:
            s = s * nobs / (nobs - nvar)
    bread = np.linalg.inv(xs_loo.T @ xs / nobs)
    return params.squeeze(), bread @ s @ bread.T / nobs


def simulate(seed, nobs=120, ninstr=6, nendog=1):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((nobs, ninstr))
    v = rng.standard_normal((nobs, nendog))
    pi = rng.uniform(0.2, 0.8, (ninstr, nendog))
    endog = z @ pi + v
    exog = np.column_stack([np.ones(nobs), rng.standard_normal(nobs)])
    y = exog @ [1.0, 0.5] + endog.sum(1) + rng.standard_normal(nobs) + v.sum(1)
    return y, exog, endog, z


def random_design(seed):
    """A random model: sizes, constant, weights and clusters all vary"""
    rng = np.random.default_rng(1000 + seed)
    nobs = int(rng.integers(40, 150))
    nendog = int(rng.integers(1, 4))
    ninstr = nendog + int(rng.integers(0, 7))
    ncont = int(rng.integers(0, 3))
    const = bool(rng.random() < 0.75) or ncont == 0
    z = rng.standard_normal((nobs, ninstr))
    v = rng.standard_normal((nobs, nendog))
    endog = z @ rng.uniform(0.3, 1.0, (ninstr, nendog)) + v
    cols = ([np.ones(nobs)] if const else []) + [
        rng.standard_normal(nobs) for _ in range(ncont)
    ]
    exog = np.column_stack(cols)
    y = exog.sum(1) + endog.sum(1) + rng.standard_normal(nobs) + v.sum(1)
    w = rng.uniform(0.3, 3.0, nobs) if rng.random() < 0.4 else None
    clusters = rng.integers(0, int(rng.integers(5, 20)), nobs)
    return y, exog, endog, z, w, clusters


@pytest.mark.parametrize("seed", range(30))
def test_random_models_match_the_definition(seed):
    # Parameters and the four covariance estimators, with and without the small
    # sample adjustment, for models with a random number of regressors,
    # instruments, with or without a constant, weights and clusters
    y, exog, endog, z, w, clusters = random_design(seed)
    mod = IVJIVE(y, exog, endog, z, weights=w)
    x = np.column_stack([exog, endog])
    zfull = np.column_stack([exog, z])
    for cov_type in ("unadjusted", "robust", "kernel", "clustered"):
        options = {}
        if cov_type == "kernel":
            options["bandwidth"] = 3
        elif cov_type == "clustered":
            options["clusters"] = clusters
        for debiased in (False, True):
            res = mod.fit(cov_type=cov_type, debiased=debiased, **options)
            params, cov = brute_force_cov(
                y, x, zfull, w, cov_type, debiased, clusters=clusters
            )
            assert_allclose(res.params.to_numpy(), params, rtol=1e-8)
            assert_allclose(res.cov.to_numpy(), cov, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("nendog", [1, 2])
def test_matches_explicit_leave_one_out_refits(nendog, weighted):
    y, exog, endog, z = simulate(1, nendog=nendog)
    w = np.random.default_rng(2).uniform(0.3, 2.0, y.shape[0]) if weighted else None
    res = IVJIVE(y, exog, endog, z, weights=w).fit()
    expected = brute_force_jive(y, np.column_stack([exog, endog]), np.c_[exog, z], w)
    assert_allclose(res.params.to_numpy(), expected, rtol=1e-9)


@pytest.mark.parametrize("scale", [1e-5, 1e-6, 1e-8])
def test_accurate_with_nearly_collinear_instruments(scale):
    # Forming (Z'Z)^{-1} squares the condition number. It gives errors of about
    # 1e-7, 2e-6 and 4.5e-3 in the parameters for these three cases, where the
    # condition number of Z is 2e5, 2e6 and 2e8. This is the situation in which
    # JIVE is used: many instruments that are highly correlated
    rng = np.random.default_rng(3)
    nobs = 200
    base = rng.standard_normal((nobs, 3))
    z = np.column_stack([base[:, 0], base[:, 0] + scale * base[:, 1], base[:, 2]])
    v = rng.standard_normal(nobs)
    endog = z[:, :1] + v[:, None]
    exog = np.ones((nobs, 1))
    y = 1 + endog[:, 0] + rng.standard_normal(nobs) + v
    res = IVJIVE(y, exog, endog, z).fit()
    expected = brute_force_jive(y, np.column_stack([exog, endog]), np.c_[exog, z])
    assert_allclose(res.params.to_numpy(), expected, rtol=1e-8)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "kernel", "clustered"])
@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_without_endogenous_variables_jive_is_ols(cov_type, debiased, weighted):
    # With no endogenous variable the leave-one-out predictions are the
    # regressors, so JIVE is OLS and has the covariance of the same model in
    # IV2SLS. This checks every covariance estimator, with and without weights
    y, exog, _, _ = simulate(4)
    w = np.random.default_rng(5).uniform(0.3, 2.0, y.shape[0]) if weighted else None
    options = {}
    if cov_type == "kernel":
        options["bandwidth"] = 5
    elif cov_type == "clustered":
        options["clusters"] = np.arange(y.shape[0]) % 12
    jive = IVJIVE(y, exog, None, None, weights=w).fit(
        cov_type=cov_type, debiased=debiased, **options
    )
    ols = IV2SLS(y, exog, None, None, weights=w).fit(
        cov_type=cov_type, debiased=debiased, **options
    )
    assert_allclose(jive.params, ols.params, rtol=1e-10)
    assert_allclose(jive.cov, ols.cov, rtol=1e-8)
    assert_allclose(jive.rsquared, ols.rsquared, rtol=1e-10)
    assert_allclose(jive.f_statistic.stat, ols.f_statistic.stat, rtol=1e-8)


# Properties that the estimator has to have


def test_invariant_to_the_instruments_spanning_the_same_space():
    # JIVE only depends on the column space of Z. Invertible combinations of
    # the instruments, and adding multiples of the exogenous regressors to
    # them, do not change anything
    y, exog, endog, z = simulate(20, ninstr=5)
    rng = np.random.default_rng(21)
    mix = rng.standard_normal((5, 5))
    shifted = z @ mix + exog @ rng.standard_normal((exog.shape[1], 5))
    for cov_type in ("unadjusted", "robust"):
        base = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type)
        other = IVJIVE(y, exog, endog, shifted).fit(cov_type=cov_type)
        assert_allclose(other.params, base.params, rtol=1e-8)
        assert_allclose(other.cov, base.cov, rtol=1e-7)


def test_invariant_to_the_order_of_observations():
    y, exog, endog, z = simulate(22)
    clusters = np.arange(y.shape[0]) % 9
    perm = np.random.default_rng(23).permutation(y.shape[0])
    for cov_type in ("unadjusted", "robust", "clustered"):
        options = {"clusters": clusters} if cov_type == "clustered" else {}
        base = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type, **options)
        if cov_type == "clustered":
            options = {"clusters": clusters[perm]}
        other = IVJIVE(y[perm], exog[perm], endog[perm], z[perm]).fit(
            cov_type=cov_type, **options
        )
        assert_allclose(other.params, base.params, rtol=1e-9)
        assert_allclose(other.cov, base.cov, rtol=1e-8)


def test_invariant_to_the_order_of_regressors():
    y, exog, endog, z = simulate(24, nendog=2)
    base = IVJIVE(y, exog, endog, z).fit()
    # Reverse the exogenous and the endogenous regressors
    other = IVJIVE(y, exog[:, ::-1], endog[:, ::-1], z).fit()
    assert_allclose(other.params.to_numpy(), base.params.to_numpy()[[1, 0, 3, 2]])
    order = [1, 0, 3, 2]
    assert_allclose(other.cov.to_numpy(), base.cov.to_numpy()[np.ix_(order, order)])


def test_scaling_the_data():
    y, exog, endog, z = simulate(25)
    base = IVJIVE(y, exog, endog, z).fit()
    # Scaling y scales the parameters and their standard errors
    scaled = IVJIVE(7.5 * y, exog, endog, z).fit()
    assert_allclose(scaled.params, 7.5 * base.params, rtol=1e-9)
    assert_allclose(scaled.std_errors, 7.5 * base.std_errors, rtol=1e-8)
    assert_allclose(scaled.rsquared, base.rsquared, rtol=1e-9)
    # Scaling an endogenous variable scales the parameter in the other direction
    rescaled = IVJIVE(y, exog, 4 * endog, z).fit()
    assert_allclose(rescaled.params.iloc[-1], base.params.iloc[-1] / 4, rtol=1e-9)
    assert_allclose(
        rescaled.std_errors.iloc[-1], base.std_errors.iloc[-1] / 4, rtol=1e-8
    )
    assert_allclose(rescaled.params.iloc[:-1], base.params.iloc[:-1], rtol=1e-9)
    # Weights only matter up to scale
    w = np.random.default_rng(26).uniform(0.3, 2.0, y.shape[0])
    one = IVJIVE(y, exog, endog, z, weights=w).fit()
    two = IVJIVE(y, exog, endog, z, weights=37.5 * w).fit()
    assert_allclose(one.params, two.params, rtol=1e-10)
    assert_allclose(one.cov, two.cov, rtol=1e-9)


# The LIML kappa, which these tests do not use, is not accurate for such data
@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "kernel", "clustered"])
def test_badly_scaled_regressors(cov_type):
    # The t-statistics do not depend on the units of the variables. Forming the
    # inverse of (X'X~)(X~'X) in the covariance squares the condition number,
    # and gave standard errors that were wrong by orders of magnitude, or NaN,
    # for data like the housing and Card examples
    y, exog, endog, z = simulate(35, nendog=2)
    options = {}
    if cov_type == "kernel":
        options["bandwidth"] = 3
    elif cov_type == "clustered":
        options["clusters"] = np.arange(y.shape[0]) % 11
    scale = np.array([1.0, 1e4, 1e-3, 1e5])
    base = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type, **options)
    units = np.column_stack([exog, endog]) * scale
    other = IVJIVE(1e3 * y, units[:, :2], units[:, 2:], z).fit(
        cov_type=cov_type, **options
    )
    assert_allclose(
        other.params.to_numpy(), 1e3 * base.params.to_numpy() / scale, rtol=1e-6
    )
    assert_allclose(
        other.std_errors.to_numpy(), 1e3 * base.std_errors.to_numpy() / scale, rtol=1e-6
    )
    assert_allclose(other.tstats.to_numpy(), base.tstats.to_numpy(), rtol=1e-6)


def test_shifting_the_dependent_variable_changes_the_constant():
    y, exog, endog, z = simulate(27)
    base = IVJIVE(y, exog, endog, z).fit()
    shifted = IVJIVE(y + 3.0, exog, endog, z).fit()
    assert_allclose(shifted.params.iloc[0], base.params.iloc[0] + 3.0, rtol=1e-9)
    assert_allclose(shifted.params.iloc[1:], base.params.iloc[1:], rtol=1e-8)
    assert_allclose(shifted.cov, base.cov, rtol=1e-8)
    assert_allclose(shifted.resids, base.resids, atol=1e-10)


def test_weighted_equals_scaled_data():
    # A weighted model is the unweighted model for data multiplied by the
    # square root of the weights, here with weights that have mean one
    y, exog, endog, z = simulate(11)
    w = np.random.default_rng(12).uniform(0.3, 2.0, y.shape[0])
    w = w / w.mean()
    root = np.sqrt(w)[:, None]
    weighted = IVJIVE(y, exog, endog, z, weights=w).fit()
    scaled = IVJIVE(y * root[:, 0], exog * root, endog * root, z * root).fit()
    assert_allclose(weighted.params, scaled.params, rtol=1e-9)
    assert_allclose(weighted.cov, scaled.cov, rtol=1e-8)


def test_many_instruments_is_not_2sls():
    # JIVE differs from 2SLS when the leverages are not small
    y, exog, endog, z = simulate(28, nobs=60, ninstr=20)
    jive = IVJIVE(y, exog, endog, z).fit()
    tsls = IV2SLS(y, exog, endog, z).fit()
    assert abs(jive.params.iloc[-1] - tsls.params.iloc[-1]) > 1e-3


def test_cov_type_is_used():
    y, exog, endog, z = simulate(6)
    clusters = np.arange(y.shape[0]) % 15
    se = {}
    for cov_type, options in (
        ("unadjusted", {}),
        ("robust", {}),
        ("kernel", {"bandwidth": 3}),
        ("clustered", {"clusters": clusters}),
    ):
        res = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type, **options)
        assert res.cov_type == cov_type
        se[cov_type] = float(res.std_errors.iloc[-1])
    assert len({round(value, 8) for value in se.values()}) == len(se)
    # The default is robust
    default = IVJIVE(y, exog, endog, z).fit()
    assert default.cov_type == "robust"
    assert_allclose(default.cov, IVJIVE(y, exog, endog, z).fit(cov_type="robust").cov)


@pytest.mark.parametrize("seed", range(6))
def test_covariance_is_symmetric_and_positive_semidefinite(seed):
    # It is exactly symmetric, as the inverse square root and Cholesky
    # decompositions that are used with it expect
    y, exog, endog, z, w, clusters = random_design(seed)
    mod = IVJIVE(y, exog, endog, z, weights=w)
    for cov_type, options in (
        ("unadjusted", {}),
        ("robust", {}),
        ("kernel", {"bandwidth": 3}),
        ("clustered", {"clusters": clusters}),
    ):
        cov = mod.fit(cov_type=cov_type, **options).cov.to_numpy()
        assert_array_equal(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > -1e-10 * np.abs(cov).max()


@pytest.mark.parametrize("alias", ["homoskedastic", "heteroskedastic"])
def test_cov_type_aliases(alias):
    y, exog, endog, z = simulate(6)
    canonical = {"homoskedastic": "unadjusted", "heteroskedastic": "robust"}[alias]
    res = IVJIVE(y, exog, endog, z).fit(cov_type=alias)
    expected = IVJIVE(y, exog, endog, z).fit(cov_type=canonical)
    assert_allclose(res.cov, expected.cov, rtol=1e-12)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_kernel_options(kernel):
    y, exog, endog, z = simulate(29)
    default = IVJIVE(y, exog, endog, z).fit(cov_type="kernel", kernel=kernel)
    again = IVJIVE(y, exog, endog, z).fit(
        cov_type="kernel", kernel=kernel, bandwidth=y.shape[0] - 2
    )
    assert_allclose(default.cov, again.cov, rtol=1e-12)
    short = IVJIVE(y, exog, endog, z).fit(cov_type="kernel", kernel=kernel, bandwidth=2)
    assert not np.allclose(default.cov, short.cov)
    assert short.cov_config["kernel"] == kernel


def test_cov_config_is_not_shared_between_fits():
    y, exog, endog, z = simulate(30)
    mod = IVJIVE(y, exog, endog, z)
    first = mod.fit(cov_type="kernel", bandwidth=3)
    second = mod.fit(cov_type="robust")
    assert "bandwidth" in first.cov_config
    assert "bandwidth" not in second.cov_config
    assert_allclose(second.cov, mod.fit().cov)
    assert first.cov_config["bandwidth"] == 3


def test_debiased_scales_the_covariance():
    y, exog, endog, z = simulate(7)
    nobs, nvar = y.shape[0], exog.shape[1] + endog.shape[1]
    for cov_type in ("unadjusted", "robust"):
        base = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type)
        debiased = IVJIVE(y, exog, endog, z).fit(cov_type=cov_type, debiased=True)
        assert_allclose(debiased.cov, base.cov * nobs / (nobs - nvar), rtol=1e-10)
        assert debiased.debiased
        assert not base.debiased
        # The distribution used for the p-values and intervals is t(df_resid)
        assert_allclose(
            debiased.pvalues, 2 * stats.t.sf(np.abs(debiased.tstats), nobs - nvar)
        )
        assert_allclose(base.pvalues, 2 * stats.norm.sf(np.abs(base.tstats)))


# Errors


def test_invalid_cov_type():
    y, exog, endog, z = simulate(8)
    with pytest.raises(ValueError, match="Unknown cov_type"):
        IVJIVE(y, exog, endog, z).fit(cov_type="unknown")


def test_unknown_kernel():
    y, exog, endog, z = simulate(8)
    with pytest.raises(KeyError):
        IVJIVE(y, exog, endog, z).fit(cov_type="kernel", kernel="unknown")


def test_options_for_a_different_covariance_estimator_are_not_ignored():
    # The covariance options used to be silently ignored
    y, exog, endog, z = simulate(8)
    mod = IVJIVE(y, exog, endog, z)
    with pytest.raises(TypeError):
        mod.fit(cov_type="robust", bandwidth=4)
    with pytest.raises(TypeError):
        mod.fit(cov_type="unadjusted", clusters=np.arange(y.shape[0]) % 5)
    with pytest.raises(TypeError):
        mod.fit(cov_type="kernel", clusters=np.arange(y.shape[0]) % 5)


def test_clustered_without_clusters_raises():
    # It must not silently fall back to a different covariance estimator
    y, exog, endog, z = simulate(9)
    with pytest.raises((TypeError, ValueError)):
        IVJIVE(y, exog, endog, z).fit(cov_type="clustered")


def test_clusters_with_the_wrong_length_raise():
    y, exog, endog, z = simulate(9)
    with pytest.raises(ValueError, match="clusters has the wrong nobs"):
        IVJIVE(y, exog, endog, z).fit(cov_type="clustered", clusters=np.arange(10))


def test_perfect_leverage_raises():
    y, exog, endog, z = simulate(10)
    single = np.zeros((y.shape[0], 1))
    single[17] = 1.0
    with pytest.raises(ValueError, match="leverage of 1"):
        IVJIVE(y, exog, endog, np.c_[z, single]).fit()
    two = np.zeros((y.shape[0], 2))
    two[[3, 50], [0, 1]] = 1.0
    with pytest.raises(ValueError, match="2 observation"):
        IVJIVE(y, exog, endog, np.c_[z, two]).fit()


def test_invalid_models_are_rejected():
    y, exog, endog, z = simulate(10, nendog=2)
    with pytest.raises(ValueError, match="instruments"):
        IVJIVE(y, exog, endog, z[:, :1])
    with pytest.raises(ValueError, match="number of instruments"):
        IVJIVE(y, exog, endog, None)
    with pytest.raises(ValueError, match="rank"):
        IVJIVE(y, exog, endog, np.c_[z, z[:, :1]])
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        IVJIVE(y, exog, endog, z, weights=-np.ones(y.shape[0]))


def test_endogenous_variable_that_is_an_instrument():
    # If the endogenous variable is a column of the instruments its
    # leave-one-out prediction is itself, so JIVE is OLS of y on the regressors
    y, exog, _, z = simulate(31)
    endog = z[:, :1] + 0.0
    res = IVJIVE(y, exog, endog, z).fit()
    ols = IV2SLS(y, np.column_stack([exog, endog]), None, None).fit()
    assert_allclose(res.params.to_numpy(), ols.params.to_numpy(), rtol=1e-8)
    assert_allclose(res.cov.to_numpy(), ols.cov.to_numpy(), rtol=1e-7)


def test_failure_to_compute_kappa_only_affects_the_tests_that_use_it(monkeypatch):
    # The LIML kappa is only needed for the Anderson-Rubin and Basmann F tests,
    # so the model is still estimated if it cannot be computed
    def fail(self):
        raise np.linalg.LinAlgError("SVD did not converge")

    y, exog, endog, z = simulate(31)
    expected = IVJIVE(y, exog, endog, z).fit()
    monkeypatch.setattr(IVJIVE, "_estimate_kappa", fail)
    res = IVJIVE(y, exog, endog, z).fit()
    assert_allclose(res.params, expected.params, rtol=0, atol=0)
    assert_allclose(res.cov, expected.cov, rtol=0, atol=0)
    assert np.isnan(res.anderson_rubin.stat)
    assert np.isnan(res.basmann_f.stat)


# Data handling and the results


def test_missing_values_are_dropped():
    y, exog, endog, z = simulate(13)
    y_missing = y.copy()
    y_missing[[3, 40]] = np.nan
    z_missing = z.copy()
    z_missing[77, 1] = np.nan
    keep = np.ones(y.shape[0], dtype=bool)
    keep[[3, 40, 77]] = False
    with pytest.warns(MissingValueWarning, match="missing values"):
        res = IVJIVE(y_missing, exog, endog, z_missing).fit()
    expected = IVJIVE(y[keep], exog[keep], endog[keep], z[keep]).fit()
    assert res.nobs == keep.sum()
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)


def test_missing_values_in_weights_and_clusters():
    y, exog, endog, z = simulate(13)
    w = np.random.default_rng(14).uniform(0.3, 2.0, y.shape[0])
    w_missing = w.copy()
    w_missing[10] = np.nan
    keep = np.ones(y.shape[0], dtype=bool)
    keep[10] = False
    with pytest.warns(MissingValueWarning, match="missing values"):
        res = IVJIVE(y, exog, endog, z, weights=w_missing).fit()
    expected = IVJIVE(y[keep], exog[keep], endog[keep], z[keep], weights=w[keep]).fit()
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)


def test_pandas_names_and_index_are_kept():
    data = SIMULATED.iloc[10:200].copy()
    data.index = pd.date_range("2000-01-01", periods=data.shape[0], freq="D")
    res = IVJIVE(
        data.y_robust, data[["const", "x3"]], data[["x1"]], data[["z1", "z2", "x4"]]
    ).fit()
    tsls = IV2SLS(
        data.y_robust, data[["const", "x3"]], data[["x1"]], data[["z1", "z2", "x4"]]
    ).fit()
    assert list(res.params.index) == ["const", "x3", "x1"]
    assert list(res.std_errors.index) == ["const", "x3", "x1"]
    assert list(res.cov.index) == list(res.cov.columns) == ["const", "x3", "x1"]
    for attr in ("resids", "wresids", "fitted_values", "idiosyncratic"):
        assert getattr(res, attr).index.equals(data.index)
    assert res.model.dependent.pandas.index.equals(data.index)
    assert list(res.params.index) == list(tsls.params.index)
    assert res.model.endog.cols == ["x1"]
    assert res.model.instruments.cols == ["z1", "z2", "x4"]


def test_numpy_names_match_the_other_models():
    y, exog, endog, z = simulate(15)
    jive = IVJIVE(y, exog, endog, z).fit()
    tsls = IV2SLS(y, exog, endog, z).fit()
    assert list(jive.params.index) == list(tsls.params.index)


def test_from_formula():
    data = SIMULATED
    formula = "y_robust ~ 1 + x3 + [x1 ~ z1 + z2 + x4 + x5]"
    res = IVJIVE.from_formula(formula, data).fit()
    expected = IVJIVE(
        data.y_robust,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4", "x5"]],
    ).fit()
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)
    assert list(res.params.index) == ["Intercept", "x3", "x1"]

    weighted = IVJIVE.from_formula(formula, data, weights=data.weights).fit()
    direct = IVJIVE(
        data.y_robust,
        data[["const", "x3"]],
        data[["x1"]],
        data[["z1", "z2", "x4", "x5"]],
        weights=data.weights,
    ).fit()
    assert_allclose(weighted.params, direct.params, rtol=1e-10)
    assert IVJIVE.from_formula(formula, data).formula == formula
    assert res.model.formula == formula


def test_from_formula_other_transformations():
    data = CARD
    formula = "lwage ~ 1 + exper + I(exper ** 2) + black + smsa + south + [educ ~ nearc2 + nearc4]"
    res = IVJIVE.from_formula(formula, data).fit()
    assert "I(exper ** 2)" in res.params.index
    assert res.nobs == data.shape[0]


def test_results():
    y, exog, endog, z = simulate(14)
    res = IVJIVE(y, exog, endog, z).fit()
    assert isinstance(res, IVResults)
    assert res.model._method == "IV-JIVE"
    # JIVE is not a k-class estimator
    assert res.kappa is None
    for attr in dir(res):
        if attr.startswith("_") or attr in ("test_linear_constraint", "wald_test"):
            continue
        value = getattr(res, attr)
        if callable(value):
            value()
        assert isinstance(str(value), str)
    assert isinstance(str(res.first_stage), str)


def test_overidentification_tests_agree_with_2sls_data():
    # The Anderson-Rubin and Basmann F statistics use the LIML kappa, which
    # only depends on the data
    y, exog, endog, z = simulate(15)
    jive = IVJIVE(y, exog, endog, z).fit()
    tsls = IV2SLS(y, exog, endog, z).fit()
    assert_allclose(jive.anderson_rubin.stat, tsls.anderson_rubin.stat)
    assert_allclose(jive.basmann_f.stat, tsls.basmann_f.stat)


def test_first_stage_is_the_same_as_for_2sls():
    # The first stage does not depend on the estimator
    y, exog, endog, z = simulate(16, nendog=2)
    jive = IVJIVE(y, exog, endog, z).fit(cov_type="unadjusted")
    tsls = IV2SLS(y, exog, endog, z).fit(cov_type="unadjusted")
    pd.testing.assert_frame_equal(
        jive.first_stage.diagnostics, tsls.first_stage.diagnostics, rtol=1e-10
    )


def test_predict_and_fitted_values():
    y, exog, endog, z = simulate(17)
    res = IVJIVE(y, exog, endog, z).fit()
    x = np.column_stack([exog, endog])
    fitted = x @ res.params.to_numpy()
    assert_allclose(res.fitted_values.to_numpy().squeeze(), fitted, rtol=1e-10)
    assert_allclose(res.resids.to_numpy(), y - fitted, rtol=1e-10, atol=1e-10)
    assert_allclose(res.predict(exog, endog).to_numpy().squeeze(), fitted, rtol=1e-10)
    # The weighted residuals
    w = np.random.default_rng(18).uniform(0.3, 2.0, y.shape[0])
    weighted = IVJIVE(y, exog, endog, z, weights=w).fit()
    scale = np.sqrt(w / w.mean())
    assert_allclose(weighted.wresids.to_numpy(), scale * weighted.resids.to_numpy())


def test_inference_methods():
    y, exog, endog, z = simulate(19)
    res = IVJIVE(y, exog, endog, z).fit()
    crit = stats.norm.ppf(0.975)
    ci = res.conf_int()
    assert_allclose(ci.iloc[:, 0], res.params - crit * res.std_errors)
    assert_allclose(ci.iloc[:, 1], res.params + crit * res.std_errors)
    assert_allclose(res.tstats, res.params / res.std_errors)
    # The Wald test that the slope is zero is the square of the t-statistic
    nvar = res.params.shape[0]
    restriction = np.zeros((1, nvar))
    restriction[0, -1] = 1.0
    wald = res.wald_test(restriction, np.zeros(1))
    assert_allclose(wald.stat, res.tstats.iloc[-1] ** 2, rtol=1e-8)
    full = res.wald_test(np.eye(nvar), np.zeros(nvar))
    params = res.params.to_numpy()
    assert_allclose(full.stat, params @ np.linalg.inv(res.cov.to_numpy()) @ params)


def test_compare_with_other_estimators():
    y, exog, endog, z = simulate(32)
    results = {
        "JIVE": IVJIVE(y, exog, endog, z).fit(),
        "2SLS": IV2SLS(y, exog, endog, z).fit(),
    }
    comparison = compare(results)
    text = str(comparison.summary)
    assert "JIVE" in text
    assert "2SLS" in text
    assert "IV-JIVE" in text
    assert_allclose(comparison.params["JIVE"], results["JIVE"].params)


def test_summary():
    y, exog, endog, z = simulate(16)
    for cov_type, expected in (
        ("unadjusted", "Unadjusted"),
        ("robust", "Robust"),
        ("kernel", "Kernel"),
        ("clustered", "Clustered"),
    ):
        options = (
            {"clusters": np.arange(y.shape[0]) % 9} if cov_type == "clustered" else {}
        )
        text = str(IVJIVE(y, exog, endog, z).fit(cov_type=cov_type, **options).summary)
        assert "IV-JIVE" in text
        assert "JIVE Covariance" in text
        assert expected in text


def test_fit_does_not_change_the_model():
    y, exog, endog, z = simulate(33)
    mod = IVJIVE(y, exog, endog, z)
    wy, wx, wz = mod._wy.copy(), mod._wx.copy(), mod._wz.copy()
    first = mod.fit()
    mod.fit(cov_type="unadjusted", debiased=True)
    again = mod.fit()
    assert_array_equal(wy, mod._wy)
    assert_array_equal(wx, mod._wx)
    assert_array_equal(wz, mod._wz)
    assert_allclose(first.params, again.params, rtol=0, atol=0)
    assert_allclose(first.cov, again.cov, rtol=0, atol=0)


def test_consistent_in_large_samples():
    # With strong instruments the estimate is within sampling error of the
    # truth, using the standard error of the estimator
    rng = np.random.default_rng(34)
    nobs, beta = 20000, 1.5
    z = rng.standard_normal((nobs, 4))
    v = rng.standard_normal(nobs)
    x = z @ [0.5, 0.4, 0.3, 0.2] + v
    y = 0.3 + beta * x + 0.5 * v + rng.standard_normal(nobs)
    res = IVJIVE(y, np.ones((nobs, 1)), x[:, None], z).fit()
    assert abs(res.params.iloc[-1] - beta) < 4 * res.std_errors.iloc[-1]
    assert res.std_errors.iloc[-1] < 0.05
