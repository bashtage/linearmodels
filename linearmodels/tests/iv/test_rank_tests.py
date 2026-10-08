"""
Tests of the Cragg-Donald and Kleibergen-Paap tests of the first-stage rank.

Reference values for the simulated data come from ``ivreg2r`` 0.1.0, an R
port of Stata's ``ivreg2`` and ``ranktest``, run on the dataset in
``results/simulated-data.dta`` using ``diagnostics()``:

* ``underid``: Anderson canonical correlation LM statistic when the errors
  are i.i.d., otherwise the Kleibergen-Paap rk LM statistic
* ``weak_id``: Cragg-Donald Wald F statistic
* ``weak_id_robust``: Kleibergen-Paap rk Wald F statistic. For the i.i.d.
  covariance ivreg2 does not report it since it is the Cragg-Donald F.

The ``kernel`` cases use a Bartlett kernel with Stata bandwidth 5, which is a
bandwidth of 4 here, and ``_w`` cases use the ``weights`` column. Models A and
C are weakly identified and B is strongly identified, so the values cover both.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.tools.tools import add_constant

from linearmodels.iv import IV2SLS, IVGMM, IVLIML
from linearmodels.iv.common import (
    cragg_donald,
    cragg_donald_f,
    kleibergen_paap,
    kleibergen_paap_f,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic

CWD = os.path.split(os.path.abspath(__file__))[0]
SIMULATED_DATA = pd.read_stata(os.path.join(CWD, "results", "simulated-data.dta"))

# key: (underid statistic, underid df, Cragg-Donald F, rk Wald F)
REFERENCE = {
    "A|iid": (1.02366462810958, 1, 0.507579977028276, 0.507579977028276),
    "A|robust": (0.989618462661128, 1, 0.507579977028276, 0.495111283552873),
    "A|clustered": (0.958326219766078, 1, 0.507579977028276, 0.481216532411928),
    "A|kernel": (0.989164039658067, 1, 0.507579977028276, 0.501262987984046),
    "A|robust_w": (2.76718568999369, 1, 2.06018527294258, 1.44152791081534),
    "A|clustered_w": (2.61622921547341, 1, 2.06018527294258, 1.39402910658697),
    "B|iid": (32.763825565663, 2, 17.1548582963803, 17.1548582963803),
    "B|robust": (28.2011546920011, 2, 17.1548582963803, 17.3639182117703),
    "B|clustered": (28.4875757936386, 2, 17.1548582963803, 22.2146455418097),
    "B|kernel": (26.0067505861319, 2, 17.1548582963803, 19.2327502441631),
    "B|robust_w": (26.9635739907289, 2, 20.719519537991, 18.2733436260455),
    "B|clustered_w": (26.8587089233714, 2, 20.719519537991, 22.689656278189),
    "C|iid": (1.0853612924986, 2, 0.358818305691267, 0.358818305691267),
    "C|robust": (1.05939417036869, 2, 0.358818305691267, 0.353702373816456),
    "C|clustered": (1.05264944480239, 2, 0.358818305691267, 0.35471587132937),
    "C|kernel": (1.05135458593092, 2, 0.358818305691267, 0.35685280129613),
    "C|robust_w": (2.97153459392194, 2, 1.65725730744342, 1.04945529707942),
    "C|clustered_w": (2.82455694558932, 2, 1.65725730744342, 1.04696991314578),
}
KEYS = list(REFERENCE)
COV_FIT_OPTIONS = {
    "iid": {"cov_type": "unadjusted"},
    "robust": {"cov_type": "robust"},
    "clustered": {"cov_type": "clustered", "clusters": SIMULATED_DATA.cluster_id},
    "kernel": {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 4},
}


def simulated_inputs(spec):
    """Model A and B have x3, x4, x5 as controls. C moves x5 to the instruments"""
    data = SIMULATED_DATA
    if spec == "A":
        return data[["x1", "x2"]], data[["x3", "x4", "x5"]], data[["z1", "z2"]]
    if spec == "B":
        return data[["x1"]], data[["x3", "x4", "x5"]], data[["z1", "z2"]]
    return data[["x1", "x2"]], data[["x3", "x4"]], data[["z1", "z2", "x5"]]


def split_key(key):
    spec, cov = key.split("|")
    weighted = cov.endswith("_w")
    return spec, cov.removesuffix("_w"), weighted


def function_arguments(key):
    spec, cov, weighted = split_key(key)
    endog, controls, instr = simulated_inputs(spec)
    exog = add_constant(controls)
    arrays = [np.asarray(a, dtype=float) for a in (endog, instr, exog)]
    if weighted:
        root_w = np.sqrt(SIMULATED_DATA.weights.to_numpy())[:, None]
        arrays = [root_w * a for a in arrays]
    options = dict(COV_FIT_OPTIONS[cov])
    cov_type = options.pop("cov_type")
    if "clusters" in options:
        options["clusters"] = np.asarray(options["clusters"])
    return arrays, cov_type, options


def random_design(seed, nobs=300, ninstr=4, nendog=2, heteroskedastic=True):
    rs = np.random.RandomState(seed)
    exog = np.c_[np.ones(nobs), rs.standard_normal((nobs, 1))]
    z = rs.standard_normal((nobs, ninstr))
    pi = 0.4 * rs.standard_normal((ninstr, nendog))
    scale = np.exp(0.5 * z[:, [0]]) if heteroskedastic else 1.0
    x = z @ pi + 0.5 * exog[:, [1]] + scale * rs.standard_normal((nobs, nendog))
    return x, z, exog


def partial_out(x, controls):
    if controls.shape[1] == 0:
        return x
    q, _ = np.linalg.qr(controls)
    return x - q @ (q.T @ x)


def canonical_correlations(endog, instr, exog):
    """Partial canonical correlations from QR factors"""
    qx, _ = np.linalg.qr(partial_out(endog, exog))
    qz, _ = np.linalg.qr(partial_out(instr, exog))
    return np.linalg.svd(qx.T @ qz, compute_uv=False)


def symmetric_inverse_root(a):
    values, vectors = np.linalg.eigh(a)
    return vectors @ np.diag(values**-0.5) @ vectors.T


def symmetric_root(a):
    values, vectors = np.linalg.eigh(a)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def kp_paper(endog, instr, exog):
    """
    Kleibergen and Paap (2006) rk statistics for H0: rank = nendog - 1 with a
    heteroskedasticity-robust covariance, using the normalized bases from
    the paper (U22, V22) and symmetric square roots of the second moments.
    Returns the LM and Wald chi2 statistics.
    """
    nobs = endog.shape[0]
    x = partial_out(endog, exog)
    z = partial_out(instr, exog)
    k, m = z.shape[1], x.shape[1]
    qzz, qxx, qzx = z.T @ z / nobs, x.T @ x / nobs, z.T @ x / nobs
    izz, ixx = symmetric_inverse_root(qzz), symmetric_inverse_root(qxx)
    theta = izz @ qzx @ ixx
    u, _, vt = np.linalg.svd(theta)
    v = vt.T
    kk = m - 1
    u22, v22 = u[kk:, kk:], v[kk:, kk:]
    aq = u[:, kk:] @ np.linalg.inv(u22) @ symmetric_root(u22 @ u22.T)
    bq = symmetric_root(v22 @ v22.T) @ np.linalg.inv(v22.T) @ v[:, kk:].T
    pihat = np.linalg.solve(qzz, qzx)
    out = []
    for series in (x, x - z @ pihat):
        scores = np.column_stack([series[:, [j]] * z for j in range(m)])
        shat = scores.T @ scores / nobs
        transform = np.kron(ixx.T, izz.T)
        kpvar = transform @ shat @ transform.T
        sel = np.kron(bq, aq.T)
        lam = sel @ theta.ravel(order="F")
        vlam = sel @ kpvar @ sel.T
        out.append(nobs * float(lam @ np.linalg.solve(vlam, lam)))
    assert theta.shape == (k, m)
    return out


def first_stage_wald(x, z, exog, clusters=None):
    """Wald test of the excluded instruments in a first stage OLS regression"""
    reg = np.c_[exog, z]
    bread = np.linalg.inv(reg.T @ reg)
    beta = bread @ reg.T @ x
    scores = reg * (x - reg @ beta)[:, None]
    if clusters is not None:
        scores = pd.DataFrame(scores).groupby(clusters).sum().to_numpy()
    cov = bread @ (scores.T @ scores) @ bread
    k = z.shape[1]
    return float(beta[-k:].T @ np.linalg.solve(cov[-k:, -k:], beta[-k:]))


@pytest.mark.parametrize("key", KEYS)
def test_reference_values(key):
    (endog, instr, exog), cov_type, options = function_arguments(key)
    underid, underid_df, cd_f, kp_f = REFERENCE[key]
    test = kleibergen_paap(endog, instr, exog, cov_type, options)
    assert isinstance(test, WaldTestStatistic)
    assert not isinstance(test, InvalidTestStatistic)
    assert_allclose(test.stat, underid, rtol=1e-8)
    assert test.df == underid_df
    assert_allclose(
        kleibergen_paap_f(endog, instr, exog, cov_type, options), kp_f, rtol=1e-8
    )
    assert_allclose(cragg_donald_f(endog, instr, exog), cd_f, rtol=1e-8)
    assert_allclose(
        cragg_donald(endog, instr, exog).stat, cd_f * instr.shape[1], rtol=1e-8
    )


@pytest.mark.parametrize("key", KEYS)
def test_first_stage_reference_values(key):
    spec, cov, weighted = split_key(key)
    endog, controls, instr = simulated_inputs(spec)
    weights = SIMULATED_DATA.weights if weighted else None
    mod = IV2SLS(
        SIMULATED_DATA.y_robust, add_constant(controls), endog, instr, weights=weights
    )
    first_stage = mod.fit(**COV_FIT_OPTIONS[cov]).first_stage
    underid, underid_df, cd_f, kp_f = REFERENCE[key]
    assert_allclose(first_stage.kleibergen_paap.stat, underid, rtol=1e-8)
    assert first_stage.kleibergen_paap.df == underid_df
    assert_allclose(first_stage.kleibergen_paap_f, kp_f, rtol=1e-8)
    assert_allclose(first_stage.cragg_donald_f, cd_f, rtol=1e-8)
    assert_allclose(first_stage.cragg_donald.stat, cd_f * instr.shape[1], rtol=1e-8)


@pytest.mark.parametrize("fit_type", [IV2SLS, IVLIML, IVGMM])
def test_first_stage_model_types(fit_type):
    endog, controls, instr = simulated_inputs("B")
    mod = fit_type(SIMULATED_DATA.y_robust, add_constant(controls), endog, instr)
    first_stage = mod.fit(cov_type="robust").first_stage
    assert_allclose(
        first_stage.kleibergen_paap.stat, REFERENCE["B|robust"][0], rtol=1e-8
    )
    assert_allclose(first_stage.kleibergen_paap_f, REFERENCE["B|robust"][3], rtol=1e-8)


@pytest.mark.parametrize(
    ("alias", "canonical", "key"),
    [
        ("homo", "unadjusted", "B|iid"),
        ("homoskedastic", "unadjusted", "B|iid"),
        ("HomoskedasticCovariance", "unadjusted", "B|iid"),
        ("hccm", "robust", "B|robust"),
        ("heteroskedastic", "robust", "B|robust"),
        ("HeteroskedasticCovariance", "robust", "B|robust"),
        ("one-way", "clustered", "B|clustered"),
        ("OneWayClusteredCovariance", "clustered", "B|clustered"),
        ("KernelCovariance", "kernel", "B|kernel"),
    ],
)
def test_cov_type_aliases(alias, canonical, key):
    (endog, instr, exog), _, options = function_arguments(key)
    expected = kleibergen_paap(endog, instr, exog, canonical, options)
    result = kleibergen_paap(endog, instr, exog, alias, options)
    assert_allclose(result.stat, expected.stat)
    assert_allclose(
        kleibergen_paap_f(endog, instr, exog, alias, options),
        kleibergen_paap_f(endog, instr, exog, canonical, options),
    )


def test_aliases_through_model():
    endog, controls, instr = simulated_inputs("B")
    mod = IV2SLS(SIMULATED_DATA.y_robust, add_constant(controls), endog, instr)
    for alias in ("hccm", "heteroskedastic", "robust"):
        first_stage = mod.fit(cov_type=alias).first_stage
        assert_allclose(first_stage.kleibergen_paap.stat, REFERENCE["B|robust"][0])
    first_stage = mod.fit(
        cov_type="one-way", clusters=SIMULATED_DATA.cluster_id
    ).first_stage
    assert_allclose(first_stage.kleibergen_paap.stat, REFERENCE["B|clustered"][0])


@pytest.mark.parametrize("nendog", [1, 2, 3])
def test_unadjusted_is_anderson_and_cragg_donald(nendog):
    x, z, exog = random_design(nendog, nendog=nendog, ninstr=5)
    nobs = x.shape[0]
    r2 = canonical_correlations(x, z, exog)[-1] ** 2
    lm = kleibergen_paap(x, z, exog, "unadjusted")
    assert_allclose(lm.stat, nobs * r2)
    assert lm.df == 5 - nendog + 1
    cd = cragg_donald(x, z, exog)
    assert_allclose(cd.stat, (nobs - 5 - 2) * r2 / (1 - r2))
    assert_allclose(cragg_donald_f(x, z, exog), cd.stat / 5)
    assert_allclose(kleibergen_paap_f(x, z, exog, "unadjusted"), cd.stat / 5)


def test_cragg_donald_single_endogenous_is_classical_f():
    x, z, exog = random_design(10, nendog=1, ninstr=3)
    nobs = x.shape[0]
    restricted = partial_out(x, exog)
    full = partial_out(x, np.c_[exog, z])
    rss_r, rss_u = float((restricted**2).sum()), float((full**2).sum())
    classical_f = (rss_r - rss_u) / 3 / (rss_u / (nobs - 3 - 2))
    assert_allclose(cragg_donald_f(x, z, exog), classical_f)
    assert_allclose(cragg_donald(x, z, exog).stat, 3 * classical_f)


@pytest.mark.parametrize("seed", range(4))
def test_cragg_donald_matches_canonical_correlations(seed):
    x, z, exog = random_design(seed, nendog=seed % 3 + 1, ninstr=4 + seed % 2)
    r2 = canonical_correlations(x, z, exog)[-1] ** 2
    nobs, k = z.shape[0], z.shape[1]
    assert_allclose(cragg_donald(x, z, exog).stat, (nobs - k - 2) * r2 / (1 - r2))


@pytest.mark.parametrize("nendog", [2, 3])
def test_robust_matches_paper_formulation(nendog):
    # Uses the normalized null-space bases in Kleibergen and Paap (2006) and
    # a different standardization of the first stage. Both are implied to
    # produce the same statistic.
    x, z, exog = random_design(20 + nendog, nendog=nendog, ninstr=5)
    nobs, k = x.shape[0], z.shape[1]
    lm_chi2, wald_chi2 = kp_paper(x, z, exog)
    assert_allclose(kleibergen_paap(x, z, exog, "robust").stat, lm_chi2, rtol=1e-8)
    expected_f = wald_chi2 / nobs * (nobs - k - 2) / k
    assert_allclose(kleibergen_paap_f(x, z, exog, "robust"), expected_f, rtol=1e-8)


@pytest.mark.parametrize("clustered", [False, True])
def test_single_endogenous_f_is_first_stage_wald(clustered):
    # With one endogenous regressor the rk Wald statistic is the robust Wald
    # test of the excluded instruments in the first stage
    x, z, exog = random_design(30, nendog=1, ninstr=3)
    nobs, k = x.shape[0], z.shape[1]
    if clustered:
        clusters = np.arange(nobs) // 15
        nclusters = np.unique(clusters).shape[0]
        wald = first_stage_wald(x[:, 0], z, exog, clusters)
        expected = wald / (nobs - 1) * (nobs - k - 2) * (nclusters - 1) / nclusters / k
        actual = kleibergen_paap_f(x, z, exog, "clustered", {"clusters": clusters})
    else:
        wald = first_stage_wald(x[:, 0], z, exog)
        expected = wald / nobs * (nobs - k - 2) / k
        actual = kleibergen_paap_f(x, z, exog, "robust")
    assert_allclose(actual, expected, rtol=1e-8)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "clustered", "kernel"])
def test_invariance_to_linear_transformations(cov_type):
    x, z, exog = random_design(40, nendog=2, ninstr=4)
    rs = np.random.RandomState(41)
    options = {"clusters": np.arange(x.shape[0]) // 10, "bandwidth": 3}
    a = rs.standard_normal((2, 2))
    b = rs.standard_normal((4, 4))
    x2, z2 = 5.0 * x @ a, z @ b / 3.0
    for func in (kleibergen_paap, kleibergen_paap_f):
        original = func(x, z, exog, cov_type, options)
        transformed = func(x2, z2, exog, cov_type, options)
        if func is kleibergen_paap:
            original, transformed = original.stat, transformed.stat
        assert_allclose(transformed, original, rtol=1e-8)
    assert_allclose(cragg_donald_f(x2, z2, exog), cragg_donald_f(x, z, exog))
    assert_allclose(cragg_donald(x2, z2, exog).stat, cragg_donald(x, z, exog).stat)


def test_invariant_to_ordering_of_endogenous():
    x, z, exog = random_design(50, nendog=3, ninstr=5)
    original = kleibergen_paap(x, z, exog, "robust").stat
    permuted = kleibergen_paap(x[:, [2, 0, 1]], z, exog, "robust").stat
    assert_allclose(permuted, original, rtol=1e-8)


def test_no_exogenous_regressors():
    x, z, _ = random_design(60, nendog=2, ninstr=4)
    exog = np.empty((x.shape[0], 0))
    lm = kleibergen_paap(x, z, exog, "robust")
    expected, _ = kp_paper(x, z, exog)
    assert_allclose(lm.stat, expected, rtol=1e-8)
    assert np.isfinite(kleibergen_paap_f(x, z, exog, "robust"))


def test_clusters_as_column_vector():
    x, z, exog = random_design(70, nendog=2, ninstr=4)
    clusters = np.arange(x.shape[0]) // 10
    expected = kleibergen_paap(x, z, exog, "clustered", {"clusters": clusters})
    column = kleibergen_paap(x, z, exog, "clustered", {"clusters": clusters[:, None]})
    assert_allclose(column.stat, expected.stat)


@pytest.mark.parametrize(
    "options",
    [{}, {"bandwidth": 5}, {"kernel": "parzen", "bandwidth": 6}, {"kernel": "qs"}],
    ids=["default", "bandwidth", "parzen", "qs-automatic"],
)
def test_kernel_options(options):
    x, z, exog = random_design(80, nendog=2, ninstr=4, heteroskedastic=False)
    test = kleibergen_paap(x, z, exog, "kernel", options)
    assert isinstance(test, WaldTestStatistic)
    assert np.isfinite(test.stat)
    assert test.df == 3
    assert np.isfinite(kleibergen_paap_f(x, z, exog, "kernel", options))


def test_kernel_bandwidth_changes_result():
    x, z, exog = random_design(81, nendog=2, ninstr=4)
    short = kleibergen_paap(x, z, exog, "kernel", {"bandwidth": 1}).stat
    long = kleibergen_paap(x, z, exog, "kernel", {"bandwidth": 12}).stat
    assert short != pytest.approx(long, rel=1e-6)


def test_exactly_identified():
    x, z, exog = random_design(90, nendog=2, ninstr=2)
    test = kleibergen_paap(x, z, exog, "robust")
    assert test.df == 1
    lm_chi2, _ = kp_paper(x, z, exog)
    assert_allclose(test.stat, lm_chi2, rtol=1e-8)


@pytest.mark.parametrize("func", [cragg_donald, kleibergen_paap])
def test_no_endogenous(func):
    x, z, exog = random_design(100, nendog=2, ninstr=4)
    result = func(np.empty((x.shape[0], 0)), z, exog)
    assert isinstance(result, InvalidTestStatistic)
    assert np.isnan(result.pval)
    assert "no endogenous regressors" in str(result)


@pytest.mark.parametrize("func", [cragg_donald, kleibergen_paap])
def test_too_few_instruments(func):
    x, z, exog = random_design(101, nendog=3, ninstr=4)
    result = func(x, z[:, :2], exog)
    assert isinstance(result, InvalidTestStatistic)
    assert "less than the number of endogenous" in str(result)


@pytest.mark.parametrize("func", [cragg_donald, kleibergen_paap])
def test_collinear_endogenous(func):
    x, z, exog = random_design(102, nendog=2, ninstr=4)
    x = np.c_[x, x[:, 0] + 2 * x[:, 1]]
    result = func(x, z, exog)
    assert isinstance(result, InvalidTestStatistic)
    assert "endogenous regressors are collinear" in str(result)


@pytest.mark.parametrize("func", [cragg_donald, kleibergen_paap])
def test_collinear_instruments(func):
    x, z, exog = random_design(103, nendog=2, ninstr=4)
    z = np.c_[z, z[:, 0] - z[:, 1]]
    result = func(x, z, exog)
    assert isinstance(result, InvalidTestStatistic)
    assert "instruments are collinear" in str(result)


@pytest.mark.parametrize("func", [cragg_donald, kleibergen_paap])
def test_exact_first_stage(func):
    x, z, exog = random_design(104, nendog=2, ninstr=4)
    rs = np.random.RandomState(0)
    x = z @ rs.standard_normal((4, 2)) + exog @ rs.standard_normal((2, 2))
    result = func(x, z, exog)
    assert isinstance(result, InvalidTestStatistic)
    assert "exactly predicted" in str(result)


def test_invalid_statistics_have_nan_f():
    x, z, exog = random_design(105, nendog=2, ninstr=4)
    empty = np.empty((x.shape[0], 0))
    assert np.isnan(cragg_donald_f(empty, z, exog))
    assert np.isnan(kleibergen_paap_f(empty, z, exog, "robust"))


def test_null_hypothesis_is_underidentification():
    x, z, exog = random_design(106, nendog=2, ninstr=4)
    for test in (cragg_donald(x, z, exog), kleibergen_paap(x, z, exog, "robust")):
        assert "underidentified" in test.null
        assert "does not have full column rank" in test.null
        assert "jointly identify" not in test.null
        assert test.dist_name == "chi2(3)"
        assert "H0: " in str(test)


def test_names():
    x, z, exog = random_design(107)
    assert "Cragg-Donald" in str(cragg_donald(x, z, exog))
    assert "Kleibergen-Paap" in str(kleibergen_paap(x, z, exog, "robust"))


def test_unsupported_cov_type():
    x, z, exog = random_design(108)
    with pytest.raises(ValueError, match="Unsupported cov_type"):
        kleibergen_paap(x, z, exog, "unknown")
    with pytest.raises(ValueError, match="Unsupported cov_type"):
        kleibergen_paap_f(x, z, exog, "unknown")


def test_clusters_required():
    x, z, exog = random_design(109)
    with pytest.raises(ValueError, match="clusters are required"):
        kleibergen_paap(x, z, exog, "clustered")
    with pytest.raises(ValueError, match="clusters are required"):
        kleibergen_paap(x, z, exog, "clustered", {"clusters": None})


def test_clusters_wrong_length():
    x, z, exog = random_design(110)
    with pytest.raises(ValueError, match="clusters has the wrong nobs"):
        kleibergen_paap(
            x, z, exog, "clustered", {"clusters": np.arange(x.shape[0] - 1)}
        )


def test_two_way_clustering_not_supported():
    x, z, exog = random_design(111)
    clusters = np.c_[np.arange(x.shape[0]) // 10, np.arange(x.shape[0]) // 7]
    result = kleibergen_paap(x, z, exog, "clustered", {"clusters": clusters})
    assert isinstance(result, InvalidTestStatistic)
    assert "two-way clustering" in str(result)
    assert np.isnan(kleibergen_paap_f(x, z, exog, "clustered", {"clusters": clusters}))


def test_too_few_clusters_warns_and_reduces_df():
    # The covariance of the cluster sums of the scores has rank at most the
    # number of clusters, since the scores of the LM statistic are not mean
    # zero. The scores of the Wald statistic are mean zero and have one less.
    x, z, exog = random_design(112, nendog=1, ninstr=5)
    clusters = np.arange(x.shape[0]) // 100
    with pytest.warns(UserWarning, match="rank deficient"):
        test = kleibergen_paap(x, z, exog, "clustered", {"clusters": clusters})
    assert test.df == 3
    assert np.isfinite(test.stat)
    with pytest.warns(UserWarning, match="rank deficient"):
        assert np.isfinite(
            kleibergen_paap_f(x, z, exog, "clustered", {"clusters": clusters})
        )


def test_one_cluster_is_invalid():
    x, z, exog = random_design(113, nendog=1, ninstr=3)
    clusters = np.zeros(x.shape[0], dtype=int)
    test = kleibergen_paap(x, z, exog, "clustered", {"clusters": clusters})
    assert isinstance(test, InvalidTestStatistic)
    assert "at least two clusters" in str(test)
    assert np.isnan(kleibergen_paap_f(x, z, exog, "clustered", {"clusters": clusters}))


def test_singular_covariance_is_invalid():
    # Observations come in pairs (x, z) and (x, -z), so the cluster sums of
    # x * z are exactly zero and the covariance of the LM scores is zero
    rs = np.random.RandomState(114)
    half = 100
    z = rs.standard_normal((half, 3))
    x = rs.standard_normal((half, 1))
    x_full, z_full = np.r_[x, x], np.r_[z, -z]
    clusters = np.r_[np.arange(half), np.arange(half)]
    exog = np.empty((2 * half, 0))
    result = kleibergen_paap(x_full, z_full, exog, "clustered", {"clusters": clusters})
    assert isinstance(result, InvalidTestStatistic)
    assert "singular" in str(result)
    nan = kleibergen_paap_f(x_full, z_full, exog, "clustered", {"clusters": clusters})
    assert np.isnan(nan)


def test_first_stage_no_endogenous():
    # IVGMM is the only estimator whose results have a first stage when the
    # model does not contain endogenous regressors
    exog = add_constant(SIMULATED_DATA[["x3", "x4"]])
    first_stage = IVGMM(SIMULATED_DATA.y_robust, exog, None, None).fit().first_stage
    assert isinstance(first_stage.cragg_donald, InvalidTestStatistic)
    assert isinstance(first_stage.kleibergen_paap, InvalidTestStatistic)
    assert np.isnan(first_stage.cragg_donald_f)
    assert np.isnan(first_stage.kleibergen_paap_f)


def test_first_stage_kernel_default_bandwidth():
    # The model selects the bandwidth when it is not provided, and the
    # statistic uses the one that the model selected
    endog, controls, instr = simulated_inputs("B")
    mod = IV2SLS(SIMULATED_DATA.y_robust, add_constant(controls), endog, instr)
    res = mod.fit(cov_type="kernel")
    bandwidth = res.cov_config["bandwidth"]
    explicit = mod.fit(cov_type="kernel", bandwidth=bandwidth).first_stage
    first_stage = res.first_stage
    assert np.isfinite(first_stage.kleibergen_paap.stat)
    assert_allclose(first_stage.kleibergen_paap.stat, explicit.kleibergen_paap.stat)
    assert_allclose(first_stage.kleibergen_paap_f, explicit.kleibergen_paap_f)


def test_debiased_has_no_effect():
    endog, controls, instr = simulated_inputs("B")
    mod = IV2SLS(SIMULATED_DATA.y_robust, add_constant(controls), endog, instr)
    plain = mod.fit(cov_type="robust").first_stage
    debiased = mod.fit(cov_type="robust", debiased=True).first_stage
    assert_allclose(debiased.kleibergen_paap.stat, plain.kleibergen_paap.stat)
    assert_allclose(debiased.kleibergen_paap_f, plain.kleibergen_paap_f)


def test_first_stage_clusters_are_used():
    endog, controls, instr = simulated_inputs("B")
    mod = IV2SLS(SIMULATED_DATA.y_robust, add_constant(controls), endog, instr)
    robust = mod.fit(cov_type="robust").first_stage
    clustered = mod.fit(
        cov_type="clustered", clusters=SIMULATED_DATA.cluster_id
    ).first_stage
    assert robust.kleibergen_paap.stat != pytest.approx(clustered.kleibergen_paap.stat)
    # Nothing depends on the covariance for the Cragg-Donald statistic
    assert_allclose(robust.cragg_donald.stat, clustered.cragg_donald.stat)
