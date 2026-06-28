"""
Tests for the Hansen-Lee (2021) misspecification-robust J-test.

References
----------
Hansen, B. E. & Lee, S. (2021). Inference for iterated GMM under
misspecification. Econometrica, 89(3), 1419-1447.
"""
import numpy as np
from numpy.linalg import pinv
from numpy.testing import assert_allclose
import pytest

from linearmodels.iv.gmm import HeteroskedasticWeightMatrix
from linearmodels.iv.model import IVGMM, IVGMMCUE
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.tests.iv._utility import generate_data


@pytest.fixture(scope="module")
def data():
    return generate_data()


@pytest.fixture(scope="module")
def res_robust(data):
    return IVGMM(data.dep, data.exog, data.endog, data.instr).fit(
        cov_type="robust"
    )


@pytest.fixture(scope="module")
def res_unadjusted(data):
    return IVGMM(data.dep, data.exog, data.endog, data.instr).fit(
        cov_type="unadjusted"
    )


@pytest.fixture(scope="module")
def res_clustered(data):
    return IVGMM(
        data.dep, data.exog, data.endog, data.instr
    ).fit(cov_type="clustered", clusters=data.clusters)


@pytest.fixture(scope="module")
def res_kernel(data):
    return IVGMM(data.dep, data.exog, data.endog, data.instr).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=4
    )


# ---------------------------------------------------------------------------
# Attribute existence and types
# ---------------------------------------------------------------------------


def test_robust_j_stat_is_wald_test_statistic(res_robust):
    assert isinstance(res_robust.robust_j_stat, WaldTestStatistic)


def test_standard_j_stat_still_present(res_robust):
    assert isinstance(res_robust.j_stat, WaldTestStatistic)


def test_robust_j_stat_stat_is_finite(res_robust):
    assert np.isfinite(res_robust.robust_j_stat.stat)


def test_robust_j_stat_pval_in_unit_interval(res_robust):
    pval = res_robust.robust_j_stat.pval
    assert 0.0 <= pval <= 1.0


def test_robust_j_stat_df_equals_overidentification_degree(res_robust, data):
    ninstr = data.instr.shape[1] + data.exog.shape[1]
    nendog = data.endog.shape[1]
    expected_df = ninstr - nendog - data.exog.shape[1]
    # df = total instruments - total params = (nexog+ninstr) - (nexog+nendog)
    #    = ninstr - nendog
    actual_df = res_robust.robust_j_stat.df
    assert actual_df == data.instr.shape[1] - data.endog.shape[1]


def test_robust_j_stat_stat_positive(res_robust):
    assert res_robust.robust_j_stat.stat >= 0.0


# ---------------------------------------------------------------------------
# Formula verification: J* = n * g_bar' @ S_c^{-1} @ g_bar
# ---------------------------------------------------------------------------


def test_robust_j_stat_formula_matches_manual_calculation(res_robust, data):
    """Verify the statistic equals the explicit closed-form expression."""
    mod = IVGMM(data.dep, data.exog, data.endog, data.instr)
    res = mod.fit(cov_type="robust")

    nobs = data.dep.shape[0]
    params = np.asarray(res.params)[:, None]
    x = mod._wx
    z = mod._wz
    y = mod._wy

    eps = y - x @ params
    g_bar = (z * eps).mean(0)

    sc_est = HeteroskedasticWeightMatrix(center=True)
    s_c = sc_est.weight_matrix(x, z, eps)
    expected = float(nobs * g_bar @ pinv(s_c) @ g_bar)

    assert_allclose(res.robust_j_stat.stat, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Under correct specification: J* ≈ J (both are chi2_q consistent)
# ---------------------------------------------------------------------------


def test_robust_and_standard_j_have_same_df(res_robust):
    """Both tests have the same degrees of freedom under correct specification."""
    assert res_robust.j_stat.df == res_robust.robust_j_stat.df


def test_robust_j_stat_nonnegative_all_cov_types(
    res_robust, res_unadjusted, res_clustered, res_kernel
):
    for res in [res_robust, res_unadjusted, res_clustered, res_kernel]:
        assert res.robust_j_stat.stat >= 0.0


def test_robust_j_pval_all_cov_types(
    res_robust, res_unadjusted, res_clustered, res_kernel
):
    for res in [res_robust, res_unadjusted, res_clustered, res_kernel]:
        pval = res.robust_j_stat.pval
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Under misspecification: J* diverges, standard J saturates
# ---------------------------------------------------------------------------


def test_robust_j_algebraic_identity_iterated_gmm():
    """
    For iterated GMM at convergence the identity

        J* = J_std * n / (n - J_std)

    follows from Ŝ = Ŝ_c + g_bar * g_bar' (Sherman-Morrison inversion).

    DGP: y = 2*x + z_invalid + noise where z_invalid enters the outcome
    directly, making the IV moment condition violated — both test statistics
    are large enough to make the identity observable.
    """
    rng = np.random.default_rng(42)
    nobs = 50_000

    z_valid = rng.standard_normal(nobs)    # valid instrument
    z_invalid = rng.standard_normal(nobs)  # invalid: correlated with eps
    z_extra = rng.standard_normal(nobs)    # extra valid instrument
    noise_x = rng.standard_normal(nobs)
    noise_y = rng.standard_normal(nobs)

    x = 0.8 * z_valid + 0.6 * noise_x     # endog, identified by z_valid
    y = 2.0 * x + 1.0 * z_invalid + noise_y  # z_invalid in outcome -> misspecified

    exog = np.ones((nobs, 1))
    endog = x[:, None]
    instr = np.column_stack([z_invalid, z_valid, z_extra])

    res = IVGMM(y[:, None], exog, endog, instr).fit(cov_type="robust", iter_limit=50)

    j_std = res.j_stat.stat
    j_star = res.robust_j_stat.stat

    expected_j_star = j_std * nobs / (nobs - j_std)
    assert_allclose(j_star, expected_j_star, rtol=1e-4)

    # Under misspecification both are large; J_std < n (bounded), J* >= J_std
    assert j_std < nobs
    assert j_star >= j_std


# ---------------------------------------------------------------------------
# IVGMMCUE also exposes robust_j_stat
# ---------------------------------------------------------------------------


def test_ivgmmcue_has_robust_j_stat(data):
    res = IVGMMCUE(data.dep, data.exog, data.endog, data.instr).fit(
        cov_type="robust"
    )
    assert isinstance(res.robust_j_stat, WaldTestStatistic)
    assert np.isfinite(res.robust_j_stat.stat)
    assert res.robust_j_stat.stat >= 0.0


# ---------------------------------------------------------------------------
# Summary display includes HL J-statistic rows
# ---------------------------------------------------------------------------


def test_summary_contains_hl_j_statistic(res_robust):
    smry_str = str(res_robust.summary)
    assert "HL J-statistic" in smry_str


def test_summary_contains_j_statistic(res_robust):
    smry_str = str(res_robust.summary)
    assert "J-statistic" in smry_str
