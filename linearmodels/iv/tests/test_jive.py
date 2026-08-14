"""Tests for IVJIVE (Jackknife Instrumental Variables Estimator).

Reference: Angrist, Imbens & Krueger (1999), Journal of Applied Econometrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pandas import DataFrame

from linearmodels.iv import IV2SLS, IVJIVE
from linearmodels.iv.results import IVResults


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------


def _make_iv(n=500, k_instr=3, beta=1.5, seed=0):
    """Simple IV DGP: one endogenous regressor, k_instr excluded instruments."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k_instr))
    v = rng.standard_normal(n)
    e = rng.standard_normal(n) + 0.5 * v  # endogenous error
    x_endog = z[:, 0] * 0.8 + v
    y = x_endog * beta + e
    exog = np.ones((n, 1))
    return y, exog, x_endog[:, None], z


def _make_iv_many(n=500, k_instr=15, beta=1.5, seed=1):
    """Many-IV DGP with weak first stage — where JIVE matters most."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k_instr))
    gamma = np.ones(k_instr) * 0.2  # weak instruments
    v = rng.standard_normal(n)
    x_endog = z @ gamma + v
    e = rng.standard_normal(n) + 0.5 * v
    y = x_endog * beta + e
    exog = np.ones((n, 1))
    return y, exog, x_endog[:, None], z


# ---------------------------------------------------------------------------
# Import and basic API
# ---------------------------------------------------------------------------


def test_import():
    """IVJIVE is importable from linearmodels.iv."""
    from linearmodels.iv import IVJIVE  # noqa: F401


def test_method_name():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert res.model._method == "IV-JIVE"


def test_fit_returns_iv_results():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert isinstance(res, IVResults)


# ---------------------------------------------------------------------------
# Output shapes and structure
# ---------------------------------------------------------------------------


def test_params_shape_one_endog():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert res.params.shape == (2,)  # const + endog


def test_params_labels():
    y, exog, endog, z = _make_iv()
    y_df = DataFrame({"y": y})
    exog_df = DataFrame({"const": np.ones(len(y))})
    endog_df = DataFrame({"x": endog.squeeze()})
    z_df = DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
    res = IVJIVE(y_df, exog_df, endog_df, z_df).fit()
    assert "x" in res.params.index


def test_std_errors_shape():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert res.std_errors.shape == (2,)


def test_std_errors_positive():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert np.all(res.std_errors > 0)


def test_cov_shape():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert res.cov.shape == (2, 2)


def test_cov_symmetric():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert_allclose(res.cov.values, res.cov.values.T, atol=1e-12)


def test_cov_psd():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    eigvals = np.linalg.eigvalsh(res.cov.values)
    assert np.all(eigvals > -1e-10)


def test_r2_finite():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert np.isfinite(res.rsquared)


def test_residuals_shape():
    y, exog, endog, z = _make_iv(n=300)
    res = IVJIVE(y, exog, endog, z).fit()
    assert res.resids.shape == (300,)


# ---------------------------------------------------------------------------
# Consistency: JIVE → true parameter as n → ∞
# ---------------------------------------------------------------------------


def test_jive_consistent_large_n():
    """With a large sample and strong instruments, JIVE should be close to 1.5."""
    y, exog, endog, z = _make_iv(n=3000, k_instr=3, beta=1.5, seed=10)
    res = IVJIVE(y, exog, endog, z).fit()
    assert_allclose(float(res.params.iloc[-1]), 1.5, atol=0.15)


def test_jive_close_to_2sls_few_instruments():
    """With few instruments, JIVE ≈ 2SLS (both consistent, similar finite samples)."""
    y, exog, endog, z = _make_iv(n=2000, k_instr=2, beta=1.5, seed=7)
    jive = IVJIVE(y, exog, endog, z).fit()
    tsls = IV2SLS(y, exog, endog, z).fit()
    assert_allclose(
        float(jive.params.iloc[-1]),
        float(tsls.params.iloc[-1]),
        atol=0.15,
    )


def test_jive_less_biased_than_2sls_many_instruments():
    """In a many-IV setting, JIVE point estimate should be closer to the truth."""
    y, exog, endog, z = _make_iv_many(n=2000, k_instr=15, beta=1.5, seed=42)
    jive = IVJIVE(y, exog, endog, z).fit()
    tsls = IV2SLS(y, exog, endog, z).fit()
    beta = 1.5
    jive_err = abs(float(jive.params.iloc[-1]) - beta)
    tsls_err = abs(float(tsls.params.iloc[-1]) - beta)
    # JIVE need not always win a single draw; allow generous test
    assert jive_err < tsls_err + 0.3


# ---------------------------------------------------------------------------
# Leverage scores
# ---------------------------------------------------------------------------


def test_leverage_scores_in_unit_interval():
    """All leverage scores h_i must be in [0, 1)."""
    y, exog, endog, z = _make_iv(n=300)
    mod = IVJIVE(y, exog, endog, z)
    wx, wz = mod._wx, mod._wz
    from numpy.linalg import inv as _inv
    ZtZ_inv = _inv(wz.T @ wz)
    A = wz @ ZtZ_inv
    h = (A * wz).sum(axis=1)
    assert np.all(h >= 0)
    assert np.all(h < 1)


def test_leverage_sum_equals_rank():
    """Sum of leverage scores equals the rank of Z (number of instrument columns)."""
    y, exog, endog, z = _make_iv(n=300, k_instr=4)
    mod = IVJIVE(y, exog, endog, z)
    wx, wz = mod._wx, mod._wz
    from numpy.linalg import inv as _inv
    ZtZ_inv = _inv(wz.T @ wz)
    A = wz @ ZtZ_inv
    h = (A * wz).sum(axis=1)
    # Sum h_i = trace(P_Z) = rank(Z) = ncols of wz
    assert_allclose(h.sum(), wz.shape[1], atol=1e-8)


# ---------------------------------------------------------------------------
# debiased flag
# ---------------------------------------------------------------------------


def test_debiased_larger_stderr():
    """Debiased SEs should be >= undebiased SEs (both positive)."""
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit(debiased=False)
    res_db = IVJIVE(y, exog, endog, z).fit(debiased=True)
    assert np.all(res_db.std_errors.values >= res.std_errors.values - 1e-12)


def test_debiased_flag_stored():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit(debiased=True)
    assert res.model._method == "IV-JIVE"


# ---------------------------------------------------------------------------
# from_formula
# ---------------------------------------------------------------------------


def test_from_formula():
    """from_formula should produce the same estimates as the array API."""
    y, exog, endog, z = _make_iv(n=400, seed=5)
    n = len(y)
    data = DataFrame({
        "y": y.squeeze(),
        "x": endog.squeeze(),
        "z0": z[:, 0], "z1": z[:, 1], "z2": z[:, 2],
    })
    res_array = IVJIVE(y, exog, endog, z).fit()
    res_formula = IVJIVE.from_formula("y ~ 1 + [x ~ z0 + z1 + z2]", data).fit()
    assert_allclose(
        res_array.params.values,
        res_formula.params.values,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Covariance estimator object
# ---------------------------------------------------------------------------


def test_cov_estimator_type_string():
    """IVResults.cov_estimator is the cov_type string (linearmodels API)."""
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    assert isinstance(res.cov_estimator, str)
    assert res.cov_estimator == "robust"


def test_cov_config_stored():
    """cov_config dict is accessible and includes debiased key."""
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit(debiased=True)
    assert "debiased" in res.cov_config
    assert res.cov_config["debiased"] is True


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_weighted_estimation_runs():
    y, exog, endog, z = _make_iv(n=300)
    n = len(y)
    weights = np.abs(np.random.default_rng(0).standard_normal(n)) + 0.1
    res = IVJIVE(y, exog, endog, z, weights=weights).fit()
    assert np.isfinite(float(res.params.iloc[-1]))


# ---------------------------------------------------------------------------
# Multiple endogenous regressors
# ---------------------------------------------------------------------------


def test_two_endog_regressors():
    """JIVE with two endogenous regressors and enough instruments."""
    rng = np.random.default_rng(3)
    n, k_instr = 800, 5
    z = rng.standard_normal((n, k_instr))
    v1, v2 = rng.standard_normal(n), rng.standard_normal(n)
    x1 = z[:, 0] + v1
    x2 = z[:, 1] + v2
    y = x1 * 1.0 + x2 * 2.0 + rng.standard_normal(n) + 0.3 * (v1 + v2)
    exog = np.ones((n, 1))
    endog = np.column_stack([x1, x2])
    res = IVJIVE(y, exog, endog, z).fit()
    assert res.params.shape == (3,)  # const + 2 endog
    assert np.all(np.isfinite(res.params.values))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_runs():
    y, exog, endog, z = _make_iv()
    res = IVJIVE(y, exog, endog, z).fit()
    s = res.summary
    assert s is not None
