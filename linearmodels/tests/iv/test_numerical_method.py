"""
Tests of the numerical method of the k-class IV models (IV2SLS and IVLIML).

method="pinv", the default, tests the rank of [exog, endog] and [exog, instruments]
as they are, projects with pinv and inverts the cross products of the k-class
estimator. method="qr" tests the rank of the columns scaled to unit length, with a
relative tolerance of 1e-9, and computes the estimates, their covariance and the
kappa of LIML from QR decompositions.

The exact values come from data built from the columns of a Hadamard matrix, which
are orthogonal. With columns h0, ..., h5 of H,

    exog = [h0, h1], instruments = [h2, h3], endog = h1 + delta (h2 + h4),
    y = h0 + h1 + endog + h5

The error h5 is orthogonal to the instruments and to the regressors, so that every
k-class estimate is exactly (1, 1, 1), and the projection of the regressors on the
instruments is [h0, h1, h1 + delta h2]. The residual of the projected problem of 2SLS
is 0, so that the rounding error of QR is about the condition number times the machine
precision. With kappa < 1, the k-class estimator also fits y - Xb, and with
kappa > 1 it subtracts the fit of M_z(y - Xb). Both have a large residual, and the
error of any backward stable method grows with the square of the condition number
times the norm of the residual. Those estimates are tested without the error,
y = X(1, 1, 1)'. Whether the rounding error of the products of the Hadamard data happens
to cancel depends on the order of the sums in BLAS, so that the tests also permute the
rows. The entries are dyadic, so that the data are
exact in float64, and the covariances are computed exactly with fractions. delta makes
the regressors nearly collinear: the condition number of the scaled regressors is
about 2.4 / delta.
"""

from fractions import Fraction
import inspect
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.linalg import eigh, hadamard

from linearmodels.formula import iv_2sls, iv_liml
from linearmodels.iv import IV2SLS, IVGMM, IVLIML
from linearmodels.iv._utility import annihilate, kclass_factors, orthonormal_basis, proj
from linearmodels.iv.common import (
    cragg_donald,
    cragg_donald_f,
    kleibergen_paap,
    kleibergen_paap_f,
)
from linearmodels.iv.covariance import (
    ClusteredCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    KernelCovariance,
)
from linearmodels.iv.model import _rank_and_condition_number
from linearmodels.iv.results import FirstStageResults

NOBS = 64
# A fixed assignment: contiguous blocks of the rows of H have scores that sum to 0
CLUSTERS = np.random.default_rng(0).permutation(NOBS) // 8
BANDWIDTH = 3


def hadamard_data(delta, error=True):
    h = hadamard(NOBS).astype(float)
    endog = h[:, 1] + delta * (h[:, 2] + h[:, 4])
    y = h[:, 0] + h[:, 1] + endog + (h[:, 5] if error else 0.0)
    return y, h[:, :2], endog[:, None], h[:, 2:4]


def exact_cov(delta, cov_type):
    """The covariance of 2SLS of the Hadamard data, as linearmodels defines it"""
    h = hadamard(NOBS)
    d = Fraction(delta)
    xhat = [
        [Fraction(int(h[i, 0])), Fraction(int(h[i, 1])), h[i, 1] + d * h[i, 2]]
        for i in range(NOBS)
    ]
    e = [Fraction(int(h[i, 5])) for i in range(NOBS)]
    k = 3

    def outer(a, b):
        return [[a[r] * b[c] for c in range(k)] for r in range(k)]

    def add(a, b, w=1):
        return [[a[r][c] + w * b[r][c] for c in range(k)] for r in range(k)]

    def mul(a, b):
        return [
            [sum(a[r][j] * b[j][c] for j in range(k)) for c in range(k)]
            for r in range(k)
        ]

    xpx = [
        [sum(xhat[i][r] * xhat[i][c] for i in range(NOBS)) for c in range(k)]
        for r in range(k)
    ]
    # Gauss-Jordan inverse, exact
    aug = [
        row[:] + [Fraction(int(r == c)) for c in range(k)] for r, row in enumerate(xpx)
    ]
    for col in range(k):
        piv = next(r for r in range(col, k) if aug[r][col] != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(k):
            if r != col:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col], strict=True)]
    a_inv = [row[k:] for row in aug]
    scores = [[e[i] * xhat[i][c] for c in range(k)] for i in range(NOBS)]
    zero = [[Fraction(0)] * k for _ in range(k)]
    if cov_type == "unadjusted":
        s2 = sum(v * v for v in e) / NOBS
        return np.array([[float(s2 * v) for v in row] for row in a_inv])
    meat = zero
    if cov_type in ("robust", "kernel"):
        for s in scores:
            meat = add(meat, outer(s, s))
    if cov_type == "kernel":
        for j in range(1, BANDWIDTH + 1):
            w = 1 - Fraction(j, BANDWIDTH + 1)
            for i in range(NOBS - j):
                op = outer(scores[i + j], scores[i])
                meat = add(meat, add(op, [list(r) for r in zip(*op, strict=True)]), w)
    if cov_type == "clustered":
        for g in np.unique(CLUSTERS):
            xi = [
                sum(scores[i][c] for i in np.flatnonzero(CLUSTERS == g))
                for c in range(k)
            ]
            meat = add(meat, outer(xi, xi))
    cov = mul(mul(a_inv, meat), a_inv)
    return np.array([[float(v) for v in row] for row in cov])


COV_OPTIONS = {
    "unadjusted": {},
    "robust": {},
    "clustered": {"clusters": CLUSTERS},
    "kernel": {"kernel": "bartlett", "bandwidth": BANDWIDTH},
}
# 2^-20 and 2^-27 are about 9.5e-7 and 7.5e-9. The condition numbers of the scaled
# regressors are about 2.5e6 and 3.2e8, so that the error of QR is about 1e-10 and
# 1e-8. method="pinv" is off by about 1e-4 and 300% (Windows, OpenBLAS)
DELTAS = {"2^-20": (2.0**-20, 1e-8), "2^-27": (2.0**-27, 1e-6)}


@pytest.mark.parametrize("delta", DELTAS.values(), ids=list(DELTAS))
@pytest.mark.parametrize(
    "options",
    [{}, {"kappa": 0.0}, {"kappa": 0.5}, {"kappa": 1.25}],
    ids=["liml", "ols", "kappa-0.5", "kappa-1.25"],
)
@pytest.mark.parametrize("permute", [False, True], ids=["rows", "permuted"])
def test_kclass_estimates_are_exact(delta, options, permute):
    delta, rtol = delta
    # A fixed kappa other than 1 also fits a residual of y - Xb, which is 0 without
    # the error. The estimated kappa of LIML is 1 up to rounding with the error
    error = "kappa" not in options
    data = hadamard_data(delta, error)
    if permute:
        rows = np.random.default_rng(0).permutation(NOBS)
        data = tuple(a[rows] for a in data)
    res = IVLIML(*data, method="qr", **options).fit()
    assert_allclose(res.params.to_numpy(), np.ones(3), rtol=rtol)


def test_fuller_estimate_is_accurate():
    # kappa is 0.983 and y - Xb has a residual of norm 8, so that the error is bounded
    # by about (1 - kappa) cond(X)^2 eps, 2.4e-5 here. It is at most 1.6e-6 over 40
    # orders of the rows (Windows, OpenBLAS). method="pinv" is 1.2e-4 off
    res = IVLIML(*hadamard_data(2.0**-20), method="qr", fuller=1.0).fit()
    assert_allclose(res.kappa, 1 - 1 / (NOBS - 4), rtol=1e-12)
    assert_allclose(res.params.to_numpy(), np.ones(3), rtol=5e-5)


@pytest.mark.parametrize("delta", DELTAS.values(), ids=list(DELTAS))
@pytest.mark.parametrize("cov_type", list(COV_OPTIONS))
def test_2sls_estimates_and_covariance_are_exact(delta, cov_type):
    delta, rtol = delta
    mod = IV2SLS(*hadamard_data(delta), method="qr")
    res = mod.fit(cov_type=cov_type, **COV_OPTIONS[cov_type])
    assert_allclose(res.params.to_numpy(), np.ones(3), rtol=rtol)
    expected = exact_cov(delta, cov_type)
    se = np.sqrt(np.diag(expected))
    assert_allclose(res.std_errors.to_numpy(), se, rtol=rtol)
    # Relative to the standard errors, since some covariances are 0
    scale = np.outer(se, se)
    assert_allclose(res.cov.to_numpy() / scale, expected / scale, rtol=0, atol=rtol)


# ----------------------------------------------------------------------------
# The rank check and the default
# ----------------------------------------------------------------------------


def test_nearly_collinear_regressors():
    # A condition number of about 1e10, which the check of method="pinv" accepts
    data = hadamard_data(2.0**-33)
    with pytest.raises(ValueError, match=r"regressors .* collinear"):
        IV2SLS(*data, method="qr")
    with pytest.raises(ValueError, match=r"regressors .* collinear"):
        IVLIML(*data, method="qr")
    # The warning of the default also says that method="qr" raises for these data
    with pytest.warns(FutureWarning, match="regressors .* nearly collinear"):
        IV2SLS(*data)
    with pytest.warns(FutureWarning, match='method="qr" raises a ValueError'):
        IVLIML(*data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IV2SLS(*data, method="pinv")


def formula_frame(data):
    y, exog, endog, instr = data
    return pd.DataFrame(
        {
            "y": y,
            "h0": exog[:, 0],
            "h1": exog[:, 1],
            "w": endog[:, 0],
            "h2": instr[:, 0],
            "h3": instr[:, 1],
        }
    )


def test_nearly_collinear_instruments():
    y, exog, endog, instr = hadamard_data(2.0**-10)
    instr = instr.copy()
    instr[:, 1] = instr[:, 0] + 2.0**-33 * hadamard(NOBS)[:, 7]
    with pytest.raises(ValueError, match=r"instruments .* collinear"):
        IV2SLS(y, exog, endog, instr, method="qr")
    with pytest.warns(FutureWarning, match="instruments .* nearly collinear"):
        IV2SLS(y, exog, endog, instr)


def test_exactly_collinear_or_zero_columns_are_rank_deficient():
    y, exog, endog, instr = hadamard_data(2.0**-10)
    with pytest.raises(ValueError, match="regressors"):
        IV2SLS(y, exog, exog[:, 1:], instr, method="qr")
    zero = instr.copy()
    zero[:, 1] = 0.0
    with pytest.raises(ValueError, match="instruments"):
        IV2SLS(y, exog, endog, zero, method="qr")


def scaled_data():
    rng = np.random.default_rng(1)
    nobs = 500
    instr = rng.standard_normal((nobs, 3))
    v = rng.standard_normal((nobs, 2))
    x = rng.standard_normal((nobs, 2))
    endog = instr @ rng.uniform(0.5, 1.0, (3, 2)) + v + 0.3 * x[:, :1]
    exog = np.c_[np.ones(nobs), x]
    u = rng.standard_normal(nobs) + v[:, 0] + 0.5 * v[:, 1]
    y = exog @ np.array([1, 0.5, -0.5]) + endog @ np.array([1, -1]) + u
    return y, exog, endog, instr


def test_unscaled_rank_check_depends_on_the_units():
    y, exog, endog, instr = scaled_data()
    instr = instr * np.array([2.0**50, 1, 1])
    with pytest.raises(ValueError, match=r'do not have full column rank.*method="qr"'):
        IV2SLS(y, exog, endog, instr)
    IV2SLS(y, exog, endog, instr, method="qr").fit()


def test_default_is_pinv():
    data = scaled_data()
    for model in (IV2SLS, IVLIML):
        with pytest.warns(FutureWarning, match="default numerical method"):
            default = model(*data).fit(cov_type="robust")
        pinv = model(*data, method="pinv").fit(cov_type="robust")
        assert_allclose(default.params, pinv.params, rtol=1e-14)
        assert_allclose(default.cov, pinv.cov, rtol=1e-14)
        assert default.model._numerical_method == "pinv"


@pytest.mark.parametrize("method", ["QR", "Pinv"])
def test_method_is_not_case_sensitive(method):
    mod = IV2SLS(*scaled_data(), method=method)
    assert mod._numerical_method == method.lower()


@pytest.mark.parametrize(
    "func",
    [
        IV2SLS,
        IVLIML,
        IV2SLS.from_formula,
        IVLIML.from_formula,
        IVLIML.estimate_parameters,
        HomoskedasticCovariance,
        HeteroskedasticCovariance,
        KernelCovariance,
        ClusteredCovariance,
        FirstStageResults,
        cragg_donald,
        cragg_donald_f,
        kleibergen_paap,
        kleibergen_paap_f,
        proj,
        annihilate,
    ],
    ids=lambda func: func.__qualname__,
)
def test_method_is_keyword_only(func):
    parameter = inspect.signature(func).parameters["method"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("method", ["svd", 1, ""])
def test_invalid_method(method):
    with pytest.raises(ValueError, match="method must be"):
        IV2SLS(*scaled_data(), method=method)
    y, exog, endog, instr = scaled_data()
    x = np.c_[exog, endog]
    z = np.c_[exog, instr]
    with pytest.raises(ValueError, match="method must be"):
        HomoskedasticCovariance(x, y[:, None], z, np.zeros((5, 1)), method=method)


# ----------------------------------------------------------------------------
# qr agrees with pinv for well-conditioned data
# ----------------------------------------------------------------------------

MODELS = {
    "2sls": (IV2SLS, {}),
    "liml": (IVLIML, {}),
    "fuller": (IVLIML, {"fuller": 4.0}),
    "kappa-0.5": (IVLIML, {"kappa": 0.5}),
}


def statistics(res):
    out = {
        name: getattr(res, name).stat
        for name in (
            "sargan",
            "basmann",
            "basmann_f",
            "wooldridge_score",
            "wooldridge_regression",
            "wooldridge_overid",
            "anderson_rubin",
        )
    }
    first_stage = res.first_stage
    out["cragg_donald"] = first_stage.cragg_donald.stat
    out["kleibergen_paap"] = first_stage.kleibergen_paap.stat
    out["durbin"] = res.durbin().stat
    out["wu_hausman"] = res.wu_hausman().stat
    out["kappa"] = res._liml_kappa
    out["rsquared"] = res.rsquared
    diagnostics = first_stage.diagnostics
    for stat in ("rsquared", "partial.rsquared", "shea.rsquared", "f.stat"):
        for i, value in enumerate(diagnostics[stat]):
            out[f"first stage {stat} {i}"] = value
    return out


@pytest.mark.parametrize("cov_type", list(COV_OPTIONS))
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("debiased", [False, True])
def test_qr_agrees_with_pinv(model, cov_type, debiased):
    cls, options = MODELS[model]
    data = scaled_data()
    cov_options = dict(COV_OPTIONS[cov_type])
    if cov_type == "clustered":
        cov_options["clusters"] = np.arange(data[0].shape[0]) // 10
    pinv = cls(*data, method="pinv", **options).fit(
        cov_type=cov_type, debiased=debiased, **cov_options
    )
    qr = cls(*data, method="qr", **options).fit(
        cov_type=cov_type, debiased=debiased, **cov_options
    )
    assert_allclose(qr.params, pinv.params, rtol=1e-10)
    assert_allclose(qr.cov, pinv.cov, rtol=1e-9)
    expected = statistics(pinv)
    for key, value in statistics(qr).items():
        assert_allclose(value, expected[key], rtol=1e-8, err_msg=key)


# ----------------------------------------------------------------------------
# qr does not depend on the units of the variables
# ----------------------------------------------------------------------------

P = 2.0**60
# Scaling by a power of 2 is exact, so that the data are the same up to the units.
# (scaled data, multiplier of the parameters)
UNITS = {
    "dependent": (lambda y, x, w, z: (y * P, x, w, z), np.full(5, P)),
    "exog": (lambda y, x, w, z: (y, x * [1, P, 1], w, z), [1, 1 / P, 1, 1, 1]),
    "endog": (lambda y, x, w, z: (y, x, w * [1, 1 / P], z), [1, 1, 1, 1, P]),
    "instruments": (lambda y, x, w, z: (y, x, w, z * [P, 1, 1 / P]), np.ones(5)),
}


@pytest.mark.parametrize("units", list(UNITS))
@pytest.mark.parametrize("cov_type", list(COV_OPTIONS))
@pytest.mark.parametrize("model", list(MODELS))
def test_qr_does_not_depend_on_the_units(model, cov_type, units):
    # The automatic kernel bandwidth uses the sum of the scores, which depends on the
    # units by definition, and so a bandwidth is given
    cls, options = MODELS[model]
    cov_options = dict(COV_OPTIONS[cov_type])
    data = scaled_data()
    if cov_type == "clustered":
        cov_options["clusters"] = np.arange(data[0].shape[0]) // 10
    scale, multiplier = UNITS[units]
    base = cls(*data, method="qr", **options).fit(cov_type=cov_type, **cov_options)
    res = cls(*scale(*data), method="qr", **options).fit(
        cov_type=cov_type, **cov_options
    )
    multiplier = np.asarray(multiplier)
    assert_allclose(
        res.params.to_numpy(), base.params.to_numpy() * multiplier, rtol=1e-10
    )
    assert_allclose(
        res.std_errors.to_numpy(), base.std_errors.to_numpy() * multiplier, rtol=1e-10
    )
    expected = statistics(base)
    for key, value in statistics(res).items():
        assert_allclose(value, expected[key], rtol=1e-8, err_msg=key)


def reference_kappa(y, exog, endog, instr):
    w = np.c_[y, endog]

    def resid(a, b):
        q, _ = np.linalg.qr(b)
        return a - q @ (q.T @ a)

    e1 = resid(w, exog)
    ez = resid(w, np.c_[exog, instr])
    return eigh(e1.T @ e1, ez.T @ ez, eigvals_only=True)[0]


@pytest.mark.parametrize("scales", [(1e6, 1, 1), (1, 1e-6, 1), (1e6, 1e-6, 1e3)])
def test_kappa_with_two_endogenous_regressors_does_not_depend_on_the_units(scales):
    # Case 2 of issue 823: the kappa of method="pinv" changes by up to 2.3e-4
    y, exog, endog, instr = scaled_data()
    expected = reference_kappa(y, exog, endog, instr)
    y = y * scales[0]
    endog = endog * np.array(scales[1:])
    res = IVLIML(y, exog, endog, instr, method="qr").fit()
    assert_allclose(res.kappa, expected, rtol=1e-10)


# ----------------------------------------------------------------------------
# The factors of the k-class estimator and the covariance estimators
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.4, 1.0, 1.25])
def test_kclass_factors(kappa):
    y, exog, endog, instr = scaled_data()
    x = np.c_[exog, endog]
    z = np.c_[exog, instr]
    y = y[:, None]
    q = orthonormal_basis(z)
    mz_x = x - q @ (q.T @ x)
    mz_y = y - q @ (q.T @ y)
    xpx = x.T @ x - kappa * (mz_x.T @ mz_x)
    xpy = x.T @ y - kappa * (mz_x.T @ mz_y)
    factors = kclass_factors(x, q, kappa, y)
    g = np.eye(x.shape[1]) if factors.g is None else factors.g
    assert_allclose(factors.r.T @ g @ factors.r, xpx, rtol=1e-10, atol=1e-8)
    assert_allclose(factors.r.T @ factors.t, xpy, rtol=1e-10, atol=1e-8)
    xhat = q @ (q.T @ x)
    assert_allclose(q @ factors.u @ factors.r, xhat, rtol=1e-10, atol=1e-10)
    assert_allclose(np.triu(factors.r), factors.r, rtol=0, atol=0)


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0, 1.1])
@pytest.mark.parametrize(
    ("estimator", "options"),
    [
        (HomoskedasticCovariance, {}),
        (HeteroskedasticCovariance, {}),
        (KernelCovariance, {}),
        (KernelCovariance, {"bandwidth": 6}),
        (ClusteredCovariance, {"clusters": np.arange(500) // 25}),
    ],
)
def test_covariance_estimators_agree(estimator, options, kappa):
    y, exog, endog, instr = scaled_data()
    x = np.c_[exog, endog]
    z = np.c_[exog, instr]
    y = y[:, None]
    params = IVLIML.estimate_parameters(x, y, z, kappa)
    pinv = estimator(x, y, z, params, kappa=kappa, **options)
    qr = estimator(x, y, z, params, kappa=kappa, method="qr", **options)
    assert_allclose(qr.cov, pinv.cov, rtol=1e-9)
    assert_allclose(qr.s, pinv.s, rtol=1e-9)
    params_qr = IVLIML.estimate_parameters(x, y, z, kappa, method="qr")
    assert_allclose(params_qr, params, rtol=1e-10)


def test_kclass_factors_of_ols_do_not_need_the_instruments():
    y, exog, endog, instr = scaled_data()
    x = np.c_[exog, endog]
    z = np.c_[exog, instr]
    y = y[:, None]
    with_basis = kclass_factors(x, orthonormal_basis(z), 0.0, y)
    without = kclass_factors(x, None, 0.0, y)
    assert without.u is None
    assert_allclose(without.r, with_basis.r, rtol=1e-14)
    assert_allclose(without.t, with_basis.t, rtol=1e-14)
    with pytest.raises(ValueError, match="z_basis is required"):
        kclass_factors(x, None, 0.5, y)


@pytest.mark.parametrize(
    ("estimator", "options"),
    [
        (HomoskedasticCovariance, {}),
        (HeteroskedasticCovariance, {}),
        (KernelCovariance, {"bandwidth": 6}),
        (ClusteredCovariance, {"clusters": np.arange(500) // 25}),
    ],
)
def test_ols_covariance_with_z_equal_to_x(estimator, options):
    # z is x in OLS, and one QR decomposition of x is used for both
    y, exog, endog, _ = scaled_data()
    x = np.c_[exog, endog]
    y = y[:, None]
    params = np.linalg.lstsq(x, y, rcond=None)[0]
    pinv = estimator(x, y, x, params, kappa=0.0, **options)
    qr = estimator(x, y, x.copy(), params, kappa=0.0, method="qr", **options)
    assert_allclose(qr.cov, pinv.cov, rtol=1e-10)
    assert_allclose(qr.s, pinv.s, rtol=1e-10)


@pytest.mark.parametrize("scale", [1.0, 2.0**30, 2.0**-40])
def test_rank_and_condition_number(scale):
    _, exog, endog, _ = scaled_data()
    x = np.c_[exog, endog * np.array([1, scale])]
    rank, cond = _rank_and_condition_number(x)
    assert rank == np.linalg.matrix_rank(x)
    assert_allclose(cond, np.linalg.cond(x), rtol=1e-8)
    deficient = np.c_[x, x[:, 1] + x[:, 2]]
    assert _rank_and_condition_number(deficient)[0] == np.linalg.matrix_rank(deficient)
    assert _rank_and_condition_number(x[:3])[1] == np.inf


def test_no_collinearity_note_for_columns_on_different_scales():
    # The condition number is about 1e9, from the units only. The columns scaled to
    # unit length are well conditioned, so that the warning has no note
    y, exog, endog, instr = scaled_data()
    with pytest.warns(FutureWarning, match="default numerical method") as record:
        res = IV2SLS(y, exog * np.array([1, 2.0**30, 1]), endog, instr).fit()
    assert len(record) == 1
    assert "nearly collinear" not in str(record[0].message)
    expected = IV2SLS(y, exog, endog, instr, method="qr").fit()
    assert_allclose(res.params.iloc[1] * 2.0**30, expected.params.iloc[1], rtol=1e-6)


DEFAULT_WARNING = (
    'The default numerical method of IV2SLS and IVLIML will change from "pinv" to '
    '"qr" in linearmodels 8.0.'
)


@pytest.mark.parametrize(
    "create",
    [
        lambda data: IV2SLS(*data),
        lambda data: IVLIML(*data),
        lambda data: IVLIML(*data, fuller=1.0),
        lambda data: IV2SLS.from_formula(
            "y ~ 0 + h0 + h1 + [w ~ h2 + h3]", formula_frame(data)
        ),
        lambda data: IVLIML.from_formula(
            "y ~ 0 + h0 + h1 + [w ~ h2 + h3]", formula_frame(data)
        ),
        lambda data: iv_2sls("y ~ 0 + h0 + h1 + [w ~ h2 + h3]", formula_frame(data)),
        lambda data: iv_liml("y ~ 0 + h0 + h1 + [w ~ h2 + h3]", formula_frame(data)),
    ],
    ids=[
        "IV2SLS",
        "IVLIML",
        "Fuller",
        "IV2SLS formula",
        "IVLIML formula",
        "iv_2sls",
        "iv_liml",
    ],
)
def test_the_default_warns_of_the_change_in_8(create):
    data = hadamard_data(2.0**-10)
    with pytest.warns(FutureWarning) as record:
        create(data)
    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith(DEFAULT_WARNING)
    # How to get the new method now and how to keep the current one
    assert 'Use method="qr" to use it now' in message
    assert (
        'method="pinv" to keep the current method and silence this warning' in message
    )
    assert "nearly collinear" not in message
    # The warning points at the caller
    assert record[0].filename == __file__


@pytest.mark.parametrize("method", ["pinv", "qr", "PINV"])
def test_an_explicit_method_does_not_warn(method):
    data = hadamard_data(2.0**-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IV2SLS(*data, method=method).fit()
        IVLIML(*data, method=method).fit()
        res = IV2SLS.from_formula(
            "y ~ 0 + h0 + h1 + [w ~ h2 + h3]", formula_frame(data), method=method
        ).fit()
        # The diagnostics use the method of the model, and do not warn either
        diagnostics = res.first_stage.diagnostics
        tests = (res.wooldridge_regression, res.durbin(), res.wu_hausman())
    assert diagnostics.shape[0] == 1
    assert all(np.isfinite(test.stat) for test in tests)


def test_models_without_the_method_keyword_do_not_warn():
    data = hadamard_data(2.0**-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IVGMM(*data).fit()
