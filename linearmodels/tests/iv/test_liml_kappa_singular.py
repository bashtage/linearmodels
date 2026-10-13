"""
Tests of the kappa of LIML when the matrix that it inverts is singular

The IV models compute the kappa of LIML in fit. IVLIML estimates the model with it,
and IV2SLS and IVJIVE use it in the Anderson-Rubin and Basmann F tests of
overidentification. Kappa is the smallest generalized eigenvalue of (A, B) with
A = W'M_1W and B = W'M_ZW. W holds the dependent variable and the endogenous
regressors, M_1 annihilates the exogenous regressors and M_Z annihilates the
exogenous regressors and the instruments.

B has a rank of at most nobs - nz, where nz is the number of exogenous regressors
and instruments, and W has 1 + nendog columns. So B is singular if there are fewer
than 1 + nendog observations more than nz, and kappa is not defined. It is also
singular if the columns of M_ZW are collinear for another reason.

The model inverted the square root of B with an eigenvalue decomposition, which
does not raise if B is singular. Kappa was then the rounding error of B, which is
arbitrary and depends on the platform: 0.8125 instead of 1 for a just-identified
model with 3 observations, a constant, an endogenous regressor and an instrument,
or NaN, with a RuntimeWarning, if an eigenvalue of B was negative. The estimates
of IVLIML and its covariance were NaN then, and IV2SLS and IVJIVE computed the
Anderson-Rubin and Basmann F tests from the rounding error.

fit has to raise the ValueError that it has for a kappa that cannot be estimated
if kappa is not given, set kappa of IV2SLS and IVJIVE to NaN so that the tests
that use it are invalid, and not emit a warning, on every platform. Whether B is
singular cannot depend on the units of the variables, and so a column of W that
is rescaled by 1e6 or 1e-6 does not change whether kappa is defined.

Where kappa is defined it is computed as it was. The accuracy of kappa depends on
the units of the variables if there are 2 or more endogenous regressors, and a
regressor that is exactly a linear combination of the exogenous regressors and
instruments is not detected (issue 823). The tests pin the values, which are
noise in the second case, to those of the previous computation.

The expected values are computed from the definition. The projections come from
QR decompositions and kappa from scipy.linalg.eigh(A, B), so that none of the
code of the models is used. 2SLS and k-class estimates, and JIVE with the first
stage estimated n times, are also computed from their definitions.
"""

import re
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose
import pytest
from scipy import stats
from scipy.linalg import eigh

from linearmodels.iv import IV2SLS, IVJIVE, IVLIML
import linearmodels.iv.model
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.shared.linalg import inv_sqrth

# ----------------------------------------------------------------------------
# Data and the definitions of the expected values
# ----------------------------------------------------------------------------


def make_data(nobs, nexog, nendog, ninstr, seed, constant=True):
    """Data where the first exogenous regressor is a constant if there is one"""
    rng = np.random.default_rng(seed)
    exog = rng.standard_normal((nobs, nexog))
    if constant and nexog:
        exog[:, 0] = 1.0
    instr = rng.standard_normal((nobs, ninstr))
    u = rng.standard_normal(nobs)
    pi = rng.uniform(0.5, 1.0, (ninstr, nendog))
    endog = instr @ pi + rng.standard_normal((nobs, nendog)) + 0.5 * u[:, None]
    y = endog.sum(1) + exog.sum(1) + u
    return y, exog, endog, instr


def args(y, exog, endog, instr):
    """The arguments of the models, which have None for an empty exog"""
    return y, (exog if exog.shape[1] else None), endog, instr


def residuals(regressors, variables):
    """Residuals of the columns of variables on the columns of regressors"""
    if regressors.shape[1] == 0:
        return variables
    q, _ = np.linalg.qr(regressors)
    return variables - q @ (q.T @ variables)


def cross_products(y, exog, endog, instr, weights=None):
    """A = W'M_1W and B = W'M_ZW of the definition, using the weighted data"""
    w = np.column_stack([y, endog])
    z = np.column_stack([exog, instr])
    if weights is not None:
        root = np.sqrt(weights)[:, None]
        w, z, exog = w * root, z * root, exog * root
    r1 = residuals(exog, w)
    rz = residuals(z, w)
    return r1.T @ r1, rz.T @ rz


def reference_kappa(y, exog, endog, instr, weights=None):
    a, b = cross_products(y, exog, endog, instr, weights)
    return float(eigh(a, b, eigvals_only=True).min())


def k_class(y, exog, endog, instr, kappa):
    """k-class estimator from its definition: (X'(I - k M_Z)X)^-1 X'(I - k M_Z)y"""
    x = np.column_stack([exog, endog])
    z = np.column_stack([exog, instr])
    annihilator = residuals(z, np.eye(y.shape[0]))
    m = np.eye(y.shape[0]) - kappa * annihilator
    return np.linalg.solve(x.T @ m @ x, x.T @ m @ y)


def reference_jive(y, exog, endog, instr):
    """JIVE from its definition: the first stage is estimated without row i, n times"""
    x = np.column_stack([exog, endog])
    z = np.column_stack([exog, instr])
    nobs = y.shape[0]
    x_loo = np.empty_like(x)
    for i in range(nobs):
        keep = np.arange(nobs) != i
        coef = np.linalg.lstsq(z[keep], x[keep], rcond=None)[0]
        x_loo[i] = z[i] @ coef
    return np.linalg.solve(x_loo.T @ x, x_loo.T @ y)


def fewest_observations(nexog, nendog, ninstr):
    """1 + nendog observations more than the exogenous regressors and instruments"""
    return nexog + ninstr + 1 + nendog


def design_label(design):
    nexog, nendog, ninstr, nobs = design
    return f"exog{nexog}-endog{nendog}-instr{ninstr}-n{nobs}"


# (nexog, nendog, ninstr), the first exogenous regressor is a constant. Just
# identified and overidentified models with up to 3 endogenous regressors, and
# models without exogenous regressors
SIZES = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 3),
    (2, 1, 2),
    (2, 2, 2),
    (1, 2, 3),
    (2, 2, 4),
    (1, 3, 4),
    (1, 3, 5),
    (0, 1, 1),
    (0, 1, 2),
    (0, 2, 3),
]

# One observation short of the fewest, where kappa is not defined
TOO_FEW = [(*s, fewest_observations(*s) - 1) for s in SIZES]
# As many observations as the columns of Z, where the residuals on Z are 0
SATURATED = [(*s, s[0] + s[2]) for s in SIZES]
# The fewest observations for which kappa is defined: the residuals have as many
# dimensions as columns
FEWEST = [(*s, fewest_observations(*s)) for s in SIZES]
LARGE = [(*s, 150) for s in SIZES]
DEFINED = LARGE + FEWEST

# The Anderson-Rubin and Basmann F tests need more instruments than endogenous
# regressors. The others are invalid for that reason, and do not use kappa
OVERIDENTIFIED_TOO_FEW = [d for d in TOO_FEW if d[2] > d[1]]
OVERIDENTIFIED_LARGE = [d for d in LARGE if d[2] > d[1]]
JUST_IDENTIFIED = [d for d in DEFINED if d[2] == d[1]]

MODELS = [IVLIML, IV2SLS, IVJIVE]


def ids(designs):
    return [design_label(d) for d in designs]


def data_for(design, seed=0, constant=True):
    nexog, nendog, ninstr, nobs = design
    seed = seed + 10 * nendog
    return make_data(nobs, nexog, nendog, ninstr, seed=seed, constant=constant)


# ----------------------------------------------------------------------------
# Kappa is unchanged if it is defined
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("design", DEFINED, ids=ids(DEFINED))
@pytest.mark.parametrize("model", MODELS)
def test_kappa_is_the_generalized_eigenvalue(model, design):
    y, exog, endog, instr = data_for(design)
    expected = reference_kappa(y, exog, endog, instr)
    res = model(*args(y, exog, endog, instr)).fit()
    liml_kappa = res._liml_kappa
    assert_allclose(liml_kappa, expected, rtol=1e-8)
    if model is IVLIML:
        assert_allclose(res.kappa, expected, rtol=1e-8)
    # M_1 - M_Z is positive semi-definite, so that kappa is at least 1
    assert liml_kappa >= 1 - 1e-8


@pytest.mark.parametrize("design", JUST_IDENTIFIED, ids=ids(JUST_IDENTIFIED))
def test_kappa_of_just_identified_models_is_one(design):
    # Closed form: the objective of LIML is 1 where the model is exactly identified
    y, exog, endog, instr = data_for(design)
    kappa = IVLIML(*args(y, exog, endog, instr)).fit().kappa
    assert_allclose(kappa, 1.0, rtol=1e-8)


@pytest.mark.parametrize("design", OVERIDENTIFIED_LARGE, ids=ids(OVERIDENTIFIED_LARGE))
def test_overidentification_tests_use_kappa(design):
    nexog, nendog, ninstr, nobs = design
    y, exog, endog, instr = data_for(design)
    kappa = reference_kappa(y, exog, endog, instr)
    res = IV2SLS(*args(y, exog, endog, instr)).fit()
    q = ninstr - nendog
    df_denom = nobs - nexog - ninstr
    ar = res.anderson_rubin
    assert isinstance(ar, WaldTestStatistic)
    assert_allclose(ar.stat, nobs * np.log(kappa), rtol=1e-6, atol=1e-10)
    assert_allclose(ar.pval, stats.chi2.sf(ar.stat, q), rtol=1e-8)
    basmann_f = res.basmann_f
    assert isinstance(basmann_f, WaldTestStatistic)
    expected = (kappa - 1) * df_denom / q
    assert_allclose(basmann_f.stat, expected, rtol=1e-6, atol=1e-10)
    assert_allclose(basmann_f.pval, stats.f.sf(expected, q, df_denom), rtol=1e-6)


def test_weighted_kappa_is_the_generalized_eigenvalue():
    design = (1, 2, 4, 90)
    y, exog, endog, instr = data_for(design)
    weights = np.random.default_rng(5).uniform(0.4, 2.5, y.shape[0])
    expected = reference_kappa(y, exog, endog, instr, weights)
    res = IVLIML(*args(y, exog, endog, instr), weights=weights).fit()
    assert_allclose(res.kappa, expected, rtol=1e-8)


# ----------------------------------------------------------------------------
# Whether kappa is defined does not depend on the units of the variables
# ----------------------------------------------------------------------------

# Scales of (y, endogenous regressors). The scales are interleaved, which matters for
# the cross product that is inverted. The accuracy of kappa depends on the units if
# there are 2 or more endogenous regressors (issue 823). Those scales test that a
# singular model is singular in any units, and that kappa is computed as it was
SCALES_ONE_ENDOG = [(1e6, 1), (1, 1e6), (1e-6, 1), (1, 1e-6), (1e6, 1e-6), (1e-6, 1e6)]
SCALES_TWO_ENDOG = [
    (1e6, 1, 1),
    (1, 1e6, 1),
    (1, 1, 1e6),
    (1e-6, 1, 1),
    (1, 1e-6, 1),
    (1, 1, 1e-6),
    (1e6, 1e-6, 1),
    (1e-6, 1, 1e6),
    (1e3, 1e-3, 1e-3),
    (1e6, 1e6, 1e-6),
]
SCALES = SCALES_ONE_ENDOG + SCALES_TWO_ENDOG


def scaled(design, scales, seed=3):
    y, exog, endog, instr = data_for(design, seed=seed)
    return y * scales[0], exog, endog * np.array(scales[1:]), instr


@pytest.mark.parametrize(
    "scales", SCALES_ONE_ENDOG, ids=[str(s) for s in SCALES_ONE_ENDOG]
)
@pytest.mark.parametrize("model", MODELS)
def test_kappa_with_one_endogenous_regressor_does_not_depend_on_the_units(
    model, scales
):
    # This is not true of 2 or more endogenous regressors, see issue 823
    design = (1, 1, 3, 120)
    expected = reference_kappa(*data_for(design, seed=3))
    res = model(*args(*scaled(design, scales))).fit()
    assert_allclose(res._liml_kappa, expected, rtol=1e-9)


@pytest.mark.parametrize("scales", SCALES, ids=[str(s) for s in SCALES])
def test_too_few_observations_are_singular_in_any_units(scales):
    # Rescaling a column cannot make the singular cross product look non-singular
    nendog = len(scales) - 1
    design = (1, nendog, nendog + 1, fewest_observations(1, nendog, nendog + 1) - 1)
    y, exog, endog, instr = scaled(design, scales)
    with pytest.raises(ValueError, match="Unable to estimate kappa"):
        IVLIML(*args(y, exog, endog, instr)).fit()


# ----------------------------------------------------------------------------
# Kappa is computed as it was where the model is not singular
# ----------------------------------------------------------------------------


def previous_kappa(model):
    """The previous computation: the same operations on the residuals as they are"""
    y, x, z = model._wy, model._wx, model._wz
    is_exog = model._regressor_is_exog
    e = np.c_[y, x[:, ~is_exog]]
    x1 = x[:, is_exog]
    ez = e - z @ (np.linalg.pinv(z) @ e)
    ex1 = e if x1.shape[1] == 0 else e - x1 @ (np.linalg.pinv(x1) @ e)
    vpmzv_sqinv = inv_sqrth(ez.T @ ez)
    q = vpmzv_sqinv @ (ex1.T @ ex1) @ vpmzv_sqinv
    return min(np.linalg.eigvalsh(q))


def current_kappa(model):
    return model._estimate_kappa()


def kappa_or_none(function, model):
    try:
        return function(model)
    except LinAlgError:
        return None


def check_computed_as_before(model):
    # Either both raise, or the values are the same. The accuracy of both is poor
    # in the cases of issue 823, where the values are not the kappa of the data
    kappa = kappa_or_none(current_kappa, model)
    previous = kappa_or_none(previous_kappa, model)
    kappa_raised = kappa is None
    previous_raised = previous is None
    assert kappa_raised == previous_raised
    if not previous_raised:
        assert_allclose(kappa, previous, rtol=1e-12)


@pytest.mark.parametrize("design", DEFINED, ids=ids(DEFINED))
def test_kappa_is_computed_as_before(design):
    y, exog, endog, instr = data_for(design)
    check_computed_as_before(IVLIML(*args(y, exog, endog, instr), kappa=1.0))


def span_data(seed):
    """A regressor that is exactly a linear combination of the instruments"""
    rng = np.random.default_rng(seed)
    nobs = 500
    instr = rng.standard_normal((nobs, 2))
    endog = instr[:, 0] + 2 * instr[:, 1]
    y = 1 + 0.5 * endog + rng.standard_normal(nobs)
    return y, np.ones((nobs, 1)), endog[:, None], instr


@pytest.mark.parametrize("seed", range(8))
def test_kappa_of_a_regressor_in_the_span_of_the_instruments_is_computed_as_before(
    seed,
):
    # The residuals of the regressor are rounding error, and so is kappa (issue 823),
    # which is near 1. Dividing the residuals by their norms magnified the rounding
    # error by 1e14, and kappa was -5.5e10 where it is 1.002657 (seed 0, Windows)
    check_computed_as_before(IVLIML(*span_data(seed), kappa=1.0))


@pytest.mark.parametrize(
    "scales", SCALES_TWO_ENDOG, ids=[str(s) for s in SCALES_TWO_ENDOG]
)
def test_kappa_of_regressors_on_different_scales_is_computed_as_before(scales):
    # The values are not the kappa of the data if the scales differ much (issue 823)
    design = (1, 2, 4, 120)
    check_computed_as_before(IVLIML(*args(*scaled(design, scales)), kappa=1.0))


# ----------------------------------------------------------------------------
# IVLIML raises if kappa is not defined
# ----------------------------------------------------------------------------


def bug_design(nobs):
    """A constant, an endogenous regressor and an instrument: 3 observations were bad"""
    rng = np.random.default_rng(11)
    z1, z2, exogenous, noise_x, noise_y = rng.standard_normal((5, nobs))
    endog = 0.8 * z1 + 0.3 * z2 + noise_x
    y = 1 + endog + exogenous + noise_y
    return y, np.ones((nobs, 1)), endog[:, None], z1[:, None]


def test_liml_raises_in_the_example_of_the_bug():
    # The matrix has an eigenvalue of 5.6e-17 and kappa was 0.8125 instead of 1, or
    # NaN where the eigenvalue is negative. 4 observations are the fewest
    with pytest.raises(ValueError, match=re.escape("It needs at least 4 observations")):
        IVLIML(*bug_design(3)).fit()
    y, exog, endog, instr = bug_design(4)
    expected = reference_kappa(y, exog, endog, instr)
    kappa = IVLIML(y, exog, endog, instr).fit().kappa
    assert_allclose(kappa, expected, rtol=1e-8)
    assert_allclose(kappa, 1.0, rtol=1e-8)


def expected_message(design):
    nexog, nendog, ninstr, nobs = design
    needed = fewest_observations(nexog, nendog, ninstr)
    return (
        f"Kappa cannot be estimated using {nobs} observations. It needs at least "
        f"{needed} observations"
    )


@pytest.mark.parametrize(
    "design", TOO_FEW + SATURATED, ids=ids(TOO_FEW) + ids(SATURATED)
)
def test_liml_raises_if_kappa_is_not_defined(design):
    y, exog, endog, instr = data_for(design)
    mod = IVLIML(*args(y, exog, endog, instr))
    with pytest.raises(ValueError, match="Unable to estimate kappa") as info:
        mod.fit()
    message = str(info.value)
    assert expected_message(design) in message
    # It is the error for a kappa that cannot be estimated
    assert "if the instrument matrix is rank deficient" in message


@pytest.mark.parametrize("design", TOO_FEW, ids=ids(TOO_FEW))
def test_liml_with_fuller_raises_if_kappa_is_not_defined(design):
    y, exog, endog, instr = data_for(design)
    mod = IVLIML(*args(y, exog, endog, instr), fuller=1)
    with pytest.raises(ValueError, match="Unable to estimate kappa"):
        mod.fit()


@pytest.mark.parametrize(
    "design", TOO_FEW + SATURATED, ids=ids(TOO_FEW) + ids(SATURATED)
)
def test_estimate_kappa_raises_linalg_error(design):
    # IVJIVE catches a LinAlgError, and fit of the other models any exception
    y, exog, endog, instr = data_for(design)
    mod = IVLIML(*args(y, exog, endog, instr), kappa=1.0)
    with pytest.raises(LinAlgError, match=re.escape(expected_message(design))):
        mod._estimate_kappa()


def test_weighted_liml_raises_if_kappa_is_not_defined():
    design = TOO_FEW[7]
    y, exog, endog, instr = data_for(design)
    weights = np.random.default_rng(9).uniform(0.4, 2.5, y.shape[0])
    mod = IVLIML(*args(y, exog, endog, instr), weights=weights)
    with pytest.raises(ValueError, match="Unable to estimate kappa"):
        mod.fit()


@pytest.mark.parametrize("seed", range(8))
def test_count_of_observations_does_not_depend_on_the_instruments_conditioning(seed):
    # A constant and 2 instruments leave 1 dimension of 4 observations for the 2
    # residuals. The projections on instruments that are nearly collinear with the
    # constant (the condition number is about 5e8) are mostly rounding error, and
    # the residuals can look as if they have full rank. The count does not depend
    # on it
    nobs = 4
    rng = np.random.default_rng(seed)
    instr = 1.0 + 1e-8 * rng.standard_normal((nobs, 2))
    u = rng.standard_normal(nobs)
    endog = rng.standard_normal(nobs) + 0.5 * u + instr @ np.array([0.8, 0.5])
    y = endog + 1 + u
    mod = IVLIML(y, np.ones((nobs, 1)), endog[:, None], instr)
    with pytest.raises(ValueError, match="Unable to estimate kappa"):
        mod.fit()


@pytest.mark.parametrize("design", FEWEST, ids=ids(FEWEST))
def test_liml_fits_with_the_fewest_observations(design):
    # One more observation than TOO_FEW
    y, exog, endog, instr = data_for(design)
    res = IVLIML(*args(y, exog, endog, instr)).fit()
    params_are_finite = np.all(np.isfinite(res.params))
    kappa_is_finite = np.isfinite(res.kappa)
    assert params_are_finite
    assert kappa_is_finite


# ----------------------------------------------------------------------------
# A kappa that is given still fits
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.7, 1.0, 1.3])
@pytest.mark.parametrize("design", TOO_FEW, ids=ids(TOO_FEW))
def test_liml_with_kappa_fits_if_kappa_is_not_defined(design, kappa):
    y, exog, endog, instr = data_for(design)
    res = IVLIML(*args(y, exog, endog, instr), kappa=kappa).fit()
    expected = k_class(y, exog, endog, instr, kappa)
    assert_allclose(res.params.to_numpy(), expected, rtol=1e-6, atol=1e-8)
    liml_kappa_is_nan = np.isnan(res._liml_kappa)
    assert res.kappa == kappa
    assert liml_kappa_is_nan


@pytest.mark.parametrize(
    "design", OVERIDENTIFIED_TOO_FEW, ids=ids(OVERIDENTIFIED_TOO_FEW)
)
def test_liml_with_kappa_has_invalid_overidentification_tests(design):
    y, exog, endog, instr = data_for(design)
    res = IVLIML(*args(y, exog, endog, instr), kappa=1.0).fit()
    for test in (res.anderson_rubin, res.basmann_f):
        assert isinstance(test, InvalidTestStatistic)


# ----------------------------------------------------------------------------
# IV2SLS and IVJIVE are estimated, and kappa of LIML is NaN
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("design", TOO_FEW, ids=ids(TOO_FEW))
def test_2sls_sets_kappa_of_liml_to_nan(design):
    y, exog, endog, instr = data_for(design)
    res = IV2SLS(*args(y, exog, endog, instr)).fit()
    liml_kappa_is_nan = np.isnan(res._liml_kappa)
    assert liml_kappa_is_nan
    assert res.kappa == 1
    expected = k_class(y, exog, endog, instr, 1.0)
    assert_allclose(res.params.to_numpy(), expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("design", TOO_FEW, ids=ids(TOO_FEW))
def test_jive_sets_kappa_of_liml_to_nan(design):
    y, exog, endog, instr = data_for(design)
    res = IVJIVE(*args(y, exog, endog, instr)).fit()
    liml_kappa_is_nan = np.isnan(res._liml_kappa)
    assert liml_kappa_is_nan
    assert res.kappa is None
    expected = reference_jive(y, exog, endog, instr)
    assert_allclose(res.params.to_numpy(), expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("model", [IV2SLS, IVJIVE])
@pytest.mark.parametrize(
    "design", OVERIDENTIFIED_TOO_FEW, ids=ids(OVERIDENTIFIED_TOO_FEW)
)
def test_overidentification_tests_are_invalid_if_kappa_is_not_defined(model, design):
    y, exog, endog, instr = data_for(design)
    res = model(*args(y, exog, endog, instr)).fit()
    for test in (res.anderson_rubin, res.basmann_f):
        stat_is_nan = np.isnan(test.stat)
        pval_is_nan = np.isnan(test.pval)
        reason = str(test)
        assert isinstance(test, InvalidTestStatistic)
        assert stat_is_nan
        assert pval_is_nan
        assert test.critical_values is None
        # The reason is the missing kappa, not the number of instruments
        assert "kappa of LIML" in reason
        assert "more instruments" not in reason


@pytest.mark.parametrize("model", [IV2SLS, IVJIVE])
def test_just_identified_tests_keep_their_reason(model):
    y, exog, endog, instr = bug_design(3)
    res = model(y, exog, endog, instr).fit()
    for test in (res.anderson_rubin, res.basmann_f):
        reason = str(test)
        assert isinstance(test, InvalidTestStatistic)
        assert "more instruments than endogenous variables" in reason


@pytest.mark.parametrize("model", [IV2SLS, IVJIVE])
def test_tests_that_do_not_use_kappa_are_unchanged(model):
    design = (1, 1, 2, 4)
    y, exog, endog, instr = data_for(design)
    res = model(*args(y, exog, endog, instr)).fit()
    assert isinstance(res.sargan, WaldTestStatistic)
    assert isinstance(res.wooldridge_overid, WaldTestStatistic)


# ----------------------------------------------------------------------------
# No warnings
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "design", TOO_FEW + SATURATED, ids=ids(TOO_FEW) + ids(SATURATED)
)
def test_no_warnings_if_kappa_is_not_defined(design):
    # An eigenvalue of the singular matrix that is negative gave a RuntimeWarning
    # (invalid value encountered in sqrt), and one that is 0 a divide by zero
    y, exog, endog, instr = data_for(design)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Unable to estimate kappa"):
            IVLIML(*args(y, exog, endog, instr)).fit()
        IVLIML(*args(y, exog, endog, instr), kappa=1.0).fit()
        IV2SLS(*args(y, exog, endog, instr)).fit()


def fail_if_decomposed(matrix):
    raise AssertionError("The singular cross product was decomposed")


@pytest.mark.parametrize(
    "design", TOO_FEW + SATURATED, ids=ids(TOO_FEW) + ids(SATURATED)
)
def test_too_few_observations_are_detected_before_the_decomposition(
    design, monkeypatch
):
    # The sign of an eigenvalue that is rounding error depends on the platform, and
    # was negative on macOS arm64. A matrix that is singular because of the number of
    # observations is not decomposed, so that nothing depends on it
    monkeypatch.setattr(linearmodels.iv.model, "inv_sqrth", fail_if_decomposed)
    y, exog, endog, instr = data_for(design)
    mod = IVLIML(*args(y, exog, endog, instr), kappa=1.0)
    with pytest.raises(LinAlgError, match="Kappa cannot be estimated"):
        mod._estimate_kappa()


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_collinear_residuals_are_detected_before_the_decomposition(scale, monkeypatch):
    monkeypatch.setattr(linearmodels.iv.model, "inv_sqrth", fail_if_decomposed)
    y, exog, endog, instr = collinear_data(scale)
    mod = IVLIML(*args(y, exog, endog, instr), kappa=1.0)
    with pytest.raises(LinAlgError, match="linearly dependent"):
        mod._estimate_kappa()


@pytest.mark.parametrize("design", TOO_FEW, ids=ids(TOO_FEW))
def test_no_warnings_from_jive_if_kappa_is_not_defined(design):
    y, exog, endog, instr = data_for(design)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IVJIVE(*args(y, exog, endog, instr)).fit()


@pytest.mark.parametrize(
    "design", OVERIDENTIFIED_TOO_FEW, ids=ids(OVERIDENTIFIED_TOO_FEW)
)
def test_no_warnings_from_the_tests_if_kappa_is_not_defined(design):
    y, exog, endog, instr = data_for(design)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = IV2SLS(*args(y, exog, endog, instr)).fit()
        # str of the tests does not warn
        anderson_rubin, basmann_f = str(res.anderson_rubin), str(res.basmann_f)
    assert "kappa of LIML" in anderson_rubin
    assert "kappa of LIML" in basmann_f


@pytest.mark.parametrize("design", DEFINED, ids=ids(DEFINED))
def test_no_warnings_if_kappa_is_defined(design):
    y, exog, endog, instr = data_for(design)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for model in MODELS:
            model(*args(y, exog, endog, instr)).fit()


# ----------------------------------------------------------------------------
# Residuals that are collinear for another reason
# ----------------------------------------------------------------------------


def collinear_data(scale=1.0, nobs=80):
    """x2 = 2 x1 + Z b, so that the residuals of x1 and x2 on Z are collinear"""
    y, exog, endog, instr = make_data(nobs, 1, 1, 3, seed=5)
    z = np.column_stack([exog, instr])
    x2 = 2 * endog[:, 0] + z @ np.array([0.3, 1.0, -2.0, 0.5])
    return y, exog, np.column_stack([endog[:, 0], x2 * scale]), instr


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_collinear_residuals_are_singular(scale):
    y, exog, endog, instr = collinear_data(scale)
    mod = IVLIML(*args(y, exog, endog, instr))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Unable to estimate kappa"):
            mod.fit()
        with pytest.raises(LinAlgError, match="linearly dependent"):
            mod._estimate_kappa()
        res = IV2SLS(*args(y, exog, endog, instr)).fit()
    liml_kappa_is_nan = np.isnan(res._liml_kappa)
    assert liml_kappa_is_nan
    assert isinstance(res.anderson_rubin, InvalidTestStatistic)


def test_collinear_residuals_fit_with_kappa():
    y, exog, endog, instr = collinear_data()
    res = IVLIML(*args(y, exog, endog, instr), kappa=1.0).fit()
    expected = k_class(y, exog, endog, instr, 1.0)
    assert_allclose(res.params.to_numpy(), expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("delta", [1e-1, 1e-3])
def test_nearly_collinear_residuals_are_not_singular(delta):
    # The cross product is ill conditioned, and kappa is accurate
    y, exog, endog, instr = make_data(200, 1, 2, 4, seed=7)
    noise = np.random.default_rng(1).standard_normal(200)
    endog = np.column_stack([endog[:, 0], endog[:, 0] + delta * noise])
    expected = reference_kappa(y, exog, endog, instr)
    kappa = IVLIML(*args(y, exog, endog, instr)).fit().kappa
    assert_allclose(kappa, expected, rtol=1e-6)


def test_zero_dependent_variable_is_singular():
    # The residuals of a variable that is 0 are 0, a column with norm 0
    _, exog, endog, instr = make_data(60, 1, 1, 2, seed=2)
    y = np.zeros(60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Unable to estimate kappa"):
            IVLIML(*args(y, exog, endog, instr)).fit()
        res = IV2SLS(*args(y, exog, endog, instr)).fit()
    liml_kappa_is_nan = np.isnan(res._liml_kappa)
    assert liml_kappa_is_nan
    assert isinstance(res.anderson_rubin, InvalidTestStatistic)


def test_ols_with_as_many_observations_as_regressors_does_not_warn():
    # The residuals are 0. The model uses kappa = 0, and only the kappa of LIML,
    # which is not defined, is affected
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 4))
    y = rng.standard_normal(4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = IV2SLS(y, x, None, None).fit()
    assert_allclose(res.params.to_numpy(), np.linalg.solve(x, y), rtol=1e-8)
