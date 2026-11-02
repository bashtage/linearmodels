import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from linearmodels.iv import IV2SLS, IVJIVE, IVLIML
from linearmodels.iv._tf import _tf_critical_value
from linearmodels.iv.results import FirstStageResults
from linearmodels.shared.exceptions import MissingValueWarning


@pytest.fixture
def data():
    rng = np.random.default_rng(98765)
    n = 200
    raw = np.column_stack([np.ones(n), rng.standard_normal((n, 4))])
    q, _ = np.linalg.qr(raw)
    w, z, v, e = q[:, :2], q[:, 2], q[:, 3], q[:, 4]
    z, v, e = (a * np.sqrt(n) for a in (z, v, e))
    x = np.sqrt(10 / n) * z + v
    u = 0.4 * np.sqrt(10) * (0.8 * v + 0.6 * e)
    y = x + w @ np.array([0.5, 2.0]) + u
    return y, w, x, z


@pytest.mark.parametrize(
    ("f_stat", "expected"),
    [
        (4, 18.656),
        (4.008, 18.236),
        (9.835, 3.463),
        (10.253, 3.385),
        (24.605, 2.461),
        (104.67, 1.96),
        (1e12, 1.96),
    ],
)
def test_published_values(f_stat, expected):
    # Table 3a, February 16, 2022 author version, page 16.
    assert _tf_critical_value(f_stat) == expected


def test_interpolation_in_f():
    # Section 3.2's worked F=10 example, using the critical-value row rather
    # than its separately rounded standard-error-adjustment row.
    expected = 3.463 + (10 - 9.835) / (10.253 - 9.835) * (3.385 - 3.463)
    assert_allclose(_tf_critical_value(10), expected, rtol=1e-14)
    assert _tf_critical_value(10) != 3.463


@pytest.mark.parametrize("f_stat", [0, 1e-15, 1.96**2, np.nextafter(4.0, 0)])
def test_below_table_is_unbounded(f_stat):
    assert _tf_critical_value(f_stat) == np.inf


@pytest.mark.parametrize("f_stat", [-1, -np.inf, np.inf, np.nan])
def test_invalid_f(f_stat):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _tf_critical_value(f_stat)


def test_monotonicity_and_table_boundaries():
    f = np.concatenate([np.linspace(4, 104.67, 10001), [104.67, 1e6]])
    critical = np.array([_tf_critical_value(x) for x in f])
    assert np.all(np.diff(critical) <= 0)
    assert np.all(critical >= 1.96)
    for knot, expected in [(4, 18.656), (9.835, 3.463), (104.67, 1.96)]:
        assert_allclose(_tf_critical_value(np.nextafter(knot, np.inf)), expected)
        if knot > 4:
            assert_allclose(_tf_critical_value(np.nextafter(knot, 0)), expected)


def test_hand_calculated_interval(data):
    y, w, x, z = data
    res = IV2SLS(y, w, x, z).fit(cov_type="unadjusted")
    # Orthogonal construction: first-stage F=10, beta=1, and IV SE=0.4.
    assert_allclose(res.params.iloc[-1], 1, atol=1e-13)
    assert_allclose(res.std_errors.iloc[-1], 0.4, rtol=1e-13)
    critical = 3.4322105263157896
    assert_allclose(res.tf_conf_int().iloc[0], [1 - 0.4 * critical, 1 + 0.4 * critical])
    assert res.conf_int().iloc[-1, 0] > 0
    assert res.tf_conf_int().iloc[0, 0] < 0


@pytest.mark.parametrize(
    "cov_type",
    [
        "unadjusted",
        "robust",
        "homoskedastic",
        "heteroskedastic",
        "homo",
        "hccm",
        "HomoskedasticCovariance",
        "HeteroskedasticCovariance",
    ],
)
@pytest.mark.parametrize("controls", [True, False])
def test_independent_covariance_calculation(data, cov_type, controls):
    y, w, x, z = data
    if not controls:
        w = np.empty((len(y), 0))
    res = IV2SLS(y, w, x, z).fit(cov_type=cov_type)
    design = np.column_stack([w, z])
    inverse = np.linalg.inv(design.T @ design)
    first = inverse @ design.T @ x
    v = x - design @ first
    z_resid = z - w @ np.linalg.lstsq(w, z, rcond=None)[0]
    beta = z_resid @ y / (z_resid @ x)
    u0 = y - beta * x
    u = u0 - w @ np.linalg.lstsq(w, u0, rcond=None)[0]
    if cov_type.lower() in (
        "unadjusted",
        "homoskedastic",
        "homo",
        "homoskedasticcovariance",
    ):
        first_var = (v @ v / len(v)) * inverse[-1, -1]
        variance = (u @ u / len(u)) * (z_resid @ z_resid) / (z_resid @ x) ** 2
    else:
        scores = design * v[:, None]
        first_var = (inverse @ scores.T @ scores @ inverse)[-1, -1]
        variance = ((z_resid * u) @ (z_resid * u)) / (z_resid @ x) ** 2
    f_stat = first[-1] ** 2 / first_var
    expected = beta + np.array([-1, 1]) * np.sqrt(variance) * _tf_critical_value(f_stat)
    assert_allclose(res.tf_conf_int().iloc[0], expected, rtol=1e-12, atol=1e-12)


def test_formula_labels_and_retained_sample(data):
    y, w, x, z = data
    frame = DataFrame({"y": y, "w": w[:, 1], "x": x, "z": z})
    frame.index = np.arange(len(frame)) * 3
    frame = frame.iloc[::2]
    res = IV2SLS.from_formula("y ~ 1 + w + [x ~ z]", frame).fit()
    array_res = IV2SLS(
        frame.y, np.column_stack([np.ones(len(frame)), frame.w]), frame.x, frame.z
    ).fit()
    assert_frame_equal(res.tf_conf_int(), array_res.tf_conf_int())
    assert list(res.tf_conf_int().index) == ["x"]
    assert list(res.tf_conf_int().columns) == ["lower", "upper"]


@pytest.mark.parametrize("instrument_scale", [-1000, -1, 1e-6, 1e6])
def test_instrument_scale_and_sign(data, instrument_scale):
    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int()
    actual = IV2SLS(y, w, x, z * instrument_scale).fit().tf_conf_int()
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("outcome_scale", "endog_scale"), [(2, 1), (-2, 1), (1, -3), (1e-4, 1e4)]
)
def test_coefficient_scale(data, outcome_scale, endog_scale):
    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int().to_numpy()
    expected = np.sort(expected * outcome_scale / endog_scale, axis=1)
    actual = IV2SLS(y * outcome_scale, w, x * endog_scale, z).fit().tf_conf_int()
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_row_permutation_and_no_mutation(data):
    y, w, x, z = data
    res = IV2SLS(y, w, x, z).fit()
    params, se, pvalues, conventional = (
        res.params.copy(),
        res.std_errors.copy(),
        res.pvalues.copy(),
        res.conf_int().copy(),
    )
    expected = res.tf_conf_int()
    actual = IV2SLS(y[::-1], w[::-1], x[::-1], z[::-1]).fit().tf_conf_int()
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert_series_equal(res.params, params)
    assert_series_equal(res.std_errors, se)
    assert_series_equal(res.pvalues, pvalues)
    assert_frame_equal(res.conf_int(), conventional)


def test_weak_first_stage(data):
    y, w, x, z = data
    weak_x = x - (np.sqrt(10 / len(y)) - 0.01) * z
    res = IV2SLS(y, w, weak_x, z).fit(cov_type="unadjusted")
    assert_array_equal(res.tf_conf_int(), [[-np.inf, np.inf]])


@pytest.mark.parametrize("model", [IVLIML, IVJIVE])
def test_unsupported_estimator(data, model):
    y, w, x, z = data
    with pytest.raises(ValueError, match="requires IV2SLS"):
        model(y, w, x, z).fit().tf_conf_int()


@pytest.mark.parametrize("two_endog", [True, False])
def test_unsupported_dimensions(data, two_endog):
    y, w, x, z = data
    instruments = np.column_stack([z, z**2])
    endog = np.column_stack([x, x**2]) if two_endog else x
    with pytest.raises(ValueError, match="one endogenous regressor"):
        IV2SLS(y, w, endog, instruments).fit().tf_conf_int()


@pytest.mark.parametrize(
    ("cov_type", "config"),
    [
        ("robust", {"debiased": True}),
        ("unadjusted", {"debiased": True}),
        ("clustered", {"clusters": np.repeat(np.arange(20), 10)}),
        ("kernel", {"bandwidth": 2}),
    ],
)
def test_unsupported_covariance(data, cov_type, config):
    y, w, x, z = data
    with pytest.raises(ValueError, match="unadjusted or robust covariance"):
        IV2SLS(y, w, x, z).fit(cov_type=cov_type, **config).tf_conf_int()


def test_weights(data):
    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int()
    actual = IV2SLS(y, w, x, z, weights=np.full(len(y), 7)).fit().tf_conf_int()
    assert_frame_equal(actual, expected)
    with pytest.raises(ValueError, match="nonconstant weights"):
        IV2SLS(y, w, x, z, weights=np.linspace(1, 2, len(y))).fit().tf_conf_int()


@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
def test_invalid_structural_variance(data, variance):
    y, w, x, z = data
    res = IV2SLS(y, w, x, z).fit()
    res.cov.iloc[-1, -1] = variance
    with pytest.raises(ValueError, match="finite estimates and positive variances"):
        res.tf_conf_int()


@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
def test_invalid_first_stage_variance(data, variance, monkeypatch):
    y, w, x, z = data
    res = IV2SLS(y, w, x, z).fit()
    first = res.first_stage.individual
    first[res.model.endog.cols[0]].cov.iloc[-1, -1] = variance
    monkeypatch.setattr(FirstStageResults, "individual", property(lambda self: first))
    with pytest.raises(ValueError, match="finite estimates and positive variances"):
        res.tf_conf_int()


def test_missing_observations(data):
    y, w, x, z = data
    y, x = y.copy(), x.copy()
    y[0], x[1] = np.nan, np.nan
    with pytest.warns(MissingValueWarning):
        res = IV2SLS(y, w, x, z).fit()
    complete = IV2SLS(y[2:], w[2:], x[2:], z[2:]).fit()
    assert_frame_equal(res.tf_conf_int(), complete.tf_conf_int())
