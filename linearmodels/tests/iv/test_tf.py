from types import ModuleType

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from scipy.special import ndtri

from linearmodels.iv import IV2SLS, IVJIVE, IVLIML, _tf
from linearmodels.iv._tf import _tf_critical_value
from linearmodels.iv.results import FirstStageResults
from linearmodels.shared.exceptions import MissingValueWarning


@pytest.fixture
def tf_package():
    return pytest.importorskip("tf_inference")


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
    ("f_stat", "printed"),
    [(4, 18.656), (4.008, 18.236), (9.835, 3.463), (10.253, 3.385), (24.605, 2.461)],
)
def test_published_rounding_bounds(tf_package, f_stat, printed):
    # Table 3a rounds both F and the critical value upward to three decimals.
    assert _tf_critical_value(f_stat) <= printed + 0.0001
    assert _tf_critical_value(f_stat - 0.001) >= printed - 0.0011


def test_numerical_values(tf_package):
    assert_allclose(_tf_critical_value(3.9), 30.92820746, rtol=1e-9)
    assert_allclose(_tf_critical_value(10), 3.43035370, rtol=1e-9)
    assert_allclose(_tf_critical_value(1e12), 1.95996398454, rtol=1e-11)
    assert_allclose(_tf_critical_value(1e12, 0.99), 2.72548606, rtol=2e-9)


@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.1, 0.0001, 0.25])
def test_exact_boundary(tf_package, alpha):
    q = float(ndtri(alpha / 2) ** 2)
    for f_stat in [0, np.nextafter(q, 0), q]:
        assert _tf_critical_value(f_stat, 1 - alpha) == np.inf
    f_stat = np.nextafter(q, np.inf)
    critical = _tf_critical_value(f_stat, 1 - alpha)
    assert np.isfinite(critical)
    assert_allclose(critical**2 * (f_stat - q), q**3, rtol=1e-12)


@pytest.mark.parametrize("f_stat", [-1, -np.inf, np.inf, np.nan])
def test_invalid_f(f_stat):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _tf_critical_value(f_stat)


def test_monotonicity_and_plateau(tf_package):
    q = float(ndtri(0.025) ** 2)
    f = q + np.geomspace(1e-12, 1e6, 1001)
    critical = np.array([_tf_critical_value(x) for x in f])
    assert np.all(np.diff(critical) <= 0)
    assert np.all(critical >= -ndtri(0.025) - 1e-14)


@pytest.mark.parametrize(
    ("level", "critical"), [(0.95, 3.43035370), (0.99, 8.85766022)]
)
def test_hand_calculated_interval(tf_package, data, level, critical):
    y, w, x, z = data
    res = IV2SLS(y, w, x, z).fit(cov_type="unadjusted")
    # Orthogonal construction: first-stage F=10, beta=1, and IV SE=0.4.
    assert_allclose(res.params.iloc[-1], 1, atol=1e-13)
    assert_allclose(res.std_errors.iloc[-1], 0.4, rtol=1e-13)
    assert_allclose(
        res.tf_conf_int(level).iloc[0], [1 - 0.4 * critical, 1 + 0.4 * critical]
    )
    assert res.conf_int().iloc[-1, 0] > 0
    assert res.tf_conf_int().iloc[0, 0] < 0


@pytest.mark.parametrize(
    "level",
    [
        0.75,
        0.8,
        0.85,
        0.875,
        0.9,
        0.925,
        0.94,
        0.95,
        0.96,
        0.97,
        0.975,
        0.98,
        0.99,
        0.995,
        0.998,
        0.999,
        0.9995,
        0.9999,
    ],
)
def test_supported_levels(tf_package, data, level):
    y, w, x, z = data
    result = IV2SLS(y, w, x, z).fit(cov_type="unadjusted")
    critical = _tf_critical_value(10, level)
    assert_allclose(
        result.tf_conf_int(level).iloc[0],
        [1 - 0.4 * critical, 1 + 0.4 * critical],
        rtol=1e-12,
    )
    assert np.isfinite(_tf_critical_value(20, level))


def test_fit_below_old_cutoff_is_finite(tf_package, data):
    y, w, x, z = data
    x = x + (np.sqrt(3.9 / len(y)) - np.sqrt(10 / len(y))) * z
    result = IV2SLS(y, w, x, z).fit(cov_type="unadjusted")
    first_t = result.first_stage.individual["endog"].tstats.iloc[-1]
    assert_allclose(first_t**2, 3.9)
    interval = result.tf_conf_int().iloc[0]
    assert np.isfinite(interval).all()
    width = (interval.iloc[1] - interval.iloc[0]) / 2
    assert_allclose(width / result.std_errors.iloc[-1], 30.92820746, rtol=1e-9)


@pytest.mark.parametrize("level", [0, 1, -1, 0.74999, 0.99999, np.nan, np.inf])
def test_invalid_level(data, level):
    y, w, x, z = data
    with pytest.raises(ValueError, match="level must be finite"):
        IV2SLS(y, w, x, z).fit().tf_conf_int(level)


def test_default_never_computes(tf_package, data, monkeypatch):
    def unexpected_calibration(*args, **kwargs):
        pytest.fail("Default lookup requested numerical calibration")

    monkeypatch.setattr(tf_package, "calibrate", unexpected_calibration)
    y, w, x, z = data
    result = IV2SLS(y, w, x, z).fit()
    assert np.isfinite(result.tf_conf_int(0.9)).all().all()
    for level in [0.963, np.nextafter(0.95, 1)]:
        with pytest.raises(ValueError, match="method='compute'"):
            result.tf_conf_int(level)


@pytest.mark.parametrize("method", ["auto", "unknown"])
def test_invalid_method(data, method):
    y, w, x, z = data
    with pytest.raises(ValueError, match="method must be"):
        IV2SLS(y, w, x, z).fit().tf_conf_int(method=method)


def test_warm_table_no_io(tf_package, monkeypatch):
    expected = _tf_critical_value(10)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Warm lookup accessed package data")

    monkeypatch.setattr(tf_package, "available_alphas", unexpected_work)
    monkeypatch.setattr(tf_package.tables, "files", unexpected_work)
    assert _tf_critical_value(10) == expected


@pytest.mark.parametrize("method", ["table", "compute"])
def test_missing_optional_package(data, monkeypatch, method):
    def missing_package(name):
        raise ModuleNotFoundError("No module named 'tf_inference'", name=name)

    monkeypatch.setattr(_tf, "import_module", missing_package)
    y, w, x, z = data
    result = IV2SLS(y, w, x, z).fit()
    assert np.isfinite(result.conf_int()).all().all()
    with pytest.raises(ImportError, match="optional tf-inference package"):
        result.tf_conf_int(method=method)


def test_package_import_failure_preserved(monkeypatch):
    error = ModuleNotFoundError(
        "No module named 'internal_dependency'", name="internal_dependency"
    )

    def broken_package(name):
        raise error

    monkeypatch.setattr(_tf, "import_module", broken_package)
    with pytest.raises(ModuleNotFoundError) as exc:
        _tf_critical_value(10)
    assert exc.value is error


@pytest.mark.parametrize("method", ["table", "compute"])
@pytest.mark.parametrize("alpha", [0.0001, 0.05, 0.25])
def test_exact_alpha_forwarding(monkeypatch, method, alpha):
    calls = []
    package = ModuleType("tf_inference")
    package.available_alphas = lambda: (alpha,)

    def critical_value(f_stat, *, alpha, method):
        calls.append((f_stat, alpha, method))
        return 4.0

    package.critical_value = critical_value
    monkeypatch.setattr(_tf, "import_module", lambda name: package)
    assert _tf_critical_value(10, 1 - alpha, method=method) == 4.0
    assert calls == [(10, alpha, method)]


def test_optional_computation_with_real_package(tf_package, data, monkeypatch):
    y, w, x, z = data
    result = IV2SLS(y, w, x, z).fit(cov_type="unadjusted")
    interval = result.tf_conf_int(0.963, method="compute")
    expected = tf_package.confidence_interval(1, 0.4, 10, 1 - 0.963, method="compute")
    assert_allclose(interval.iloc[0], expected, rtol=1e-12)

    def unexpected_calibration(*args, **kwargs):
        pytest.fail("Repeated inference recalibrated the same level")

    monkeypatch.setattr(tf_package, "calibrate", unexpected_calibration)
    assert_frame_equal(result.tf_conf_int(0.963, method="compute"), interval)


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
def test_independent_covariance_calculation(tf_package, data, cov_type, controls):
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


def test_formula_labels_and_retained_sample(tf_package, data):
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


def test_duplicate_control_and_endogenous_labels(tf_package, data):
    y, w, x, z = data
    expected = IV2SLS(y, w, DataFrame({"x": x}), z).fit().tf_conf_int()
    controls = DataFrame(w, columns=["x", "w"])
    res = IV2SLS(y, controls, DataFrame({"x": x}), z).fit()
    assert_frame_equal(res.tf_conf_int(), expected)


@pytest.mark.parametrize("label", [7, ("endogenous", "x")])
def test_nonstring_endogenous_label(tf_package, data, label):
    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int()
    actual = IV2SLS(y, w, Series(x, name=label), z).fit().tf_conf_int()
    assert_allclose(actual, expected)
    assert list(actual.index) == [label]


@pytest.mark.parametrize("instrument_scale", [-1000, -1, 1e-6, 1e6])
def test_instrument_scale_and_sign(tf_package, data, instrument_scale):
    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int()
    actual = IV2SLS(y, w, x, z * instrument_scale).fit().tf_conf_int()
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("outcome_scale", "endog_scale"), [(2, 1), (-2, 1), (1, -3), (1e-4, 1e4)]
)
def test_coefficient_scale(tf_package, data, outcome_scale, endog_scale):
    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int().to_numpy()
    expected = np.sort(expected * outcome_scale / endog_scale, axis=1)
    actual = IV2SLS(y * outcome_scale, w, x * endog_scale, z).fit().tf_conf_int()
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_row_permutation_and_no_mutation(tf_package, data):
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


def test_weak_first_stage(tf_package, data):
    y, w, x, z = data
    weak_x = x - (np.sqrt(10 / len(y)) - 0.01) * z
    res = IV2SLS(y, w, weak_x, z).fit(cov_type="unadjusted")
    assert_array_equal(res.tf_conf_int(), [[-np.inf, np.inf]])


@pytest.mark.parametrize(
    ("model", "config"),
    [(IVLIML, {}), (IVLIML, {"kappa": 1}), (IVLIML, {"fuller": 1}), (IVJIVE, {})],
)
def test_unsupported_estimator(data, model, config):
    y, w, x, z = data
    with pytest.raises(ValueError, match="requires IV2SLS"):
        model(y, w, x, z, **config).fit().tf_conf_int()


def test_iv2sls_subclass(tf_package, data):
    class CustomIV2SLS(IV2SLS):
        pass

    y, w, x, z = data
    expected = IV2SLS(y, w, x, z).fit().tf_conf_int()
    actual = CustomIV2SLS(y, w, x, z).fit().tf_conf_int()
    assert_frame_equal(actual, expected)


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


def test_weights(tf_package, data):
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


def test_missing_observations(tf_package, data):
    y, w, x, z = data
    y, x = y.copy(), x.copy()
    y[0], x[1] = np.nan, np.nan
    with pytest.warns(MissingValueWarning):
        res = IV2SLS(y, w, x, z).fit()
    complete = IV2SLS(y[2:], w[2:], x[2:], z[2:]).fit()
    assert_frame_equal(res.tf_conf_int(), complete.tf_conf_int())
