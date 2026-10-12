"""
Input types of the estimators (decision D5)

A parameter that is documented as array_like is converted to an array when it is
received, so that a list or a tuple works like a NumPy array, a pandas Series or
DataFrame, or an xarray DataArray. The data of the panel models need labels or a
layout, and so they list the containers that they accept, and raise a TypeError
that names the parameter for anything else. Complex values are rejected rather
than silently discarding their imaginary part.

The tests compare the numbers that are used in estimation, and the estimates,
of an input in each of the supported containers with a pandas float64 input.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from linearmodels import OLS
from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv import (
    IV2SLS,
    IVGMM,
    IVGMMCUE,
    IVJIVE,
    IVLIML,
    AbsorbingLS,
    Interaction,
)
from linearmodels.iv.data import IVData
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.data import PanelData
from linearmodels.panel.utility import dummy_matrix
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover
    HAS_XARRAY = False

NOBS = 80
IV_ESTIMATORS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]
# IVGMMCUE minimizes a function, which is slow, and so it is only fit once
FAST_ESTIMATORS = [IV2SLS, IVLIML, IVGMM, IVJIVE]
IV_IDS = [c.__name__ for c in IV_ESTIMATORS]
FAST_IDS = [c.__name__ for c in FAST_ESTIMATORS]
XARRAY_KINDS = ["dataarray", "dataarray2d"] if HAS_XARRAY else []
VECTOR_KINDS = [
    "list",
    "tuple",
    "ndarray",
    "ndarray2d",
    "series",
    "series_unnamed",
    "dataframe",
    "ivdata",
    *XARRAY_KINDS,
]
MATRIX_KINDS = ["list", "tuple", "ndarray", "dataframe", "ivdata", *XARRAY_KINDS]
IV_PARAMS = ["dependent", "exog", "endog", "instruments", "weights"]
VECTOR_PARAMS = {"dependent", "endog", "weights"}


def iv_data(seed=0, nobs=NOBS):
    """Canonical IV data: pandas objects with the default index"""
    rng = np.random.default_rng(seed)
    instruments = pd.DataFrame(
        rng.standard_normal((nobs, 3)), columns=["z1", "z2", "z3"]
    )
    u = rng.standard_normal(nobs)
    endog = pd.Series(
        instruments.to_numpy() @ [1.0, 0.5, 0.25] + 0.5 * u + rng.standard_normal(nobs),
        name="x",
    )
    exog = pd.DataFrame({"const": np.ones(nobs), "w": rng.standard_normal(nobs)})
    dependent = pd.Series(
        1 + 0.5 * exog.w + endog + u + rng.standard_normal(nobs), name="y"
    )
    weights = pd.Series(np.exp(0.3 * rng.standard_normal(nobs)), name="wt")
    clusters = pd.Series(rng.integers(0, 10, nobs), name="cl")
    clusters2 = pd.DataFrame(
        {"c1": rng.integers(0, 10, nobs), "c2": rng.integers(0, 7, nobs)}
    )
    return {
        "dependent": dependent,
        "exog": exog,
        "endog": endog,
        "instruments": instruments,
        "weights": weights,
        "clusters": clusters,
        "clusters2": clusters2,
    }


def make(value, kind):
    """The pandas object value as another kind of input"""
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    series = value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value
    nested = frame.to_numpy().tolist()
    vector = isinstance(value, pd.Series) or frame.shape[1] == 1
    columns = [str(c) for c in frame.columns]
    makers = {
        "list": lambda: series.tolist() if vector else nested,
        "list2d": lambda: nested,
        "tuple": lambda: (
            tuple(series.tolist()) if vector else tuple(map(tuple, nested))
        ),
        "ndarray": lambda: series.to_numpy() if vector else frame.to_numpy(),
        "ndarray2d": frame.to_numpy,
        "series": lambda: series.rename("v"),
        "series_unnamed": lambda: series.rename(None),
        "dataframe": lambda: frame,
        "dataarray": lambda: (
            xr.DataArray(series.to_numpy(), dims=["obs"])
            if vector
            else xr.DataArray(
                frame.to_numpy(), dims=["obs", "var"], coords={"var": columns}
            )
        ),
        "dataarray2d": lambda: xr.DataArray(
            frame.to_numpy(), dims=["obs", "var"], coords={"var": columns}
        ),
        "ivdata": lambda: IVData(value),
    }
    return makers[kind]()


def build_iv(cls, data, **replace):
    values = {**data, **replace}
    if cls is OLS:
        return cls(values["dependent"], values["exog"], weights=values["weights"])
    return cls(
        values["dependent"],
        values["exog"],
        values["endog"],
        values["instruments"],
        weights=values["weights"],
    )


def estimation_arrays(mod):
    """The numbers that are used in estimation"""
    return [
        mod.dependent.ndarray,
        mod.exog.ndarray,
        mod.endog.ndarray,
        mod.instruments.ndarray,
        mod.weights.ndarray,
    ]


def estimates(res):
    return np.r_[np.asarray(res.params).ravel(), np.asarray(res.cov).ravel()]


@pytest.fixture(scope="module")
def data():
    return iv_data()


@pytest.fixture(scope="module")
def reference(data):
    """Models that use the canonical pandas inputs"""
    return {cls: build_iv(cls, data) for cls in [*IV_ESTIMATORS, OLS]}


# --------------------------------------------------------------------------- #
# IV models
# --------------------------------------------------------------------------- #
ALL_IV = [*IV_ESTIMATORS, OLS]
ALL_IV_IDS = [*IV_IDS, "OLS"]
VECTOR_CASES = [
    (cls, name)
    for cls in ALL_IV
    for name in ("dependent", "endog", "weights")
    if not (cls is OLS and name == "endog")
]
MATRIX_CASES = [
    (cls, name)
    for cls in ALL_IV
    for name in ("exog", "instruments")
    if not (cls is OLS and name == "instruments")
]


def case_ids(cases):
    return [f"{'OLS' if c is OLS else c.__name__}-{n}" for c, n in cases]


def assert_same_arrays(mod, other):
    for new, old in zip(estimation_arrays(mod), estimation_arrays(other), strict=True):
        assert_array_equal(new, old)


@pytest.mark.parametrize(("cls", "name"), VECTOR_CASES, ids=case_ids(VECTOR_CASES))
@pytest.mark.parametrize("kind", VECTOR_KINDS)
def test_iv_vector_containers(data, reference, cls, name, kind):
    mod = build_iv(cls, data, **{name: make(data[name], kind)})
    assert_same_arrays(mod, reference[cls])


@pytest.mark.parametrize(("cls", "name"), MATRIX_CASES, ids=case_ids(MATRIX_CASES))
@pytest.mark.parametrize("kind", MATRIX_KINDS)
def test_iv_matrix_containers(data, reference, cls, name, kind):
    mod = build_iv(cls, data, **{name: make(data[name], kind)})
    assert_same_arrays(mod, reference[cls])


@pytest.mark.parametrize("cls", FAST_ESTIMATORS, ids=FAST_IDS)
@pytest.mark.parametrize("name", IV_PARAMS)
@pytest.mark.parametrize("kind", ["list", "tuple"])
def test_iv_lists_give_the_same_estimates(data, reference, cls, name, kind):
    mod = build_iv(cls, data, **{name: make(data[name], kind)})
    assert_allclose(estimates(mod.fit()), estimates(reference[cls].fit()), rtol=1e-12)


def test_iv_cue_lists_give_the_same_estimates(data, reference):
    lists = {name: make(data[name], "list") for name in IV_PARAMS}
    mod = build_iv(IVGMMCUE, data, **lists)
    assert_allclose(
        estimates(mod.fit()), estimates(reference[IVGMMCUE].fit()), rtol=1e-6
    )


@pytest.mark.skipif(not HAS_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("cls", FAST_ESTIMATORS, ids=FAST_IDS)
def test_iv_dataarrays_give_the_same_estimates(data, reference, cls):
    arrays = {
        name: make(data[name], "dataarray") for name in IV_PARAMS if name != "weights"
    }
    mod = build_iv(cls, data, **arrays)
    assert_allclose(estimates(mod.fit()), estimates(reference[cls].fit()), rtol=1e-12)


def test_list_is_not_labelled_but_original_is_kept(data):
    # The raw input is what decides whether IVData carries labels
    raw = make(data["exog"], "list")
    ivd = IVData(raw)
    assert ivd.original is raw
    assert ivd.ndarray.dtype == np.float64
    assert ivd.shape == data["exog"].shape


def binary_data(seed=1, nobs=NOBS):
    """Data where all values are exact in any dtype: 0 and 1, or small integers"""
    rng = np.random.default_rng(seed)
    out = {
        "instruments": pd.DataFrame(
            rng.integers(0, 2, (nobs, 3)).astype(float), columns=["z1", "z2", "z3"]
        ),
        "endog": pd.Series(rng.integers(0, 2, nobs).astype(float), name="x"),
        "exog": pd.DataFrame(
            {"const": np.ones(nobs), "w": rng.integers(0, 2, nobs).astype(float)}
        ),
        "dependent": pd.Series(rng.integers(0, 2, nobs).astype(float), name="y"),
        "weights": pd.Series(rng.integers(1, 3, nobs).astype(float), name="wt"),
    }
    return out


DTYPES = ["int64", "float32", "bool", "object"]
# pandas object columns are not numeric, and so they are not accepted
DTYPE_CASES = [
    (container, dtype)
    for container in ("ndarray", "dataframe", "list")
    for dtype in DTYPES
    if (container, dtype) != ("dataframe", "object")
]


@pytest.mark.parametrize("name", ["dependent", "exog", "endog", "instruments"])
@pytest.mark.parametrize(("container", "dtype"), DTYPE_CASES)
def test_iv_dtypes(name, container, dtype):
    values = binary_data()
    canonical = build_iv(IV2SLS, values)
    value = values[name]
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    if container == "ndarray":
        new = frame.to_numpy().astype(dtype)
    elif container == "dataframe":
        new = frame.astype(dtype)
    else:
        new = frame.to_numpy().astype(dtype).tolist()
    mod = build_iv(IV2SLS, values, **{name: new})
    assert_same_arrays(mod, canonical)


@pytest.mark.parametrize("dtype", ["int64", "float32", "object"])
def test_iv_weight_dtypes(dtype):
    values = binary_data()
    new = values["weights"].to_numpy().astype(dtype)
    assert_same_arrays(build_iv(IV2SLS, values, weights=new), build_iv(IV2SLS, values))


def test_iv_categorical_and_string_columns_become_dummies():
    values = binary_data()
    levels = pd.Series(np.arange(NOBS) % 3, name="g")
    dummies = pd.get_dummies(levels.astype("category"), drop_first=True).astype(float)
    exog = values["exog"]
    expected = pd.concat([exog, dummies.set_axis(["g.1", "g.2"], axis=1)], axis=1)
    for column in (levels.astype("category"), levels.astype(str)):
        mixed = pd.concat([exog, column], axis=1)
        mod = build_iv(IV2SLS, values, exog=mixed)
        assert_array_equal(mod.exog.ndarray, expected.to_numpy())


def complex_inputs(value):
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    out = {
        "ndarray": frame.to_numpy().astype(complex),
        "list": frame.to_numpy().astype(complex).tolist(),
        "dataframe": frame.astype(complex),
    }
    if HAS_XARRAY:
        out["dataarray"] = xr.DataArray(
            frame.to_numpy().astype(complex), dims=["obs", "var"]
        )
    return out


COMPLEX_KINDS = ["ndarray", "list", "dataframe", *(["dataarray"] if HAS_XARRAY else [])]


@pytest.mark.parametrize("name", IV_PARAMS)
@pytest.mark.parametrize("kind", COMPLEX_KINDS)
def test_complex_input_raises_and_names_the_input(data, name, kind):
    value = complex_inputs(data[name])[kind]
    with warnings.catch_warnings():
        # The imaginary part must not be discarded with a warning
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=rf"{name} has a complex dtype"):
            build_iv(IV2SLS, data, **{name: value})


@pytest.mark.parametrize("cls", [IVLIML, IVGMM, IVGMMCUE, IVJIVE, OLS])
def test_complex_input_raises_in_every_iv_model(data, cls):
    value = complex_inputs(data["exog"])["ndarray"]
    with pytest.raises(TypeError, match="exog has a complex dtype"):
        build_iv(cls, data, exog=value)


def test_complex_series_with_a_zero_imaginary_part_raises(data):
    # The imaginary part is zero, but the dtype is not supported either way
    with pytest.raises(TypeError, match="weights has a complex dtype"):
        build_iv(IV2SLS, data, weights=data["weights"].astype(complex))


def test_complex_predict_raises(data):
    res = build_iv(IV2SLS, data).fit()
    exog = complex_inputs(data["exog"])["ndarray"]
    with pytest.raises(TypeError, match="exog has a complex dtype"):
        res.predict(exog=exog, endog=data["endog"])


@pytest.mark.parametrize("name", IV_PARAMS)
def test_text_input_raises_and_names_the_input(data, name):
    frame = data[name].to_frame() if isinstance(data[name], pd.Series) else data[name]
    text = np.full(frame.shape, "abc")
    with pytest.raises(TypeError, match=rf"{name} must contain numbers"):
        build_iv(IV2SLS, data, **{name: text})
    with pytest.raises(TypeError, match=rf"{name} must contain numbers"):
        build_iv(IV2SLS, data, **{name: text.tolist()})


@pytest.mark.parametrize("name", IV_PARAMS)
def test_object_columns_raise_and_name_the_input(data, name):
    value = data[name].astype(object)
    with pytest.raises(ValueError, match=rf"{name} has a column .* object"):
        build_iv(IV2SLS, data, **{name: value})


@pytest.mark.parametrize("value", [5.0, "abc", {"a": 1}, {1, 2}])
def test_not_array_like_raises_and_names_the_input(data, value):
    with pytest.raises(TypeError, match="exog must be array_like"):
        build_iv(IV2SLS, data, exog=value)


def test_ragged_list_raises_and_names_the_input(data):
    ragged = [[1.0, 2.0], [3.0]] + [[1.0]] * (NOBS - 2)
    with pytest.raises(ValueError, match="weights could not be converted"):
        build_iv(IV2SLS, data, weights=ragged)


def test_ragged_list_as_object_array_raises(data, monkeypatch):
    # NumPy < 1.24 makes an object array of the rows of a ragged list, with a
    # warning, instead of raising
    asarray = np.asarray

    def old_asarray(value, *args, **kwargs):
        try:
            return asarray(value, *args, **kwargs)
        except ValueError:
            out = np.empty(len(value), dtype=object)
            out[:] = list(value)
            return out

    monkeypatch.setattr(np, "asarray", old_asarray)
    ragged = [[1.0, 2.0], [3.0]] + [[1.0]] * (NOBS - 2)
    with pytest.raises(ValueError, match="weights could not be converted"):
        build_iv(IV2SLS, data, weights=ragged)


def test_wrong_number_of_observations_in_a_list(data):
    with pytest.raises(ValueError, match="exog has the wrong number of observations"):
        IV2SLS(
            data["dependent"],
            data["exog"].iloc[:-1].to_numpy().tolist(),
            data["endog"],
            data["instruments"],
        )


def test_ivdata_input_has_its_number_of_observations_checked(data):
    # IVData is not converted again, but it must have the right shape
    exog = IVData(data["exog"].iloc[:-1])
    with pytest.raises(ValueError, match="exog has the wrong number of observations"):
        IVData(exog, var_name="exog", nobs=NOBS)
    with pytest.raises(ValueError, match="exog has the wrong number of observations"):
        IV2SLS(data["dependent"], exog, data["endog"], data["instruments"])


@pytest.mark.parametrize("cls", IV_ESTIMATORS, ids=IV_IDS)
def test_column_name_as_weights_raises(cls, data):
    frame = pd.concat(
        [data["dependent"], data["exog"][["w"]], data["endog"], data["instruments"]],
        axis=1,
    )
    frame["wt"] = data["weights"]
    formula = "y ~ 1 + w + [x ~ z1 + z2 + z3]"
    with pytest.raises(TypeError, match=r"weights must be array_like.*data\['wt'\]"):
        cls.from_formula(formula, frame, weights="wt")
    # The values of the column work
    mod = cls.from_formula(formula, frame, weights=frame["wt"])
    expected = data["weights"] / data["weights"].mean()
    assert_allclose(mod.weights.ndarray[:, 0], expected)


# --------------------------------------------------------------------------- #
# Clusters
# --------------------------------------------------------------------------- #
CLUSTER_KINDS = [
    "list",
    "tuple",
    "ndarray",
    "ndarray2d",
    "series",
    "series_unnamed",
    "dataframe",
    "ivdata",
    "strings",
    "categorical",
    *(["dataarray"] if HAS_XARRAY else []),
]


def make_clusters(value, kind):
    if kind == "strings":
        return value.astype(str)
    if kind == "categorical":
        return value.astype(str).astype("category")
    return make(value, kind)


def fit_clustered(cls, data, clusters, **kw):
    mod = build_iv(cls, data)
    return estimates(mod.fit(cov_type="clustered", clusters=clusters, **kw))


@pytest.mark.parametrize("cls", FAST_ESTIMATORS, ids=FAST_IDS)
@pytest.mark.parametrize("kind", CLUSTER_KINDS)
def test_iv_clusters_containers(data, cls, kind):
    expected = fit_clustered(cls, data, data["clusters"].to_numpy())
    clusters = make_clusters(data["clusters"], kind)
    assert_allclose(fit_clustered(cls, data, clusters), expected, rtol=1e-10)


@pytest.mark.parametrize("kind", ["list", "tuple", "series", "strings", "categorical"])
def test_iv_cue_clusters_containers(data, kind):
    expected = fit_clustered(IVGMMCUE, data, data["clusters"].to_numpy())
    clusters = make_clusters(data["clusters"], kind)
    assert_allclose(fit_clustered(IVGMMCUE, data, clusters), expected, rtol=1e-6)


@pytest.mark.parametrize("cls", [IVGMM, IVGMMCUE], ids=["IVGMM", "IVGMMCUE"])
@pytest.mark.parametrize("kind", ["list", "tuple", "series", "strings", "categorical"])
def test_gmm_weight_config_clusters_containers(data, cls, kind):
    def fit(clusters):
        mod = cls(
            data["dependent"],
            data["exog"],
            data["endog"],
            data["instruments"],
            weight_type="clustered",
            clusters=clusters,
        )
        return estimates(mod.fit())

    expected = fit(data["clusters"].to_numpy())
    assert_allclose(fit(make_clusters(data["clusters"], kind)), expected, rtol=1e-6)


@pytest.mark.parametrize("cls", [IV2SLS, IVLIML], ids=["IV2SLS", "IVLIML"])
@pytest.mark.parametrize(
    "kind", ["list2d", "tuple", "ndarray", "dataframe", *XARRAY_KINDS[:1]]
)
def test_iv_two_way_clusters_containers(data, cls, kind):
    expected = fit_clustered(cls, data, data["clusters2"].to_numpy())
    clusters = make(data["clusters2"], kind)
    assert_allclose(fit_clustered(cls, data, clusters), expected, rtol=1e-10)


@pytest.mark.parametrize("cls", [IV2SLS, IVLIML], ids=["IV2SLS", "IVLIML"])
def test_three_cluster_columns_raise(data, cls):
    clusters = np.c_[data["clusters2"].to_numpy(), data["clusters"].to_numpy()]
    mod = build_iv(cls, data)
    match = r"clusters must be a 1-d array or a 2-d array with 1 or 2 columns"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=clusters)
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=clusters.tolist())
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=np.zeros((NOBS, 0)))


@pytest.mark.parametrize(
    "cls", [IVGMM, IVGMMCUE, IVJIVE], ids=["IVGMM", "IVGMMCUE", "IVJIVE"]
)
@pytest.mark.parametrize("columns", [2, 3])
def test_multiple_cluster_columns_raise_in_one_way_estimators(data, cls, columns):
    clusters = np.c_[data["clusters2"].to_numpy(), data["clusters"].to_numpy()]
    mod = build_iv(cls, data)
    with pytest.raises(ValueError, match="clusters") as exc:
        mod.fit(cov_type="clustered", clusters=clusters[:, :columns])
    message = str(exc.value)
    assert "single column" in message or "one-way" in message


@pytest.mark.parametrize("cls", [IVGMM, IVGMMCUE], ids=["IVGMM", "IVGMMCUE"])
def test_weight_config_cluster_columns_raise(data, cls):
    with pytest.raises(ValueError, match="clusters must be a 1-d array or a 2-d array"):
        cls(
            data["dependent"],
            data["exog"],
            data["endog"],
            data["instruments"],
            weight_type="clustered",
            clusters=data["clusters2"],
        )


@pytest.mark.parametrize("cls", IV_ESTIMATORS, ids=IV_IDS)
def test_complex_clusters_raise(data, cls):
    mod = build_iv(cls, data)
    clusters = data["clusters"].to_numpy().astype(complex)
    with pytest.raises(TypeError, match="clusters has a complex dtype"):
        mod.fit(cov_type="clustered", clusters=clusters)


# --------------------------------------------------------------------------- #
# Weights and missing values
# --------------------------------------------------------------------------- #
MISSING_ROWS = {"dependent": [3, 40], "endog": [17], "exog": [25], "instruments": [9]}


def with_missing(data):
    """Data where some of the observations have a missing value"""
    out = {k: v.copy() for k, v in data.items()}
    dropped = []
    for name, rows in MISSING_ROWS.items():
        for row in rows:
            if isinstance(out[name], pd.DataFrame):
                out[name].iloc[row, 0 if name == "instruments" else 1] = np.nan
            else:
                out[name].iloc[row] = np.nan
            dropped.append(row)
    keep = np.ones(NOBS, dtype=bool)
    keep[dropped] = False
    return out, keep


def results_arrays(res):
    return {
        "params": np.asarray(res.params),
        "cov": np.asarray(res.cov),
        "resids": np.asarray(res.resids),
        "wresids": np.asarray(res.wresids),
        "resid_ss": res.resid_ss,
        "s2": res.s2,
        "total_ss": res.total_ss,
        "rsquared": res.rsquared,
        "nobs": res.nobs,
    }


@pytest.mark.parametrize("cls", FAST_ESTIMATORS, ids=FAST_IDS)
def test_weights_are_normalized_after_dropping_missing_values(data, cls):
    # A weight is relative to the weights of the observations that are used, and
    # so the model is the same model as the one fit to the complete observations
    values, keep = with_missing(data)
    with pytest.warns(MissingValueWarning):
        res = build_iv(cls, values).fit()
    manual = {k: v.loc[keep].reset_index(drop=True) for k, v in data.items()}
    expected = build_iv(cls, manual).fit()
    actual = results_arrays(res)
    for key, value in results_arrays(expected).items():
        assert_allclose(actual[key], value, rtol=1e-10, err_msg=key)
    # The weights have mean 1
    assert_allclose(res.model.weights.ndarray.mean(), 1.0)


def test_cue_weights_are_normalized_after_dropping_missing_values(data):
    values, keep = with_missing(data)
    with pytest.warns(MissingValueWarning):
        res = build_iv(IVGMMCUE, values).fit()
    manual = {k: v.loc[keep].reset_index(drop=True) for k, v in data.items()}
    expected = build_iv(IVGMMCUE, manual).fit()
    assert_allclose(res.resid_ss, expected.resid_ss, rtol=1e-6)
    assert_allclose(res.s2, expected.s2, rtol=1e-6)
    assert_allclose(res.total_ss, expected.total_ss, rtol=1e-6)


def test_weights_with_missing_values_in_the_weights(data):
    values = {k: v.copy() for k, v in data.items()}
    values["weights"].iloc[[5, 6]] = np.nan
    keep = np.ones(NOBS, dtype=bool)
    keep[[5, 6]] = False
    with pytest.warns(MissingValueWarning):
        res = build_iv(IV2SLS, values).fit()
    manual = {k: v.loc[keep].reset_index(drop=True) for k, v in data.items()}
    expected = build_iv(IV2SLS, manual).fit()
    assert_allclose(res.resid_ss, expected.resid_ss, rtol=1e-12)
    assert_allclose(res.params, expected.params, rtol=1e-12)


@pytest.mark.parametrize("cls", ALL_IV, ids=ALL_IV_IDS)
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_iv_weights_must_be_strictly_positive(data, cls, bad):
    weights = data["weights"].copy()
    weights.iloc[4] = bad
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        build_iv(cls, data, weights=weights.tolist())


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def test_predict_accepts_lists(data):
    mod = build_iv(IV2SLS, data)
    res = mod.fit()
    exog, endog = data["exog"].iloc[:10], data["endog"].iloc[:10]
    expected = res.predict(exog=exog, endog=endog)
    for kind in ("list", "tuple"):
        listed = res.predict(exog=make(exog, kind), endog=make(endog, kind))
        assert_allclose(listed.to_numpy(), expected.to_numpy())
        direct = mod.predict(res.params, exog=make(exog, kind), endog=make(endog, kind))
        assert_allclose(direct.to_numpy(), expected.to_numpy())
    params = list(np.asarray(res.params))
    listed = mod.predict(params, exog=exog, endog=endog)
    assert_allclose(listed.to_numpy(), expected.to_numpy())


def test_predict_accepts_lists_for_ols(data):
    res = build_iv(OLS, data).fit()
    exog = data["exog"].iloc[:10]
    expected = res.predict(exog=exog, endog=None)
    listed = res.predict(exog=exog.to_numpy().tolist())
    assert_allclose(listed.to_numpy(), expected.to_numpy())


# --------------------------------------------------------------------------- #
# AbsorbingLS
# --------------------------------------------------------------------------- #
def absorbing_data(seed=2, nobs=NOBS):
    rng = np.random.default_rng(seed)
    exog = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["w1", "w2"])
    cats = pd.DataFrame(
        {
            "a": pd.Categorical(rng.integers(0, 6, nobs)),
            "b": pd.Categorical(rng.integers(0, 4, nobs)),
        }
    )
    dependent = pd.Series(
        exog.w1 - 0.5 * exog.w2 + cats.a.cat.codes + rng.standard_normal(nobs),
        name="y",
    )
    weights = pd.Series(np.exp(0.3 * rng.standard_normal(nobs)), name="wt")
    clusters = pd.Series(rng.integers(0, 9, nobs), name="cl")
    clusters2 = pd.DataFrame(
        {"c1": rng.integers(0, 9, nobs), "c2": rng.integers(0, 7, nobs)}
    )
    return {
        "dependent": dependent,
        "exog": exog,
        "absorb": cats,
        "weights": weights,
        "clusters": clusters,
        "clusters2": clusters2,
    }


@pytest.fixture(scope="module")
def absorbing():
    return absorbing_data()


def build_absorbing(values, **replace):
    values = {**values, **replace}
    return AbsorbingLS(
        values["dependent"],
        values["exog"],
        absorb=values["absorb"],
        weights=values.get("weights"),
    )


@pytest.fixture(scope="module")
def absorbing_reference(absorbing):
    return build_absorbing(absorbing).fit()


ABSORBING_CASES = [
    (name, kind)
    for name, kinds in (
        ("dependent", VECTOR_KINDS),
        ("exog", MATRIX_KINDS),
        ("weights", VECTOR_KINDS),
    )
    for kind in kinds
]


@pytest.mark.parametrize(("name", "kind"), ABSORBING_CASES)
def test_absorbing_containers(absorbing, absorbing_reference, name, kind):
    res = build_absorbing(absorbing, **{name: make(absorbing[name], kind)}).fit()
    assert_allclose(res.params, absorbing_reference.params, rtol=1e-10)
    assert_allclose(res.cov, absorbing_reference.cov, rtol=1e-10)
    assert_allclose(res.std_errors, absorbing_reference.std_errors, rtol=1e-10)


@pytest.mark.parametrize("name", ["dependent", "exog", "weights"])
def test_absorbing_complex_input_raises(absorbing, name):
    value = complex_inputs(absorbing[name])["ndarray"]
    with pytest.raises(TypeError, match=rf"{name} has a complex dtype"):
        build_absorbing(absorbing, **{name: value})


@pytest.mark.parametrize("bad", [0.0, -1.0])
@pytest.mark.parametrize("kind", ["series", "list"])
def test_absorbing_weights_must_be_strictly_positive(absorbing, bad, kind):
    weights = absorbing["weights"].copy()
    weights.iloc[7] = bad
    with pytest.raises(ValueError, match="weights must be strictly positive"):
        build_absorbing(absorbing, weights=make(weights, kind))


def test_absorbing_nan_weights_are_dropped_as_missing(absorbing):
    weights = absorbing["weights"].copy()
    weights.iloc[7] = np.nan
    with pytest.warns(MissingValueWarning):
        res = build_absorbing(absorbing, weights=weights).fit()
    assert res.nobs == NOBS - 1


def test_absorbing_string_effects_raise_and_say_to_use_categoricals(absorbing):
    strings = absorbing["absorb"].astype(str)
    with pytest.raises(TypeError, match=r"column 'a' of absorb has dtype") as exc:
        build_absorbing(absorbing, absorb=strings)
    message = str(exc.value)
    assert "categorical" in message
    assert "astype('category')" in message
    # Categorical effects work, and so do numeric variables to absorb
    build_absorbing(absorbing, absorb=strings.astype("category"))
    numeric = absorbing["absorb"].apply(lambda s: s.cat.codes.astype(float))
    build_absorbing(absorbing, absorb=numeric)


def test_absorbing_string_effects_in_interactions_raise(absorbing):
    frame = pd.DataFrame(
        {"g": absorbing["absorb"].a.astype(str), "c": absorbing["exog"].w1}
    )
    with pytest.raises(TypeError, match=r"column 'g' of interactions"):
        AbsorbingLS(
            absorbing["dependent"],
            absorbing["exog"],
            absorb=absorbing["absorb"],
            interactions=frame,
        )
    with pytest.raises(TypeError, match=r"column 'g' of interactions"):
        AbsorbingLS(
            absorbing["dependent"],
            absorbing["exog"],
            absorb=absorbing["absorb"],
            interactions=[frame],
        )
    with pytest.raises(TypeError, match=r"column 'g' of frame"):
        Interaction.from_frame(frame)


def test_absorbing_container_errors_name_the_input(absorbing):
    codes = np.column_stack([absorbing["absorb"][c].cat.codes for c in "ab"])
    with pytest.raises(TypeError, match="absorb must"):
        build_absorbing(absorbing, absorb=codes)
    with pytest.raises(TypeError, match="frame must be a DataFrame"):
        Interaction.from_frame(codes)


def test_interaction_accepts_lists(absorbing):
    cats = absorbing["absorb"][["a"]]
    cont = absorbing["exog"][["w1"]]
    expected = Interaction(cat=cats, cont=cont)

    def fit(interaction):
        mod = AbsorbingLS(
            absorbing["dependent"],
            absorbing["exog"][["w2"]],
            absorb=absorbing["absorb"][["b"]],
            interactions=interaction,
        )
        return estimates(mod.fit())

    codes = cats["a"].cat.codes
    reference = fit(expected)
    for cat in (codes.tolist(), tuple(codes), codes.to_numpy()[:, None].tolist()):
        for cont_value in (cont.to_numpy().tolist(), cont["w1"].tolist()):
            assert_allclose(fit(Interaction(cat=cat, cont=cont_value)), reference)
    assert_allclose(fit(Interaction(cat=codes.tolist(), cont=cont)), reference)
    only_cont = Interaction(cont=cont["w1"].tolist(), nobs=NOBS)
    assert only_cont.cont.shape == (NOBS, 1)


def test_interaction_rejects_complex_values(absorbing):
    cats = absorbing["absorb"][["a"]]
    cont = absorbing["exog"][["w1"]]
    with pytest.raises(TypeError, match="cont has a complex dtype"):
        Interaction(cat=cats, cont=cont.to_numpy().astype(complex))
    with pytest.raises(TypeError, match="cont has a complex dtype"):
        Interaction(cat=cats, cont=cont.astype(complex))
    with pytest.raises(TypeError, match="cat has a complex dtype"):
        Interaction(cat=cats.to_numpy().astype(complex), cont=cont)


@pytest.mark.parametrize("kind", ["list2d", "tuple", "ndarray", "dataframe"])
def test_absorbing_two_way_clusters_containers(absorbing, kind):
    mod = build_absorbing(absorbing)
    expected = mod.fit(cov_type="clustered", clusters=absorbing["clusters2"].to_numpy())
    clusters = make(absorbing["clusters2"], kind)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)


@pytest.mark.parametrize("kind", ["list", "tuple", "series", "ndarray2d"])
def test_absorbing_clusters_containers(absorbing, kind):
    mod = build_absorbing(absorbing)
    expected = mod.fit(cov_type="clustered", clusters=absorbing["clusters"].to_numpy())
    clusters = make(absorbing["clusters"], kind)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)


def test_absorbing_three_cluster_columns_raise(absorbing):
    clusters = np.c_[
        absorbing["clusters2"].to_numpy(), absorbing["clusters"].to_numpy()
    ]
    mod = build_absorbing(absorbing)
    match = r"clusters must be a 1-d array or a 2-d array with 1 or 2 columns"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=clusters)


def test_dummy_matrix_accepts_lists_and_rejects_other_shapes(absorbing):
    cats = absorbing["absorb"]
    codes = np.column_stack([cats[c].cat.codes for c in cats])
    expected = dummy_matrix(cats, precondition=False)[0].toarray()
    for value in (codes, codes.tolist(), tuple(map(tuple, codes.tolist()))):
        assert_array_equal(
            dummy_matrix(value, precondition=False)[0].toarray(), expected
        )
    with pytest.raises(ValueError, match="cats must be a DataFrame or a 2-d array"):
        dummy_matrix(codes[:, 0])
    with pytest.raises(TypeError, match="cats must only contain categorical"):
        dummy_matrix(pd.DataFrame(codes))


# --------------------------------------------------------------------------- #
# Systems of equations
# --------------------------------------------------------------------------- #
def system_data(seed=3, nobs=NOBS):
    rng = np.random.default_rng(seed)
    out = {}
    common = rng.standard_normal(nobs)
    z = pd.DataFrame(rng.standard_normal((nobs, 3)), columns=["z1", "z2", "z3"])
    x = pd.DataFrame(
        {"x": z.z1 + 0.5 * z.z2 + rng.standard_normal(nobs) + common},
    )
    w1 = pd.DataFrame({"const": np.ones(nobs), "w1": rng.standard_normal(nobs)})
    w2 = pd.DataFrame({"const": np.ones(nobs), "w2": rng.standard_normal(nobs)})
    y1 = pd.Series(1 + w1.w1 + x.x + rng.standard_normal(nobs) + common, name="y1")
    y2 = pd.Series(2 - 0.5 * w2.w2 + rng.standard_normal(nobs) + common, name="y2")
    weights1 = pd.Series(np.exp(0.3 * rng.standard_normal(nobs)), name="wt1")
    weights2 = pd.Series(np.exp(0.3 * rng.standard_normal(nobs)), name="wt2")
    out["eq1"] = {
        "dependent": y1,
        "exog": w1,
        "endog": x,
        "instruments": z,
        "weights": weights1,
    }
    out["eq2"] = {"dependent": y2, "exog": w2, "weights": weights2}
    out["clusters"] = pd.Series(rng.integers(0, 9, nobs), name="cl")
    out["clusters2"] = pd.DataFrame(
        {"c1": rng.integers(0, 9, nobs), "c2": rng.integers(0, 7, nobs)}
    )
    return out


SYSTEM_MODELS = [SUR, IV3SLS, IVSystemGMM]
SYSTEM_IDS = [c.__name__ for c in SYSTEM_MODELS]
SYSTEM_INPUTS = {
    SUR: ["dependent", "exog", "weights"],
    IV3SLS: ["dependent", "exog", "endog", "instruments", "weights"],
    IVSystemGMM: ["dependent", "exog", "endog", "instruments", "weights"],
}
SYSTEM_CASES = [(cls, name) for cls in SYSTEM_MODELS for name in SYSTEM_INPUTS[cls]]


@pytest.fixture(scope="module")
def system():
    return system_data()


def equations(system, cls, form="dict", **replace):
    """Equations where an input of the first equation can be replaced"""
    names = SYSTEM_INPUTS[cls]
    eq1 = {name: system["eq1"][name] for name in names}
    eq1.update(replace)
    eq2 = {name: system["eq2"][name] for name in ("dependent", "exog", "weights")}
    if form == "dict":
        return {"eq1": eq1, "eq2": eq2}
    first = tuple(eq1.get(n) for n in ("dependent", "exog", "endog", "instruments"))
    second = (eq2["dependent"], eq2["exog"], None, None)
    return {
        "eq1": (*first, eq1["weights"]),
        "eq2": (*second, eq2["weights"]),
    }


def system_arrays(mod):
    return [*mod._y, *mod._x, *mod._z, *mod._w]


@pytest.fixture(scope="module")
def system_reference(system):
    return {cls: cls(equations(system, cls)) for cls in SYSTEM_MODELS}


@pytest.mark.parametrize(
    ("cls", "name"),
    SYSTEM_CASES,
    ids=[f"{c.__name__}-{n}" for c, n in SYSTEM_CASES],
)
@pytest.mark.parametrize(
    "kind", ["list", "tuple", "ndarray", "dataframe", "ivdata", *XARRAY_KINDS[:1]]
)
def test_system_equation_containers(system, system_reference, cls, name, kind):
    new = make(system["eq1"][name], kind)
    mod = cls(equations(system, cls, **{name: new}))
    for a, b in zip(
        system_arrays(mod), system_arrays(system_reference[cls]), strict=True
    ):
        assert_array_equal(a, b)


@pytest.mark.parametrize("cls", [IV3SLS, SUR], ids=["IV3SLS", "SUR"])
@pytest.mark.parametrize("form", ["dict", "tuple"])
def test_system_lists_give_the_same_estimates(system, cls, form):
    names = SYSTEM_INPUTS[cls]
    lists = {name: make(system["eq1"][name], "list") for name in names}
    reference = cls(equations(system, cls, form)).fit()
    res = cls(equations(system, cls, form, **lists)).fit()
    assert_allclose(res.params, reference.params, rtol=1e-12)
    assert_allclose(res.cov, reference.cov, rtol=1e-12)


def test_system_gmm_lists_give_the_same_estimates(system):
    names = SYSTEM_INPUTS[IVSystemGMM]
    lists = {name: make(system["eq1"][name], "list") for name in names}
    reference = IVSystemGMM(equations(system, IVSystemGMM)).fit()
    res = IVSystemGMM(equations(system, IVSystemGMM, **lists)).fit()
    assert_allclose(res.params, reference.params, rtol=1e-10)


@pytest.mark.parametrize("cls", [SUR, IV3SLS], ids=["SUR", "IV3SLS"])
@pytest.mark.parametrize("kind", ["list", "tuple", "ndarray", "dataframe"])
def test_system_sigma_containers(system, cls, kind):
    sigma = pd.DataFrame([[1.0, 0.25], [0.25, 1.5]])
    eqns = equations(system, cls)
    expected = cls(eqns, sigma=sigma.to_numpy()).fit()
    res = cls(eqns, sigma=make(sigma, "list2d" if kind == "list" else kind)).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)


SYSTEM_FORM_CASES = [(cls, name, "dict") for cls, name in SYSTEM_CASES] + [
    (IV3SLS, name, "tuple") for name in SYSTEM_INPUTS[IV3SLS]
]


@pytest.mark.parametrize(
    ("cls", "name", "form"),
    SYSTEM_FORM_CASES,
    ids=[f"{c.__name__}-{n}-{f}" for c, n, f in SYSTEM_FORM_CASES],
)
def test_system_complex_input_raises(system, cls, name, form):
    value = complex_inputs(system["eq1"][name])["ndarray"]
    stem = {"instruments": "instr"}.get(name, name)
    with pytest.raises(TypeError, match=rf"{stem}_0 has a complex dtype"):
        cls(equations(system, cls, form, **{name: value}))


@pytest.mark.parametrize("cls", SYSTEM_MODELS, ids=SYSTEM_IDS)
def test_system_text_input_raises(system, cls):
    text = np.full(system["eq1"]["exog"].shape, "abc")
    with pytest.raises(TypeError, match="exog_0 must contain numbers"):
        cls(equations(system, cls, exog=text))


@pytest.mark.parametrize("kind", ["list", "tuple", "ndarray", "series"])
def test_multivariate_interfaces_accept_lists(system, kind):
    dependent = pd.concat(
        [system["eq1"]["dependent"], system["eq2"]["dependent"]], axis=1
    )
    exog = system["eq1"]["exog"]
    expected = SUR.multivariate_ls(dependent, exog).fit()
    new_dependent = make(
        dependent,
        "list" if kind == "list" else kind if kind != "series" else "dataframe",
    )
    new_exog = make(
        exog, "list" if kind == "list" else kind if kind != "series" else "dataframe"
    )
    res = SUR.multivariate_ls(new_dependent, new_exog).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)
    endog = system["eq1"]["endog"]
    instruments = system["eq1"]["instruments"]
    expected = IV3SLS.multivariate_iv(dependent, exog, endog, instruments).fit()
    res = IV3SLS.multivariate_iv(
        new_dependent,
        new_exog,
        make(
            endog,
            "list" if kind == "list" else kind if kind != "series" else "dataframe",
        ),
        make(
            instruments,
            "list" if kind == "list" else kind if kind != "series" else "dataframe",
        ),
    ).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)


def test_multivariate_interfaces_reject_complex_values(system):
    dependent = pd.concat(
        [system["eq1"]["dependent"], system["eq2"]["dependent"]], axis=1
    )
    with pytest.raises(TypeError, match="dependent has a complex dtype"):
        SUR.multivariate_ls(dependent.to_numpy().astype(complex), system["eq1"]["exog"])
    with pytest.raises(TypeError, match="exog has a complex dtype"):
        SUR.multivariate_ls(dependent, system["eq1"]["exog"].to_numpy().astype(complex))


SYSTEM_DATA_COLUMNS = {
    "eq1": "y1 ~ 1 + w1 + [x ~ z1 + z2 + z3]",
    "eq2": "y2 ~ 1 + w2",
}


def system_frame(system):
    return pd.concat(
        [
            system["eq1"]["dependent"],
            system["eq1"]["exog"][["w1"]],
            system["eq1"]["endog"],
            system["eq1"]["instruments"],
            system["eq2"]["dependent"],
            system["eq2"]["exog"][["w2"]],
        ],
        axis=1,
    )


@pytest.mark.parametrize("kind", ["list", "tuple", "ndarray", "series", "dataframe"])
def test_system_formula_weights_containers(system, kind):
    frame = system_frame(system)
    weights = {"eq1": system["eq1"]["weights"], "eq2": system["eq2"]["weights"]}
    expected = IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=weights).fit()
    new = {name: make(value, kind) for name, value in weights.items()}
    res = IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=new).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)
    # The weights can be a DataFrame
    frame_weights = pd.DataFrame(weights)
    res = IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=frame_weights).fit()
    assert_allclose(res.params, expected.params, rtol=1e-12)


def test_system_formula_weights_errors_name_the_input(system):
    frame = system_frame(system)
    weights = [system["eq1"]["weights"].tolist(), system["eq2"]["weights"].tolist()]
    with pytest.raises(TypeError, match="weights must be dictionary-like"):
        IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=weights)
    with pytest.raises(TypeError, match="weights must be dictionary-like"):
        IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=np.ones((NOBS, 2)))
    with pytest.raises(TypeError, match=r"weights must be array_like.*data\['wt'\]"):
        IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights={"eq1": "wt"})
    complex_weights = {"eq1": system["eq1"]["weights"].to_numpy().astype(complex)}
    with pytest.raises(TypeError, match="weights has a complex dtype"):
        IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=complex_weights)
    text = {"eq1": np.full(NOBS, "abc")}
    with pytest.raises(TypeError, match="weights must contain numbers"):
        IV3SLS.from_formula(SYSTEM_DATA_COLUMNS, frame, weights=text)


def test_system_predict_accepts_lists(system):
    mod = IV3SLS(equations(system, IV3SLS))
    res = mod.fit()
    exog, endog = system["eq1"]["exog"].iloc[:10], system["eq1"]["endog"].iloc[:10]
    expected = mod.predict(
        res.params, equations={"eq1": {"exog": exog, "endog": endog}}
    )
    listed = {
        "eq1": {"exog": exog.to_numpy().tolist(), "endog": endog.to_numpy().tolist()}
    }
    new = mod.predict(res.params, equations=listed)
    assert_allclose(new.to_numpy(), expected.to_numpy())
    new = res.predict(equations=listed, dataframe=True)
    assert_allclose(new.to_numpy(), expected.to_numpy())


@pytest.mark.parametrize("cls", [SUR, IV3SLS], ids=["SUR", "IV3SLS"])
@pytest.mark.parametrize("kind", ["list", "tuple", "series", "strings"])
def test_system_clusters_containers(system, cls, kind):
    mod = cls(equations(system, cls))
    expected = mod.fit(cov_type="clustered", clusters=system["clusters"].to_numpy())
    clusters = make_clusters(system["clusters"], kind)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)


@pytest.mark.parametrize("cls", [SUR, IV3SLS], ids=["SUR", "IV3SLS"])
@pytest.mark.parametrize("shape", [(NOBS, 3), (NOBS, 2, 2), (NOBS - 1,)])
def test_system_clusters_with_the_wrong_shape_raise_a_plain_message(system, cls, shape):
    mod = cls(equations(system, cls))
    clusters = np.arange(np.prod(shape)).reshape(shape) % 5
    with pytest.raises(ValueError, match="clusters must be an ndarray") as exc:
        mod.fit(cov_type="clustered", clusters=clusters)
    # The message is the only argument, and so it is not printed as a tuple
    assert len(exc.value.args) == 1
    assert str(exc.value).startswith("clusters must be an ndarray")


# --------------------------------------------------------------------------- #
# Asset pricing
# --------------------------------------------------------------------------- #
def factor_data(seed=4, nobs=NOBS):
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["f1", "f2"])
    loadings = rng.uniform(0.5, 1.5, (2, 4))
    portfolios = pd.DataFrame(
        factors.to_numpy() @ loadings + rng.standard_normal((nobs, 4)),
        columns=["p0", "p1", "p2", "p3"],
    )
    return portfolios, factors


FACTOR_MODELS = [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
FACTOR_IDS = [c.__name__ for c in FACTOR_MODELS]


@pytest.fixture(scope="module")
def factors_data():
    return factor_data()


@pytest.mark.parametrize("cls", FACTOR_MODELS, ids=FACTOR_IDS)
@pytest.mark.parametrize("name", ["portfolios", "factors"])
@pytest.mark.parametrize(
    "kind", ["list", "tuple", "ndarray", "dataframe", "ivdata", *XARRAY_KINDS[:1]]
)
def test_factor_model_containers(factors_data, cls, name, kind):
    portfolios, factors = factors_data
    values = {"portfolios": portfolios, "factors": factors}
    expected = cls(**values)
    values[name] = make(values[name], kind)
    mod = cls(**values)
    assert_array_equal(mod.portfolios.ndarray, expected.portfolios.ndarray)
    assert_array_equal(mod.factors.ndarray, expected.factors.ndarray)


@pytest.mark.parametrize(
    "cls", [TradedFactorModel, LinearFactorModel], ids=FACTOR_IDS[:2]
)
def test_factor_model_lists_give_the_same_estimates(factors_data, cls):
    portfolios, factors = factors_data
    expected = cls(portfolios, factors).fit()
    res = cls(make(portfolios, "list"), make(factors, "list")).fit()
    assert_allclose(res.risk_premia, expected.risk_premia, rtol=1e-12)
    assert_allclose(res.params, expected.params, rtol=1e-12)
    assert_allclose(res.cov, expected.cov, rtol=1e-12)


@pytest.mark.parametrize("kind", ["list", "tuple", "ndarray", "dataframe"])
def test_factor_model_sigma_containers(factors_data, kind):
    portfolios, factors = factors_data
    sigma = pd.DataFrame(np.eye(4) + 0.25)
    expected = LinearFactorModel(portfolios, factors, sigma=sigma.to_numpy()).fit()
    new = make(sigma, "list2d" if kind == "list" else kind)
    res = LinearFactorModel(portfolios, factors, sigma=new).fit()
    assert_allclose(res.risk_premia, expected.risk_premia, rtol=1e-12)


@pytest.mark.parametrize(
    "cls", [TradedFactorModel, LinearFactorModel], ids=FACTOR_IDS[:2]
)
def test_factor_model_formula_portfolios_accept_lists(factors_data, cls):
    portfolios, factors = factors_data
    frame = pd.concat([factors, portfolios], axis=1)
    expected = cls.from_formula("f1 + f2", frame, portfolios=portfolios).fit()
    listed = make(portfolios, "list")
    res = cls.from_formula("f1 + f2", frame, portfolios=listed).fit()
    assert_allclose(res.risk_premia, expected.risk_premia, rtol=1e-12)


@pytest.mark.parametrize("cls", FACTOR_MODELS, ids=FACTOR_IDS)
@pytest.mark.parametrize("name", ["portfolios", "factors"])
@pytest.mark.parametrize("kind", COMPLEX_KINDS)
def test_factor_model_complex_input_raises(factors_data, cls, name, kind):
    portfolios, factors = factors_data
    values = {"portfolios": portfolios, "factors": factors}
    values[name] = complex_inputs(values[name])[kind]
    stem = name[:-1]
    with pytest.raises(TypeError, match=rf"{stem} has a complex dtype"):
        cls(**values)


@pytest.mark.parametrize("name", ["portfolios", "factors"])
def test_factor_model_text_input_raises(factors_data, name):
    portfolios, factors = factors_data
    values = {"portfolios": portfolios, "factors": factors}
    values[name] = np.full(values[name].shape, "abc")
    with pytest.raises(TypeError, match=rf"{name[:-1]} must contain numbers"):
        TradedFactorModel(**values)


# --------------------------------------------------------------------------- #
# Panel models
# --------------------------------------------------------------------------- #
PN, PT = 12, 6
PANEL_MODELS = [
    PooledOLS,
    PanelOLS,
    RandomEffects,
    BetweenOLS,
    FirstDifferenceOLS,
    FamaMacBeth,
]
PANEL_IDS = [c.__name__ for c in PANEL_MODELS]
# FamaMacBeth does not have clustered covariance estimators
CLUSTER_MODELS = PANEL_MODELS[:-1]
CLUSTER_IDS = PANEL_IDS[:-1]
# The estimators with a different treatment of the data and of the clusters
WEIGHT_MODELS = [PooledOLS, PanelOLS]
WEIGHT_IDS = ["PooledOLS", "PanelOLS"]
PATH_MODELS = [PooledOLS, BetweenOLS, FirstDifferenceOLS]
PATH_IDS = ["PooledOLS", "BetweenOLS", "FirstDifferenceOLS"]


def panel_data(seed=5):
    rng = np.random.default_rng(seed)
    entities = [f"e{i:02d}" for i in range(PN)]
    index = pd.MultiIndex.from_product([entities, range(PT)], names=["entity", "time"])
    nrows = PN * PT
    exog = pd.DataFrame(
        rng.standard_normal((nrows, 2)), index=index, columns=["x1", "x2"]
    )
    alpha = np.repeat(rng.standard_normal(PN), PT)
    dependent = pd.Series(
        alpha + exog.x1 - 0.5 * exog.x2 + rng.standard_normal(nrows),
        index=index,
        name="y",
    )
    weights = pd.Series(
        np.exp(0.3 * rng.standard_normal(nrows)), index=index, name="wt"
    )
    # Constant within an entity, so that every estimator accepts them
    entity_group = np.repeat(np.arange(PN) % 4, PT)
    clusters = pd.Series(entity_group, index=index, name="cl")
    clusters2 = pd.DataFrame(
        {"c1": entity_group, "c2": np.tile(np.arange(PT) % 3, PN)}, index=index
    )
    other = pd.DataFrame({"oe": (np.arange(nrows) * 7) % 5}, index=index)
    return {
        "dependent": dependent,
        "exog": exog,
        "weights": weights,
        "clusters": clusters,
        "clusters2": clusters2,
        "other": other,
        "time_weights": pd.Series(np.exp(0.3 * rng.standard_normal(PT)), name="tw"),
        "entity_weights": pd.Series(
            np.exp(0.3 * rng.standard_normal(PN)), index=entities, name="ew"
        ),
    }


@pytest.fixture(scope="module")
def panel():
    return panel_data()


def grid(value):
    """A time by entity array of a variable of a panel"""
    series = value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value
    return series.to_numpy().reshape(PN, PT).T


def cube(value):
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    return np.stack([grid(frame[c]) for c in frame])


def make_panel(value, kind):
    """A variable, or the variables, of a panel in another container"""
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    series = value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value
    entities = list(series.index.get_level_values(0).unique())
    times = list(range(PT))
    makers = {
        "series": lambda: series,
        "dataframe": lambda: frame,
        "panel_data": lambda: PanelData(frame, convert_dummies=False),
        "grid": lambda: grid(series),
        "grid3d": lambda: grid(series)[None],
        "cube": lambda: cube(frame),
        "wide": lambda: pd.DataFrame(grid(series), index=times, columns=entities),
        "dataarray_grid": lambda: xr.DataArray(
            grid(series),
            dims=["time", "entity"],
            coords={"time": times, "entity": entities},
        ),
        "dataarray_cube": lambda: xr.DataArray(
            cube(frame),
            dims=["var", "time", "entity"],
            coords={"var": list(frame.columns), "time": times, "entity": entities},
        ),
    }
    return makers[kind]()


def build_panel(cls, panel, **replace):
    values = {
        "dependent": panel["dependent"],
        "exog": panel["exog"],
        "weights": None,
        **replace,
    }
    kw = {"entity_effects": True} if cls is PanelOLS else {}
    return cls(values["dependent"], values["exog"], weights=values["weights"], **kw)


def panel_arrays(mod):
    """The numbers that are used in estimation"""
    return [mod.dependent.values2d, mod.exog.values2d, mod.weights.values2d]


def assert_same_panel(mod, other):
    for new, old in zip(panel_arrays(mod), panel_arrays(other), strict=True):
        assert_array_equal(new, old)


@pytest.fixture(scope="module")
def panel_reference(panel):
    return {cls: build_panel(cls, panel) for cls in PANEL_MODELS}


DEPENDENT_KINDS = [
    "series",
    "dataframe",
    "panel_data",
    "grid",
    "grid3d",
    "wide",
    *(["dataarray_grid"] if HAS_XARRAY else []),
]
EXOG_KINDS = [
    "dataframe",
    "panel_data",
    "cube",
    *(["dataarray_cube"] if HAS_XARRAY else []),
]
PANEL_TYPE_ERROR = r"{name}: Only ndarrays, DataFrames or DataArrays are supported"


@pytest.mark.parametrize("cls", PANEL_MODELS, ids=PANEL_IDS)
@pytest.mark.parametrize("kind", DEPENDENT_KINDS)
def test_panel_dependent_containers(panel, panel_reference, cls, kind):
    mod = build_panel(cls, panel, dependent=make_panel(panel["dependent"], kind))
    assert_same_panel(mod, panel_reference[cls])


@pytest.mark.parametrize("cls", PANEL_MODELS, ids=PANEL_IDS)
@pytest.mark.parametrize("kind", EXOG_KINDS)
def test_panel_exog_containers(panel, panel_reference, cls, kind):
    mod = build_panel(cls, panel, exog=make_panel(panel["exog"], kind))
    assert_same_panel(mod, panel_reference[cls])


@pytest.mark.parametrize("kind", DEPENDENT_KINDS)
def test_panel_dependent_containers_give_the_same_estimates(panel, kind):
    expected = estimates(build_panel(PooledOLS, panel).fit())
    mod = build_panel(PooledOLS, panel, dependent=make_panel(panel["dependent"], kind))
    assert_allclose(estimates(mod.fit()), expected, rtol=1e-10)


@pytest.mark.parametrize("kind", EXOG_KINDS)
def test_panel_exog_containers_give_the_same_estimates(panel, kind):
    expected = estimates(build_panel(PooledOLS, panel).fit())
    mod = build_panel(PooledOLS, panel, exog=make_panel(panel["exog"], kind))
    assert_allclose(estimates(mod.fit()), expected, rtol=1e-10)


@pytest.mark.parametrize("cls", PANEL_MODELS, ids=PANEL_IDS)
def test_panel_numpy_dependent_and_labelled_exog(panel, panel_reference, cls):
    # An array has no labels, and so the rows are paired by position. The
    # labels of the other input are used by the estimators
    expected = estimates(panel_reference[cls].fit())
    mod = build_panel(cls, panel, dependent=grid(panel["dependent"]))
    assert_allclose(estimates(mod.fit()), expected, rtol=1e-10)
    mod = build_panel(cls, panel, exog=cube(panel["exog"]))
    assert_allclose(estimates(mod.fit()), expected, rtol=1e-10)


@pytest.mark.parametrize("cls", PANEL_MODELS, ids=PANEL_IDS)
@pytest.mark.parametrize("name", ["dependent", "exog"])
@pytest.mark.parametrize("kind", ["list", "tuple", "nested"])
def test_panel_data_rejects_lists_and_names_the_input(panel, cls, name, kind):
    array = grid(panel[name]) if name == "dependent" else cube(panel[name])
    value = {"list": panel[name].to_numpy().tolist(), "nested": array.tolist()}
    value["tuple"] = tuple(value["list"])
    with pytest.raises(TypeError, match=PANEL_TYPE_ERROR.format(name=name)) as exc:
        build_panel(cls, panel, **{name: value[kind]})
    assert "Got list" in str(exc.value) or "Got tuple" in str(exc.value)


@pytest.mark.parametrize("name", ["dependent", "exog"])
def test_panel_data_rejects_other_types_and_names_the_input(panel, name):
    for value in (5.0, "abc", {"a": 1}):
        with pytest.raises(TypeError, match=PANEL_TYPE_ERROR.format(name=name)):
            build_panel(PooledOLS, panel, **{name: value})


@pytest.mark.parametrize("name", ["dependent", "exog"])
def test_panel_array_with_the_wrong_dimensions_names_the_input(panel, name):
    with pytest.raises(ValueError, match=rf"{name}: 2 or 3-d array required"):
        build_panel(PooledOLS, panel, **{name: np.zeros(PN * PT)})
    with pytest.raises(ValueError, match=rf"{name}: 2 or 3-d array required"):
        build_panel(PooledOLS, panel, **{name: np.zeros((2, 2, 2, 2))})


@pytest.mark.parametrize("cls", PANEL_MODELS, ids=PANEL_IDS)
@pytest.mark.parametrize("name", ["dependent", "exog", "weights"])
def test_panel_complex_input_raises(panel, cls, name):
    base = {"dependent": grid, "exog": cube, "weights": grid}[name]
    source = panel[name]
    for value in (base(source).astype(complex), source.astype(complex)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match=rf"{name}.*complex dtype"):
                build_panel(cls, panel, **{name: value})


def test_panel_text_input_raises_and_names_the_input(panel):
    with pytest.raises(TypeError, match=r"dependent: .*must contain numbers"):
        build_panel(PooledOLS, panel, dependent=np.full((PT, PN), "abc"))
    with pytest.raises(TypeError, match="weights must contain numbers"):
        build_panel(PooledOLS, panel, weights=np.full(PN * PT, "abc"))


def test_panel_predict_rejects_lists_and_names_the_input(panel):
    mod = build_panel(PooledOLS, panel)
    res = mod.fit()
    exog = panel["exog"]
    expected = mod.predict(res.params, exog=exog)
    for kind in ("dataframe", "panel_data", "cube"):
        new = mod.predict(res.params, exog=make_panel(exog, kind))
        assert_allclose(new.to_numpy(), expected.to_numpy())
        new = res.predict(exog=make_panel(exog, kind))
        assert_allclose(new.to_numpy(), expected.to_numpy())
    for value in (exog.to_numpy().tolist(), cube(exog).tolist()):
        with pytest.raises(TypeError, match="exog: Only ndarrays, DataFrames"):
            mod.predict(res.params, exog=value)
        with pytest.raises(TypeError, match="exog: Only ndarrays, DataFrames"):
            res.predict(exog=value)
    with pytest.raises(TypeError, match=r"exog.*complex dtype"):
        res.predict(exog=cube(exog).astype(complex))


def test_panel_data_class_still_rejects_lists():
    with pytest.raises(TypeError, match="Only ndarrays, DataFrames or DataArrays"):
        PanelData([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError, match="Got list"):
        PanelData([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError, match="x has a complex dtype"):
        PanelData(np.ones((2, 2), dtype=complex))


def test_panel_formula_column_name_as_weights_and_bad_data(panel):
    data = pd.concat([panel["dependent"], panel["exog"], panel["weights"]], axis=1)
    formula = "y ~ 1 + x1 + x2"
    with pytest.raises(TypeError, match=r"weights must be array_like.*data\['wt'\]"):
        PooledOLS.from_formula(formula, data, weights="wt")
    mod = PooledOLS.from_formula(formula, data, weights=data["wt"])
    assert mod.weights.values2d.shape == (PN * PT, 1)
    with pytest.raises(TypeError, match="data: Only ndarrays, DataFrames"):
        PooledOLS.from_formula(formula, data.to_numpy().tolist())


WEIGHT_KINDS = [
    "series",
    "dataframe",
    "panel_data",
    "list",
    "tuple",
    "ndarray",
    "grid",
    "grid_list",
    "wide",
    "grid3d",
    *(["dataarray_grid"] if HAS_XARRAY else []),
]


def make_weights(value, kind):
    if kind == "list":
        return value.tolist()
    if kind == "tuple":
        return tuple(value.tolist())
    if kind == "ndarray":
        return value.to_numpy()
    if kind == "grid_list":
        return grid(value).tolist()
    return make_panel(value, kind)


@pytest.mark.parametrize("cls", WEIGHT_MODELS, ids=WEIGHT_IDS)
@pytest.mark.parametrize("kind", WEIGHT_KINDS)
def test_panel_weights_containers(panel, cls, kind):
    expected = build_panel(cls, panel, weights=panel["weights"])
    mod = build_panel(cls, panel, weights=make_weights(panel["weights"], kind))
    assert_same_panel(mod, expected)


@pytest.mark.parametrize("kind", ["list", "tuple", "grid_list"])
def test_panel_weights_lists_give_the_same_estimates(panel, kind):
    expected = estimates(build_panel(PanelOLS, panel, weights=panel["weights"]).fit())
    weights = make_weights(panel["weights"], kind)
    mod = build_panel(PanelOLS, panel, weights=weights)
    assert_allclose(estimates(mod.fit()), expected, rtol=1e-10)


@pytest.mark.parametrize("cls", WEIGHT_MODELS, ids=WEIGHT_IDS)
@pytest.mark.parametrize("kind", ["series", "list", "tuple", "ndarray"])
@pytest.mark.parametrize("shape", ["time_weights", "entity_weights"])
def test_panel_weights_of_each_time_or_entity(panel, cls, kind, shape):
    value = panel[shape]
    expected = build_panel(cls, panel, weights=value)
    new = value if kind == "series" else make(value, kind)
    mod = build_panel(cls, panel, weights=new)
    assert_same_panel(mod, expected)


def test_panel_weights_of_each_time_are_used_for_every_entity(panel):
    mod = build_panel(PooledOLS, panel, weights=panel["time_weights"].tolist())
    expected = np.tile(panel["time_weights"].to_numpy(), PN)
    assert_allclose(mod.weights.values2d[:, 0], expected / expected.mean())
    mod = build_panel(PooledOLS, panel, weights=panel["entity_weights"].tolist())
    expected = np.repeat(panel["entity_weights"].to_numpy(), PT)
    assert_allclose(mod.weights.values2d[:, 0], expected / expected.mean())


# --------------------------------------------------------------------------- #
# Panel clusters and effects
# --------------------------------------------------------------------------- #
def make_panel_clusters(value, kind):
    series = value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value
    text = "g" + series.astype(int).astype(str)
    makers = {
        "list": series.tolist,
        "tuple": lambda: tuple(series.tolist()),
        "ndarray": series.to_numpy,
        "strings_ndarray": text.to_numpy,
        "strings_list": text.tolist,
        "grid_list": lambda: grid(series).tolist(),
        "cube_list": lambda: cube(value).tolist(),
    }
    if kind in makers:
        return makers[kind]()
    return make_panel(value, kind)


def setup_clusters(mod, clusters, **kw):
    """The clusters that the estimator passes to the covariance estimator"""
    return np.asarray(mod._setup_clusters({"clusters": clusters, **kw})["clusters"])


ONE_D_KINDS = ["list", "tuple", "ndarray"]
LABELLED_CLUSTER_KINDS = [
    "series",
    "dataframe",
    "panel_data",
    "wide",
    *(["dataarray_grid"] if HAS_XARRAY else []),
]
ARRAY_CLUSTER_KINDS = ["grid", "grid3d", "grid_list"]
TEXT_CLUSTER_KINDS = ["strings_ndarray", "strings_list"]


@pytest.mark.parametrize("cls", PATH_MODELS, ids=PATH_IDS)
@pytest.mark.parametrize("kind", [*ONE_D_KINDS, *LABELLED_CLUSTER_KINDS])
def test_panel_clusters_containers(panel, panel_reference, cls, kind):
    mod = panel_reference[cls]
    expected = setup_clusters(mod, panel["clusters"])
    clusters = make_panel_clusters(panel["clusters"], kind)
    assert_array_equal(setup_clusters(mod, clusters), expected)


@pytest.mark.parametrize("cls", PATH_MODELS, ids=PATH_IDS)
@pytest.mark.parametrize("kind", ARRAY_CLUSTER_KINDS)
def test_panel_array_clusters_containers(panel, panel_reference, cls, kind):
    # Arrays have the layout of the data, and no labels
    mod = panel_reference[cls]
    expected = setup_clusters(mod, panel["clusters"])
    clusters = make_panel_clusters(panel["clusters"], kind)
    assert_array_equal(setup_clusters(mod, clusters), expected)


@pytest.mark.parametrize("cls", PATH_MODELS, ids=PATH_IDS)
@pytest.mark.parametrize("kind", TEXT_CLUSTER_KINDS)
def test_panel_text_clusters_containers(panel, panel_reference, cls, kind):
    mod = panel_reference[cls]
    expected = setup_clusters(mod, panel["clusters"])
    clusters = make_panel_clusters(panel["clusters"], kind)
    assert_array_equal(setup_clusters(mod, clusters), expected)


@pytest.mark.parametrize("cls", CLUSTER_MODELS, ids=CLUSTER_IDS)
@pytest.mark.parametrize("kind", ["list", "ndarray"])
def test_panel_one_dimensional_clusters_give_the_same_estimates(panel, cls, kind):
    mod = build_panel(cls, panel)
    expected = estimates(mod.fit(cov_type="clustered", clusters=panel["clusters"]))
    clusters = make_panel_clusters(panel["clusters"], kind)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    assert_allclose(estimates(res), expected, rtol=1e-10)


@pytest.mark.parametrize("cls", [PooledOLS, PanelOLS, RandomEffects], ids=PANEL_IDS[:3])
@pytest.mark.parametrize("kind", ["dataframe", "panel_data", "cube", "cube_list"])
def test_panel_two_way_clusters_containers(panel, panel_reference, cls, kind):
    mod = panel_reference[cls]
    expected = setup_clusters(mod, panel["clusters2"])
    clusters = make_panel_clusters(panel["clusters2"], kind)
    assert_array_equal(setup_clusters(mod, clusters), expected)


def test_panel_two_way_clusters_from_arrays_give_the_same_estimates(panel):
    mod = build_panel(PooledOLS, panel)
    expected = mod.fit(cov_type="clustered", clusters=panel["clusters2"])
    res = mod.fit(cov_type="clustered", clusters=cube(panel["clusters2"]))
    assert_allclose(res.cov, expected.cov, rtol=1e-10)
    res = mod.fit(cov_type="clustered", clusters=cube(panel["clusters2"]).tolist())
    assert_allclose(res.cov, expected.cov, rtol=1e-10)


@pytest.mark.parametrize("cls", CLUSTER_MODELS, ids=CLUSTER_IDS)
def test_panel_one_dimensional_clusters_must_have_a_value_for_each_row(
    panel_reference, cls
):
    mod = panel_reference[cls]
    with pytest.raises(ValueError, match="clusters is 1-d and has"):
        mod.fit(cov_type="clustered", clusters=np.arange(PN * PT - 1))
    with pytest.raises(ValueError, match="clusters is 1-d and has"):
        mod.fit(cov_type="clustered", clusters=list(range(PN)))


def test_panel_clusters_with_missing_values_in_the_model(panel):
    # The clusters have a value for each row of dependent, including the rows
    # that are dropped because they have a missing value
    dependent = panel["dependent"].copy()
    dependent.iloc[[3, 30]] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = PooledOLS(dependent, panel["exog"])
    expected = mod.fit(cov_type="clustered", clusters=panel["clusters"])
    for clusters in (panel["clusters"].to_numpy(), panel["clusters"].tolist()):
        res = mod.fit(cov_type="clustered", clusters=clusters)
        assert_allclose(res.cov, expected.cov, rtol=1e-10)


@pytest.mark.parametrize("option", ["cluster_entity", "cluster_time"])
@pytest.mark.parametrize("cls", [PooledOLS, PanelOLS, RandomEffects], ids=PANEL_IDS[:3])
@pytest.mark.parametrize("kind", ["grid", "ndarray", "list"])
def test_panel_unlabelled_clusters_with_entity_or_time_clusters(
    panel, panel_reference, cls, kind, option
):
    # The clusters from an array have the labels of the rows of dependent, and so
    # they can be combined with the clusters from the entities or periods
    mod = panel_reference[cls]
    expected = setup_clusters(mod, panel["clusters"], **{option: True})
    assert expected.shape == (PN * PT, 2)
    clusters = (
        grid(panel["clusters"])
        if kind == "grid"
        else make_panel_clusters(panel["clusters"], kind)
    )
    assert_array_equal(setup_clusters(mod, clusters, **{option: True}), expected)


@pytest.mark.parametrize("cls", CLUSTER_MODELS, ids=CLUSTER_IDS)
def test_panel_text_and_categorical_clusters_are_labels(panel, panel_reference, cls):
    # Clusters that are not numbers are coerced to integers as categories
    mod = panel_reference[cls]
    expected = setup_clusters(mod, panel["clusters"])
    text = "g" + panel["clusters"].astype(int).astype(str)
    for value in (text, text.astype("category")):
        assert_array_equal(setup_clusters(mod, value), expected)


@pytest.mark.parametrize("cls", CLUSTER_MODELS, ids=CLUSTER_IDS)
def test_panel_complex_clusters_raise(panel_reference, panel, cls):
    mod = panel_reference[cls]
    clusters = panel["clusters"].to_numpy().astype(complex)
    with pytest.raises(TypeError, match=r"clusters.*complex dtype"):
        mod.fit(cov_type="clustered", clusters=clusters)


def test_panel_three_cluster_variables_raise(panel_reference, panel):
    mod = panel_reference[PooledOLS]
    clusters = panel["clusters2"].assign(c3=panel["clusters"])
    with pytest.raises(ValueError, match="Only 1 or 2-way clustering supported"):
        mod.fit(cov_type="clustered", clusters=clusters)


OTHER_KINDS = ["dataframe", "series", "grid", "list", "tuple", "ndarray", "grid_list"]


@pytest.mark.parametrize("kind", OTHER_KINDS)
def test_panel_other_effects_containers(panel, kind):
    expected = PanelOLS(panel["dependent"], panel["exog"], other_effects=panel["other"])
    other = panel["other"]
    if kind in ("list", "tuple", "ndarray", "grid_list"):
        value = make_panel_clusters(other, kind)
    else:
        value = make_panel(other, kind)
    mod = PanelOLS(panel["dependent"], panel["exog"], other_effects=value)
    assert_same_panel(mod, expected)
    assert_array_equal(
        mod._other_effect_cats.values2d, expected._other_effect_cats.values2d
    )


def test_panel_other_effects_lists_give_the_same_estimates(panel):
    expected = PanelOLS(
        panel["dependent"], panel["exog"], other_effects=panel["other"]
    ).fit()
    value = panel["other"]["oe"].tolist()
    res = PanelOLS(panel["dependent"], panel["exog"], other_effects=value).fit()
    assert_allclose(estimates(res), estimates(expected), rtol=1e-10)


def test_panel_other_effects_errors_name_the_input(panel):
    with pytest.raises(ValueError, match="other_effects is 1-d and has"):
        PanelOLS(panel["dependent"], panel["exog"], other_effects=np.arange(PN))
    with pytest.raises(TypeError, match=r"other_effects.*complex dtype"):
        PanelOLS(
            panel["dependent"],
            panel["exog"],
            other_effects=panel["other"].to_numpy()[:, 0].astype(complex),
        )


def test_reformat_clusters_accepts_lists(panel_reference, panel):
    mod = panel_reference[PooledOLS]
    expected = mod.reformat_clusters(panel["clusters"])
    for value in (panel["clusters"].tolist(), panel["clusters"].to_numpy()):
        new = mod.reformat_clusters(value)
        assert_array_equal(new.values2d, expected.values2d)
        assert new.index.equals(expected.index)
