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
