"""
Inputs that carry labels (pandas, xarray) have to use the labels of dependent.

The system estimators pair the rows of their inputs by position, both within
an equation and across the equations: the residuals of all equations in the
same row are assumed to be the errors of the same observation. A labelled input
whose rows are in a different order from dependent would silently be attached
to the wrong observations. These tests check that this is rejected, that inputs
that carry no labels (NumPy arrays and pandas objects with a default index) are
paired by position, and that inputs that use the labels of dependent are
accepted. If no dependent variable carries labels, the first of the other inputs
that does sets the labels, and the inputs after it have to use them.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv.data import IVData
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 100
LABELS = ["eq1", "eq2", "eq3"]
INDEX_STYLES = ["str", "datetime", "int", "multi"]
# (model, kind of equations, inputs of each equation that can be misaligned)
SUR_INPUTS = ["exog", "weights"]
IV_INPUTS = ["exog", "endog", "instruments", "weights"]
MODELS = [
    pytest.param(SUR, "sur", SUR_INPUTS, id="SUR"),
    pytest.param(IV3SLS, "iv", IV_INPUTS, id="IV3SLS"),
    pytest.param(IVSystemGMM, "iv", IV_INPUTS, id="IVSystemGMM"),
]
LS_MODELS = [
    pytest.param(SUR, "sur", id="SUR"),
    pytest.param(IV3SLS, "iv", id="IV3SLS"),
]


def make_index(style, nobs=NOBS):
    rng = np.random.default_rng(99)
    if style == "str":
        return pd.Index([f"r{i}" for i in range(nobs)], name="obs")
    if style == "datetime":
        return pd.date_range("2000-01-01", periods=nobs, freq="D", name="date")
    if style == "int":
        return pd.Index(rng.permutation(10 * nobs)[:nobs] + 7, name="id")
    return pd.MultiIndex.from_product(
        [np.arange(nobs // 10), np.arange(10)], names=["firm", "year"]
    )


class System:
    """Labelled data for a system. All of the pandas objects share an index"""

    def __init__(self, style="str", nobs=NOBS, seed=4):
        rng = np.random.default_rng(seed)
        self.index = idx = make_index(style, nobs)
        self.nobs = nobs
        common = rng.standard_normal(nobs)
        self.data = {}
        for k, label in enumerate(LABELS):
            instruments = pd.DataFrame(
                rng.standard_normal((nobs, 2)),
                index=idx,
                columns=[f"z{k}_a", f"z{k}_b"],
            )
            endog = pd.DataFrame(
                {f"e{k}": instruments.iloc[:, 0] + instruments.iloc[:, 1]}, index=idx
            )
            endog += rng.standard_normal((nobs, 1))
            exog = pd.DataFrame(
                {"const": 1.0, f"x{k}": rng.standard_normal(nobs)}, index=idx
            )
            noise = rng.standard_normal(nobs) + common
            dependent = pd.Series(
                1 + exog.iloc[:, 1] + endog.iloc[:, 0] + noise, index=idx, name=f"y{k}"
            )
            weights = pd.Series(
                np.exp(rng.standard_normal(nobs)), index=idx, name=f"w{k}"
            )
            self.data[label] = {
                "dependent": dependent,
                "exog": exog,
                "endog": endog,
                "instruments": instruments,
                "weights": weights,
            }
        self.clusters = pd.Series(rng.integers(0, 10, nobs), index=idx, name="cl")

    def frame(self):
        """A single frame with all of the variables"""
        parts = [value for eq in self.data.values() for value in eq.values()]
        frame = pd.concat(parts, axis=1)
        return frame.loc[:, ~frame.columns.duplicated()]


@pytest.fixture(params=INDEX_STYLES)
def system(request):
    return System(request.param)


@pytest.fixture
def simple():
    return System("str")


def reverse(value):
    """The same labels in the opposite order"""
    return value.iloc[::-1]


def relabel(value):
    """The same values, with the labels attached to the wrong rows"""
    out = value.copy()
    out.index = value.index[np.roll(np.arange(len(value)), 1)]
    return out


CORRUPTIONS = {"reverse": reverse, "relabel": relabel}


def equations(system, kind, *, weights=False, replace=None, form="dict", raw=False):
    """
    Equations of a system

    kind is "sur" or "iv". replace is a dictionary with keys (label, input) and
    values that replace the input in the equation. raw produces NumPy arrays.
    """
    names = ["dependent", "exog"] + (["endog", "instruments"] if kind == "iv" else [])
    if weights:
        names.append("weights")
    replace = {} if replace is None else replace
    out = {}
    for label in LABELS:
        values = {}
        for name in names:
            value = replace.get((label, name), system.data[label][name])
            if raw and (label, name) not in replace:
                value = value.to_numpy()
            values[name] = value
        out[label] = tuple(values.values()) if form == "tuple" else values
    return out


def arrays(mod):
    """Arrays used to estimate the model"""
    return [*mod._y, *mod._x, *mod._z, *mod._w]


def assert_same_arrays(left, right):
    for a, b in zip(arrays(left), arrays(right), strict=True):
        assert_allclose(a, b)


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
@pytest.mark.parametrize("form", ["dict", "tuple"])
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
@pytest.mark.parametrize("label", LABELS)
def test_misaligned_input_rejected(system, cls, kind, inputs, label, corruption, form):
    # The model is accepted when all of the inputs use the labels of dependent
    cls(equations(system, kind, weights=True, form=form))
    for name in inputs:
        original = system.data[label][name]
        bad = CORRUPTIONS[corruption](original)
        replace = {(label, name): bad}
        eqns = equations(system, kind, weights=True, replace=replace, form=form)
        match = (
            f"row index of {name} in equation '{label}' does not match "
            f"dependent in equation '{label}'"
        )
        with pytest.raises(ValueError, match=match):
            cls(eqns)


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_message_names_input_and_remedy(simple, cls, kind, inputs):
    replace = {("eq2", "exog"): reverse(simple.data["eq2"]["exog"])}
    with pytest.raises(ValueError, match="does not match") as exc:
        cls(equations(simple, kind, replace=replace))
    message = str(exc.value)
    assert "exog in equation 'eq2'" in message
    assert "dependent in equation 'eq2'" in message
    assert "reindex" in message
    assert ".to_numpy()" in message


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
@pytest.mark.parametrize("label", LABELS[1:])
def test_equations_must_share_the_index(simple, cls, kind, inputs, label, corruption):
    # Only dependent is different from the first equation
    corrupt = CORRUPTIONS[corruption]
    replace = {(label, "dependent"): corrupt(simple.data[label]["dependent"])}
    match = (
        f"row index of dependent in equation '{label}' does not match "
        "dependent in equation 'eq1'"
    )
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, replace=replace))
    # An equation where everything has the order of its own dependent variable
    replace = {
        (label, name): corrupt(simple.data[label][name])
        for name in ["dependent", "exog", "endog", "instruments", "weights"]
        if (kind == "iv" or name in ("dependent", "exog", "weights"))
    }
    eqns = equations(simple, kind, weights=True, replace=replace)
    with pytest.raises(ValueError, match=match):
        cls(eqns)


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_first_equation_is_the_reference(simple, cls, kind, inputs):
    # eq1 is reversed, and the other equations do not match it
    replace = {
        ("eq1", name): reverse(simple.data["eq1"][name])
        for name in ["dependent", "exog", "endog", "instruments"]
        if (kind == "iv" or name in ("dependent", "exog"))
    }
    with pytest.raises(ValueError, match="dependent in equation 'eq1'"):
        cls(equations(simple, kind, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
@pytest.mark.parametrize("name", ["exog", "weights"])
def test_numpy_input_is_paired_by_position(simple, cls, kind, inputs, name):
    # An array has no labels, and so it is paired by position
    values = simple.data["eq2"][name].to_numpy()[::-1]
    replace = {("eq2", name): values}
    mod = cls(equations(simple, kind, weights=True, replace=replace))
    raw_replace = {("eq2", name): values}
    raw = cls(equations(simple, kind, weights=True, replace=raw_replace, raw=True))
    assert_same_arrays(mod, raw)
    # The values of eq2 are those of the opposite rows
    if name == "exog":
        assert_allclose(mod._exog[1].ndarray, values)


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_default_index_is_paired_by_position(simple, cls, kind, inputs):
    exog = simple.data["eq2"]["exog"]
    flipped = pd.DataFrame(exog.to_numpy()[::-1], columns=exog.columns)
    assert isinstance(flipped.index, pd.RangeIndex)
    weights = pd.Series(simple.data["eq3"]["weights"].to_numpy()[::-1])
    replace = {("eq2", "exog"): flipped, ("eq3", "weights"): weights}
    mod = cls(equations(simple, kind, weights=True, replace=replace))
    raw_replace = {
        ("eq2", "exog"): exog.to_numpy()[::-1],
        ("eq3", "weights"): weights.to_numpy(),
    }
    raw = cls(equations(simple, kind, weights=True, replace=raw_replace, raw=True))
    assert_same_arrays(mod, raw)
    # A reversed range is not a default index
    replace = {("eq2", "exog"): flipped.iloc[::-1]}
    with pytest.raises(ValueError, match="row index of exog in equation 'eq2'"):
        cls(equations(simple, kind, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
@pytest.mark.parametrize("form", ["dict", "tuple"])
def test_same_labels_match_numpy(system, cls, kind, inputs, form):
    labelled = cls(equations(system, kind, weights=True, form=form))
    raw = cls(equations(system, kind, weights=True, form=form, raw=True))
    assert_same_arrays(labelled, raw)


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_same_labels_match_numpy_results(simple, cls, kind):
    labelled = cls(equations(simple, kind, weights=True)).fit()
    raw = cls(equations(simple, kind, weights=True, raw=True)).fit()
    assert_allclose(labelled.params, raw.params)
    assert_allclose(labelled.std_errors, raw.std_errors)


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_everything_reordered_together_is_allowed(simple, cls, kind, inputs):
    expected = cls(equations(simple, kind, weights=True))
    replace = {
        (label, name): reverse(simple.data[label][name])
        for label in LABELS
        for name in ["dependent", "exog", "endog", "instruments", "weights"]
        if (kind == "iv" or name in ("dependent", "exog", "weights"))
    }
    reordered = cls(equations(simple, kind, weights=True, replace=replace))
    for a, b in zip(arrays(expected), arrays(reordered), strict=True):
        assert_allclose(a, b[::-1])


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_unlabelled_dependent_is_paired_by_position(simple, cls, kind, inputs):
    # There are no labels in the dependent variable of eq1, so its inputs are
    # paired with it by position. They use the labels of the other equations
    dependent = simple.data["eq1"]["dependent"].to_numpy()
    replace = {("eq1", "dependent"): dependent}
    for label in LABELS:
        for name in ["dependent", *inputs]:
            replace.setdefault((label, name), reverse(simple.data[label][name]))
    mod = cls(equations(simple, kind, weights=True, replace=replace))
    assert_allclose(
        mod._exog[0].ndarray, reverse(simple.data["eq1"]["exog"]).to_numpy()
    )
    assert_allclose(mod._y[0], dependent[:, None])
    # The same is true of the other equations
    dependent = simple.data["eq2"]["dependent"].to_numpy()
    replace = {("eq2", "dependent"): dependent}
    for label in LABELS:
        for name in ["dependent", *inputs]:
            replace.setdefault((label, name), reverse(simple.data[label][name]))
    mod = cls(equations(simple, kind, weights=True, replace=replace))
    assert_allclose(mod._y[1], dependent[:, None])


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
@pytest.mark.parametrize("label", LABELS)
def test_inputs_use_the_labels_of_the_system(
    simple, cls, kind, inputs, label, corruption
):
    # The dependent variable of the equation has no labels. Its inputs have to use
    # the labels of the dependent variables of the other equations
    reference = "eq2" if label == "eq1" else "eq1"
    arrays_ = {(label, "dependent"): simple.data[label]["dependent"].to_numpy()}
    cls(equations(simple, kind, weights=True, replace=arrays_))
    for name in inputs:
        bad = CORRUPTIONS[corruption](simple.data[label][name])
        replace = {**arrays_, (label, name): bad}
        match = (
            f"row index of {name} in equation '{label}' does not match "
            f"dependent in equation '{reference}'"
        )
        with pytest.raises(ValueError, match=match):
            cls(equations(simple, kind, weights=True, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_first_input_with_labels_sets_the_labels(simple, cls, kind, inputs, corruption):
    # No dependent variable has labels, so the first input with labels sets them
    arrays_ = {
        (label, "dependent"): simple.data[label]["dependent"].to_numpy()
        for label in LABELS
    }
    cls(equations(simple, kind, weights=True, replace=arrays_))
    for label in LABELS:
        for name in inputs:
            if (label, name) == ("eq1", "exog"):
                continue
            bad = CORRUPTIONS[corruption](simple.data[label][name])
            replace = {**arrays_, (label, name): bad}
            match = (
                f"row index of {name} in equation '{label}' does not match "
                "exog in equation 'eq1'"
            )
            with pytest.raises(ValueError, match=match):
                cls(equations(simple, kind, weights=True, replace=replace))
    # If it is exog in eq1 that has changed, the next input is the one that
    # does not match
    bad = CORRUPTIONS[corruption](simple.data["eq1"]["exog"])
    replace = {**arrays_, ("eq1", "exog"): bad}
    second = "endog" if kind == "iv" else "weights"
    match = (
        f"row index of {second} in equation 'eq1' does not match "
        "exog in equation 'eq1'"
    )
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, weights=True, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_inputs_without_labels_do_not_set_the_labels(simple, cls, kind, inputs):
    # Nothing in eq1 has labels, and so exog in eq2 is the first input with labels
    arrays_ = {
        (label, "dependent"): simple.data[label]["dependent"].to_numpy()
        for label in LABELS
    }
    for name in inputs:
        arrays_[("eq1", name)] = simple.data["eq1"][name].to_numpy()
    cls(equations(simple, kind, weights=True, replace=arrays_))
    bad = {**arrays_, ("eq3", "exog"): reverse(simple.data["eq3"]["exog"])}
    match = "row index of exog in equation 'eq3' does not match exog in equation 'eq2'"
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, weights=True, replace=bad))
    # A default index has no labels, and so the next input sets them
    exog = simple.data["eq2"]["exog"]
    flipped = pd.DataFrame(exog.to_numpy()[::-1], columns=exog.columns)
    replace = {**bad, ("eq2", "exog"): flipped}
    second = "endog" if kind == "iv" else "weights"
    match = (
        f"row index of exog in equation 'eq3' does not match {second} in "
        "equation 'eq2'"
    )
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, weights=True, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_inputs_are_compared_to_the_dependent_of_the_equation(
    simple, cls, kind, inputs
):
    # eq1 does not have labels. eq2 and eq3 are consistent with themselves
    arrays_ = {("eq1", "dependent"): simple.data["eq1"]["dependent"].to_numpy()}
    cls(equations(simple, kind, replace=arrays_))
    # eq2 is not consistent with itself
    replace = {**arrays_, ("eq2", "exog"): reverse(simple.data["eq2"]["exog"])}
    with pytest.raises(ValueError, match="row index of exog in equation 'eq2'"):
        cls(equations(simple, kind, replace=replace))
    # eq3 dependent does not have the labels of eq2 dependent, which is the first
    # dependent with labels
    replace = {
        **arrays_,
        ("eq3", "dependent"): reverse(simple.data["eq3"]["dependent"]),
    }
    match = (
        "row index of dependent in equation 'eq3' does not match "
        "dependent in equation 'eq2'"
    )
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_index_names_are_ignored(simple, cls, kind, inputs):
    exog = simple.data["eq2"]["exog"].rename_axis("another name")
    mod = cls(equations(simple, kind, replace={("eq2", "exog"): exog}))
    assert_allclose(mod._exog[1].ndarray, simple.data["eq2"]["exog"].to_numpy())


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_different_lengths_are_not_labels(simple, cls, kind, inputs):
    # The number of observations in an equation is reported as before
    exog = simple.data["eq2"]["exog"].iloc[:-1]
    with pytest.raises(ValueError, match="Array required to have"):
        cls(equations(simple, kind, replace={("eq2", "exog"): exog}))
    # An equation with fewer observations than the others
    names = ["dependent", "exog"] + (["endog", "instruments"] if kind == "iv" else [])
    shorter = {("eq2", name): simple.data["eq2"][name].iloc[:-1] for name in names}
    match = (
        "All equations must have the same number of observations. The dependent "
        "variable in equation 'eq2' has 99 observations, but in equation 'eq1' "
        "it has 100."
    )
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, replace=shorter))
    # The same is true without labels
    values = {key: value.to_numpy() for key, value in shorter.items()}
    with pytest.raises(ValueError, match=match):
        cls(equations(simple, kind, replace=values))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_missing_values_do_not_change_the_check(simple, cls, kind, inputs):
    exog = simple.data["eq2"]["exog"].copy()
    exog.iloc[3, 1] = np.nan
    weights = simple.data["eq3"]["weights"].copy()
    weights.iloc[10] = np.nan
    replace = {("eq2", "exog"): exog, ("eq3", "weights"): weights}
    with pytest.warns(MissingValueWarning):
        mod = cls(equations(simple, kind, weights=True, replace=replace))
    assert mod._y[0].shape[0] == NOBS - 2
    replace[("eq1", "exog")] = reverse(simple.data["eq1"]["exog"])
    with pytest.raises(ValueError, match="row index of exog in equation 'eq1'"):
        cls(equations(simple, kind, weights=True, replace=replace))


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_ivdata_inputs(simple, cls, kind, inputs):
    exog = simple.data["eq2"]["exog"]
    cls(equations(simple, kind, replace={("eq2", "exog"): IVData(exog)}))
    with pytest.raises(ValueError, match="row index of exog in equation 'eq2'"):
        cls(equations(simple, kind, replace={("eq2", "exog"): IVData(reverse(exog))}))
    cls(
        equations(
            simple, kind, replace={("eq2", "exog"): IVData(exog.to_numpy()[::-1])}
        )
    )


@pytest.mark.parametrize(("cls", "kind", "inputs"), MODELS)
def test_xarray_inputs(simple, cls, kind, inputs):
    xr = pytest.importorskip("xarray")

    def to_xarray(frame):
        return xr.DataArray(
            frame.to_numpy(),
            dims=["obs", "var"],
            coords={"obs": frame.index.to_numpy(), "var": list(frame)},
        )

    exog = simple.data["eq2"]["exog"]
    replace = {("eq2", "exog"): to_xarray(exog)}
    mod = cls(equations(simple, kind, replace=replace))
    assert_allclose(mod._exog[1].ndarray, exog.to_numpy())
    flipped = to_xarray(exog).isel(obs=slice(None, None, -1))
    with pytest.raises(ValueError, match="row index of exog in equation 'eq2'"):
        cls(equations(simple, kind, replace={("eq2", "exog"): flipped}))
    # An xarray dependent sets the labels
    dep = to_xarray(simple.data["eq1"]["dependent"].to_frame())
    replace = {
        ("eq1", "dependent"): dep,
        ("eq3", "exog"): reverse(simple.data["eq3"]["exog"]),
    }
    with pytest.raises(ValueError, match="row index of exog in equation 'eq3'"):
        cls(equations(simple, kind, replace=replace))


# Multivariate interfaces


class Multivariate:
    def __init__(self, style="str"):
        system = System(style)
        names = LABELS
        self.dependent = pd.concat(
            [system.data[label]["dependent"] for label in names], axis=1
        )
        self.exog = system.data["eq1"]["exog"]
        self.endog = system.data["eq1"]["endog"]
        self.instruments = system.data["eq1"]["instruments"]


@pytest.fixture
def multi():
    return Multivariate("str")


def test_multivariate_ls(multi):
    mod = SUR.multivariate_ls(multi.dependent, multi.exog)
    raw = SUR.multivariate_ls(multi.dependent.to_numpy(), multi.exog.to_numpy())
    assert_same_arrays(mod, raw)
    with pytest.raises(ValueError, match="row index of exog in equation 'y0'"):
        SUR.multivariate_ls(multi.dependent, reverse(multi.exog))
    # An array dependent has no labels, so exog sets them
    SUR.multivariate_ls(multi.dependent.to_numpy(), multi.exog)
    # An array has no labels
    mod = SUR.multivariate_ls(multi.dependent, multi.exog.to_numpy()[::-1])
    assert_allclose(mod._exog[0].ndarray, multi.exog.to_numpy()[::-1])
    with pytest.raises(ValueError, match="row index of exog"):
        SUR.multivariate_ls(reverse(multi.dependent), multi.exog)


def test_multivariate_iv(multi):
    mod = IV3SLS.multivariate_iv(
        multi.dependent, multi.exog, multi.endog, multi.instruments
    )
    raw = IV3SLS.multivariate_iv(
        multi.dependent.to_numpy(),
        multi.exog.to_numpy(),
        multi.endog.to_numpy(),
        multi.instruments.to_numpy(),
    )
    assert_same_arrays(mod, raw)
    for name in ["exog", "endog", "instruments"]:
        args = {
            "exog": multi.exog,
            "endog": multi.endog,
            "instruments": multi.instruments,
            name: reverse(getattr(multi, name)),
        }
        with pytest.raises(ValueError, match=f"row index of {name} in equation 'y0'"):
            IV3SLS.multivariate_iv(multi.dependent, **args)
        # Arrays have no labels
        args[name] = getattr(multi, name).to_numpy()[::-1]
        IV3SLS.multivariate_iv(multi.dependent, **args)


def test_multivariate_iv_inputs_must_agree_without_labels_in_dependent(multi):
    dependent = multi.dependent.to_numpy()
    IV3SLS.multivariate_iv(dependent, multi.exog, multi.endog, multi.instruments)
    args = {"exog": multi.exog, "endog": multi.endog, "instruments": multi.instruments}
    match = r"row index of instruments in equation 'dependent\.0' does not match exog"
    with pytest.raises(ValueError, match=match):
        IV3SLS.multivariate_iv(
            dependent, **{**args, "instruments": reverse(multi.instruments)}
        )
    match = r"row index of endog in equation 'dependent\.0' does not match exog"
    with pytest.raises(ValueError, match=match):
        IV3SLS.multivariate_iv(dependent, **{**args, "endog": reverse(multi.endog)})


# Formulas use a single DataFrame


@pytest.mark.parametrize("cls", [SUR, IV3SLS, IVSystemGMM])
def test_from_formula(simple, cls):
    frame = simple.frame()
    assert not isinstance(frame.index, pd.RangeIndex)
    if cls is SUR:
        formula = {"eq1": "y0 ~ 1 + x0", "eq2": "y1 ~ 1 + x1"}
    else:
        formula = {
            "eq1": "y0 ~ 1 + x0 + [e0 ~ z0_a + z0_b]",
            "eq2": "y1 ~ 1 + x1 + [e1 ~ z1_a + z1_b]",
        }
    plain = cls.from_formula(formula, frame)
    weights = {"eq1": frame["w0"], "eq2": frame["w1"]}
    weighted = cls.from_formula(formula, frame, weights=weights)
    raw_weights = {key: value.to_numpy() for key, value in weights.items()}
    raw = cls.from_formula(formula, frame, weights=raw_weights)
    assert_same_arrays(weighted, raw)
    assert len(plain._y) == 2
    for corrupt in CORRUPTIONS.values():
        bad = {"eq1": frame["w0"], "eq2": corrupt(frame["w1"])}
        match = "row index of weights in equation 'eq2' does not match"
        with pytest.raises(ValueError, match=match):
            cls.from_formula(formula, frame, weights=bad)
    # The same equations in the string format
    if cls is SUR:
        text = "{y0 ~ 1 + x0} {y1 ~ 1 + x1}"
    else:
        text = "{y0 ~ 1 + x0 + [e0 ~ z0_a + z0_b]} {y1 ~ 1 + x1 + [e1 ~ z1_a + z1_b]}"
    string = cls.from_formula(
        text, frame, weights={"y0": frame["w0"], "y1": frame["w1"]}
    )
    assert len(string._y) == 2
    bad = {"y0": frame["w0"], "y1": reverse(frame["w1"])}
    with pytest.raises(ValueError, match="row index of weights in equation 'y1'"):
        cls.from_formula(text, frame, weights=bad)


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_from_formula_matches_equations(simple, cls, kind):
    frame = simple.frame()
    if kind == "sur":
        formula = {"eq1": "y0 ~ 1 + x0", "eq2": "y1 ~ 1 + x1"}
    else:
        formula = {
            "eq1": "y0 ~ 1 + x0 + [e0 ~ z0_a + z0_b]",
            "eq2": "y1 ~ 1 + x1 + [e1 ~ z1_a + z1_b]",
        }
    formula_res = cls.from_formula(formula, frame).fit()
    eqns = equations(simple, kind)
    del eqns["eq3"]
    eqn_res = cls(eqns).fit()
    assert_allclose(formula_res.params, eqn_res.params)


# Clusters


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_clusters_in_fit(simple, cls, kind, corruption):
    mod = cls(equations(simple, kind))
    res = mod.fit(cov_type="clustered", clusters=simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=simple.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    bad = CORRUPTIONS[corruption](simple.clusters)
    match = "row index of clusters does not match dependent in equation 'eq1'"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=bad)


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_clusters_numpy_is_paired_by_position(simple, cls, kind):
    mod = cls(equations(simple, kind))
    values = simple.clusters.to_numpy()[::-1]
    raw = mod.fit(cov_type="clustered", clusters=values)
    # A default index has no labels
    for default in (pd.Series(values), pd.DataFrame({"cl": values})):
        res = mod.fit(cov_type="clustered", clusters=default)
        assert_allclose(res.std_errors, raw.std_errors)


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_two_way_clusters(simple, cls, kind):
    rng = np.random.default_rng(11)
    second = pd.Series(rng.integers(0, 7, NOBS), index=simple.index, name="cl2")
    clusters = pd.concat([simple.clusters, second], axis=1)
    mod = cls(equations(simple, kind))
    res = mod.fit(cov_type="clustered", clusters=clusters)
    raw = mod.fit(cov_type="clustered", clusters=clusters.to_numpy())
    assert_allclose(res.cov, raw.cov)
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=reverse(clusters))
    unnamed = pd.DataFrame(clusters.to_numpy(), columns=["a", "b"])
    res = mod.fit(cov_type="clustered", clusters=unnamed)
    assert_allclose(res.cov, raw.cov)


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_clusters_use_the_first_labelled_dependent(simple, cls, kind):
    replace = {("eq1", "dependent"): simple.data["eq1"]["dependent"].to_numpy()}
    mod = cls(equations(simple, kind, replace=replace))
    mod.fit(cov_type="clustered", clusters=simple.clusters)
    match = "row index of clusters does not match dependent in equation 'eq2'"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=reverse(simple.clusters))


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_clusters_use_the_labels_of_the_first_input_with_labels(simple, cls, kind):
    # No dependent variable has labels, so clusters use the labels of exog in eq1
    replace = {
        (label, "dependent"): simple.data[label]["dependent"].to_numpy()
        for label in LABELS
    }
    mod = cls(equations(simple, kind, replace=replace))
    res = mod.fit(cov_type="clustered", clusters=simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=simple.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    match = "row index of clusters does not match exog in equation 'eq1'"
    for corrupt in CORRUPTIONS.values():
        with pytest.raises(ValueError, match=match):
            mod.fit(cov_type="clustered", clusters=corrupt(simple.clusters))
    # Without labels in any input there is nothing to compare
    mod = cls(equations(simple, kind, raw=True))
    mod.fit(cov_type="clustered", clusters=reverse(simple.clusters))
    # weights can be the only input with labels
    replace = {
        (label, name): simple.data[label][name].to_numpy()
        for label in LABELS
        for name in ["dependent", "exog", "endog", "instruments"]
        if (kind == "iv" or name in ("dependent", "exog"))
    }
    for label in ("eq1", "eq3"):
        replace[(label, "weights")] = simple.data[label]["weights"].to_numpy()
    weighted = cls(equations(simple, kind, weights=True, replace=replace))
    weighted.fit(cov_type="clustered", clusters=simple.clusters)
    match = "row index of clusters does not match weights in equation 'eq2'"
    with pytest.raises(ValueError, match=match):
        weighted.fit(cov_type="clustered", clusters=reverse(simple.clusters))


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_clusters_and_missing_values_without_labels_in_dependent(simple, cls, kind):
    exog = simple.data["eq2"]["exog"].copy()
    exog.iloc[7, 1] = np.nan
    replace = {
        (label, "dependent"): simple.data[label]["dependent"].to_numpy()
        for label in LABELS
    }
    replace[("eq2", "exog")] = exog
    with pytest.warns(MissingValueWarning):
        mod = cls(equations(simple, kind, replace=replace))
    mask = np.ones(NOBS, dtype=bool)
    mask[7] = False
    kept = simple.clusters[mask]
    res = mod.fit(cov_type="clustered", clusters=kept)
    raw = mod.fit(cov_type="clustered", clusters=kept.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    match = "row index of clusters does not match exog in equation 'eq1'"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=reverse(kept))


@pytest.mark.parametrize(("cls", "kind"), LS_MODELS)
def test_clusters_and_missing_values(simple, cls, kind):
    exog = simple.data["eq2"]["exog"].copy()
    exog.iloc[7, 1] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = cls(equations(simple, kind, replace={("eq2", "exog"): exog}))
    mask = np.ones(NOBS, dtype=bool)
    mask[7] = False
    kept = simple.clusters[mask]
    res = mod.fit(cov_type="clustered", clusters=kept)
    raw = mod.fit(cov_type="clustered", clusters=kept.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=reverse(kept))
    # All of the clusters is a length error, as before
    with pytest.raises(ValueError, match="clusters must be an ndarray"):
        mod.fit(cov_type="clustered", clusters=simple.clusters)
