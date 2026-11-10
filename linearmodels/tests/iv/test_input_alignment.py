"""
Inputs that carry labels (pandas, xarray) have to use the labels of dependent.

The IV and absorbing estimators pair the rows of their inputs by position. A
labelled input whose rows are in a different order from dependent would
silently be attached to the wrong observations. These tests check that this
is rejected, that inputs that carry no labels (NumPy arrays and pandas objects
with a default index) are paired by position, and that inputs that use the
labels of dependent are accepted. If dependent carries no labels, the first of
the other inputs that does sets the labels, and the inputs after it have to use
them.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels import OLS
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
from linearmodels.iv.model import _OLS
from linearmodels.shared.exceptions import MissingValueWarning

NOBS = 120
ESTIMATORS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]
ESTIMATOR_IDS = [e.__name__ for e in ESTIMATORS]
INDEX_STYLES = ["str", "datetime", "int", "multi"]
INPUTS = ["exog", "endog", "instruments", "weights"]
PARTS = ["dependent", "exog", "endog", "instruments"]
# Estimators that are fast to fit
FAST = [IV2SLS, IVLIML, IVGMM, IVJIVE]
FAST_IDS = [e.__name__ for e in FAST]
GMM = [IVGMM, IVGMMCUE]
GMM_IDS = [e.__name__ for e in GMM]
WITH_CLUSTERS = [*FAST, IVGMMCUE]
WITH_CLUSTERS_IDS = [e.__name__ for e in WITH_CLUSTERS]


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


class Data:
    """Labelled IV data. The attributes are pandas objects with one index."""

    def __init__(self, style="str", nobs=NOBS, seed=0):
        rng = np.random.default_rng(seed)
        self.index = idx = make_index(style, nobs)
        self.instruments = pd.DataFrame(
            rng.standard_normal((nobs, 2)), index=idx, columns=["z1", "z2"]
        )
        z = self.instruments
        self.endog = pd.DataFrame(
            {"x": z.z1 + z.z2 + rng.standard_normal(nobs)}, index=idx
        )
        self.exog = pd.DataFrame(
            {"const": 1.0, "w": rng.standard_normal(nobs)}, index=idx
        )
        self.dependent = pd.Series(
            1 + self.endog.x + self.exog.w + rng.standard_normal(nobs),
            index=idx,
            name="y",
        )
        self.weights = pd.Series(
            np.exp(rng.standard_normal(nobs)), index=idx, name="wt"
        )
        self.clusters = pd.Series(rng.integers(0, 12, nobs), index=idx, name="cl")

    def parts(self):
        return {name: getattr(self, name) for name in PARTS}

    def raw(self):
        """The arguments of an IV model as NumPy arrays"""
        return [value.to_numpy() for value in self.parts().values()]


@pytest.fixture(params=INDEX_STYLES)
def data(request):
    return Data(request.param)


@pytest.fixture
def simple():
    return Data("str")


def reverse(value):
    """The same labels in the opposite order"""
    return value.iloc[::-1]


def relabel(value):
    """The same values, with the labels attached to the wrong rows"""
    out = value.copy()
    out.index = value.index[np.roll(np.arange(len(value)), 1)]
    return out


CORRUPTIONS = {"reverse": reverse, "relabel": relabel}


def build(cls, data, **replace):
    """Model using data, where some inputs are replaced"""
    weights = replace.pop("weights", None)
    parts = {**data.parts(), **replace}
    return cls(*parts.values(), weights=weights)


def arrays(mod):
    """The arrays used to estimate a model"""
    return [
        mod.dependent.ndarray,
        mod.exog.ndarray,
        mod.endog.ndarray,
        mod.instruments.ndarray,
        mod.weights.ndarray,
    ]


def assert_same_arrays(left, right):
    for a, b in zip(arrays(left), arrays(right), strict=True):
        assert_allclose(a, b)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
@pytest.mark.parametrize("name", INPUTS)
def test_misaligned_input_rejected(data, cls, name, corruption):
    # The model is accepted when the input uses the labels of dependent
    build(cls, data, **{name: getattr(data, name)})
    bad = CORRUPTIONS[corruption](getattr(data, name))
    with pytest.raises(ValueError, match=f"row index of {name} does not match"):
        build(cls, data, **{name: bad})


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_message_names_input_and_remedy(simple, cls):
    with pytest.raises(ValueError, match="does not match") as exc:
        build(cls, simple, instruments=reverse(simple.instruments))
    message = str(exc.value)
    assert "instruments" in message
    assert "reindex" in message
    assert ".to_numpy()" in message
    assert "in the same order as dependent" in message


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_dependent_reordered_rejected(simple, cls):
    # It is the other inputs that do not match dependent
    for corrupt in CORRUPTIONS.values():
        with pytest.raises(ValueError, match="row index of exog does not match"):
            build(cls, simple, dependent=corrupt(simple.dependent))


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_everything_reordered_together_is_allowed(simple, cls):
    expected = build(cls, simple, weights=simple.weights)
    flipped = {name: reverse(value) for name, value in simple.parts().items()}
    reordered = cls(*flipped.values(), weights=reverse(simple.weights))
    # The same observations in the opposite order
    for a, b in zip(arrays(expected), arrays(reordered), strict=True):
        assert_allclose(a, b[::-1])
    if cls is not IVGMMCUE:
        assert_allclose(expected.fit().params, reordered.fit().params, rtol=1e-8)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_same_labels_match_numpy(data, cls):
    labelled = build(cls, data, weights=data.weights)
    raw = cls(*data.raw(), weights=data.weights.to_numpy())
    assert_same_arrays(labelled, raw)


@pytest.mark.parametrize("cls", FAST, ids=FAST_IDS)
def test_same_labels_match_numpy_results(simple, cls):
    labelled = build(cls, simple, weights=simple.weights).fit()
    raw = cls(*simple.raw(), weights=simple.weights.to_numpy()).fit()
    assert_allclose(labelled.params, raw.params)
    assert_allclose(labelled.std_errors, raw.std_errors)
    assert_allclose(labelled.resids, raw.resids)


def numpy_equivalent(cls, data, name, values):
    """Model where the labels of all inputs have been removed"""
    raw = {part: value.to_numpy() for part, value in data.parts().items()}
    if name == "weights":
        return cls(*raw.values(), weights=values)
    raw[name] = values
    return cls(*raw.values())


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("name", INPUTS)
def test_numpy_input_is_paired_by_position(simple, cls, name):
    # A reordered array has no labels, and so it is paired by position
    original = getattr(simple, name)
    values = original.to_numpy()[::-1]
    mod = build(cls, simple, **{name: values})
    assert_same_arrays(mod, numpy_equivalent(cls, simple, name, values))
    if name != "weights":
        # The values were paired with the opposite rows
        assert_allclose(
            getattr(mod, name).ndarray, np.asarray(original).reshape(NOBS, -1)[::-1]
        )


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("name", INPUTS)
def test_default_index_is_paired_by_position(simple, cls, name):
    # pandas objects that have a default index carry no labels. This is also
    # the case for a DataFrame that has named columns
    original = getattr(simple, name)
    values = original.to_numpy()[::-1]
    if isinstance(original, pd.DataFrame):
        framed = pd.DataFrame(values, columns=original.columns)
    else:
        framed = pd.Series(values, name=original.name)
    assert isinstance(framed.index, pd.RangeIndex)
    mod = build(cls, simple, **{name: framed})
    assert_same_arrays(mod, numpy_equivalent(cls, simple, name, values))


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_reversed_default_index_has_labels(simple, cls):
    # The labels of a reversed range are not 0, 1, ...
    exog = pd.DataFrame(simple.exog.to_numpy(), columns=simple.exog.columns)
    build(cls, simple, exog=exog)
    reversed_exog = exog.iloc[::-1]
    assert isinstance(reversed_exog.index, pd.RangeIndex)
    with pytest.raises(ValueError, match="row index of exog does not match"):
        build(cls, simple, exog=reversed_exog)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_unlabelled_dependent_is_paired_by_position(simple, cls):
    # There are no labels in dependent, so the other inputs are paired with it by
    # position. They use the same labels as each other
    y = simple.dependent.to_numpy()
    for corrupt in CORRUPTIONS.values():
        mod = cls(
            y,
            corrupt(simple.exog),
            corrupt(simple.endog),
            corrupt(simple.instruments),
            weights=corrupt(simple.weights),
        )
        assert_allclose(mod.exog.ndarray, corrupt(simple.exog).to_numpy())
        assert_allclose(mod.dependent.ndarray, y[:, None])
    # The same holds if dependent has a default index
    flipped = (reverse(simple.exog), reverse(simple.endog), reverse(simple.instruments))
    cls(pd.Series(y), *flipped)


UNLABELLED_DEPENDENT = {
    "numpy": lambda y: y.to_numpy(),
    "default-index": lambda y: pd.Series(y.to_numpy(), name=y.name),
}


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("form", list(UNLABELLED_DEPENDENT))
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
@pytest.mark.parametrize("name", INPUTS)
def test_inputs_must_agree_when_dependent_has_no_labels(
    simple, cls, form, name, corruption
):
    dependent = UNLABELLED_DEPENDENT[form](simple.dependent)
    build(cls, simple, dependent=dependent, weights=simple.weights)
    bad = CORRUPTIONS[corruption](getattr(simple, name))
    # exog sets the labels. If it is the input that changed, endog is the first
    # input that does not use them
    other = "endog" if name == "exog" else name
    replace = {"dependent": dependent, "weights": simple.weights, name: bad}
    with pytest.raises(ValueError, match=f"row index of {other} does not match exog"):
        build(cls, simple, **replace)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_message_names_the_input_that_sets_the_labels(simple, cls):
    y = simple.dependent.to_numpy()
    with pytest.raises(ValueError, match="does not match") as exc:
        cls(y, simple.exog, simple.endog, reverse(simple.instruments))
    message = str(exc.value)
    assert message.startswith("The row index of instruments does not match exog.")
    assert "in the same order as exog" in message
    assert "reindex instruments to the index of exog" in message
    assert ".to_numpy()" in message


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_inputs_without_labels_do_not_set_the_labels(simple, cls):
    y, exog = simple.dependent.to_numpy(), simple.exog.to_numpy()
    cls(y, exog, simple.endog, simple.instruments)
    # endog is the first input with labels
    match = "row index of instruments does not match endog"
    with pytest.raises(ValueError, match=match):
        cls(y, exog, simple.endog, reverse(simple.instruments))
    match = "row index of weights does not match endog"
    with pytest.raises(ValueError, match=match):
        cls(y, exog, simple.endog, simple.instruments, weights=reverse(simple.weights))
    # A default index has no labels, so exog does not set them
    default = pd.DataFrame(exog, columns=simple.exog.columns)
    with pytest.raises(ValueError, match=match):
        cls(
            y,
            default,
            simple.endog,
            simple.instruments,
            weights=reverse(simple.weights),
        )
    # Only one input has labels
    cls(y, exog, simple.endog.to_numpy(), reverse(simple.instruments))
    cls(y, exog, reverse(simple.endog), simple.instruments.to_numpy())
    # weights set the labels if no other input has them
    raw = [y, exog, simple.endog.to_numpy(), simple.instruments.to_numpy()]
    cls(*raw, weights=reverse(simple.weights))


@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_inputs_must_agree_for_every_index_style(data, corruption):
    y = data.dependent.to_numpy()
    IV2SLS(y, data.exog, data.endog, data.instruments)
    bad = CORRUPTIONS[corruption](data.instruments)
    with pytest.raises(
        ValueError, match="row index of instruments does not match exog"
    ):
        IV2SLS(y, data.exog, data.endog, bad)


def test_ols_inputs_must_agree_when_dependent_has_no_labels(simple):
    y = simple.dependent.to_numpy()
    OLS(y, simple.exog, weights=simple.weights)
    with pytest.raises(ValueError, match="row index of weights does not match exog"):
        OLS(y, simple.exog, weights=reverse(simple.weights))
    with pytest.raises(ValueError, match="row index of weights does not match exog"):
        IV2SLS(y, simple.exog, None, None, weights=relabel(simple.weights))
    mod = OLS(y, simple.exog)
    mod.fit(cov_type="clustered", clusters=simple.clusters)
    with pytest.raises(ValueError, match="row index of clusters does not match exog"):
        mod.fit(cov_type="clustered", clusters=reverse(simple.clusters))


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_ivdata_inputs_set_the_labels_when_dependent_has_none(simple, cls):
    y = simple.dependent.to_numpy()
    cls(y, IVData(simple.exog), simple.endog, simple.instruments)
    with pytest.raises(ValueError, match="row index of endog does not match exog"):
        cls(y, IVData(reverse(simple.exog)), simple.endog, simple.instruments)
    # IVData built from an array has no labels
    cls(y, IVData(simple.exog.to_numpy()[::-1]), simple.endog, simple.instruments)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_xarray_inputs_set_the_labels_when_dependent_has_none(simple, cls):
    y = simple.dependent.to_numpy()
    exog = to_xarray(simple.exog)
    cls(y, exog, simple.endog, simple.instruments)
    with pytest.raises(ValueError, match="row index of endog does not match exog"):
        cls(y, to_xarray(reverse(simple.exog)), simple.endog, simple.instruments)
    # A DataArray without coordinates has no labels
    no_coords = to_xarray(reverse(simple.exog), coords=False)
    cls(y, no_coords, simple.endog, simple.instruments)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_index_names_are_ignored(simple, cls):
    exog = simple.exog.rename_axis("another name")
    weights = simple.weights.rename_axis(None)
    mod = build(cls, simple, exog=exog, weights=weights)
    assert_allclose(mod.exog.ndarray, simple.exog.to_numpy())


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_duplicate_labels(cls):
    # Duplicated labels are equal if they are in the same order
    rng = np.random.default_rng(3)
    index = pd.Index(np.repeat(["a", "b", "c", "d"], NOBS // 4))
    exog = pd.DataFrame({"c": 1.0, "w": rng.standard_normal(NOBS)}, index=index)
    z = pd.DataFrame(rng.standard_normal((NOBS, 2)), index=index)
    x = pd.Series(z.sum(axis=1) + rng.standard_normal(NOBS), index=index, name="x")
    y = pd.Series(x + rng.standard_normal(NOBS), index=index, name="y")
    cls(y, exog, x, z)
    with pytest.raises(ValueError, match="row index of instruments"):
        cls(y, exog, x, z.iloc[::-1])


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_different_length_reports_the_length(simple, cls):
    with pytest.raises(ValueError, match="Array required to have"):
        build(cls, simple, exog=simple.exog.iloc[:-1])
    with pytest.raises(ValueError, match="Array required to have"):
        build(cls, simple, weights=reverse(simple.weights).iloc[:-1])


def test_ols(simple):
    # linearmodels.OLS is the same as _OLS
    assert OLS is _OLS
    mod = IV2SLS(simple.dependent, simple.exog, None, None)
    assert mod._method == "OLS"
    with pytest.raises(ValueError, match="row index of exog does not match"):
        IV2SLS(simple.dependent, reverse(simple.exog), None, None)
    with pytest.raises(ValueError, match="row index of weights does not match"):
        IV2SLS(
            simple.dependent, simple.exog, None, None, weights=reverse(simple.weights)
        )
    with pytest.raises(ValueError, match="row index of exog does not match"):
        OLS(simple.dependent, reverse(simple.exog))
    with pytest.raises(ValueError, match="row index of weights does not match"):
        OLS(simple.dependent, simple.exog, weights=relabel(simple.weights))
    res = OLS(simple.dependent, simple.exog).fit(
        cov_type="clustered", clusters=simple.clusters
    )
    raw = OLS(simple.dependent, simple.exog).fit(
        cov_type="clustered", clusters=simple.clusters.to_numpy()
    )
    assert_allclose(res.std_errors, raw.std_errors)
    with pytest.raises(ValueError, match="row index of clusters does not match"):
        OLS(simple.dependent, simple.exog).fit(
            cov_type="clustered", clusters=reverse(simple.clusters)
        )
    labelled = _OLS(simple.dependent, simple.exog, weights=simple.weights)
    raw = _OLS(
        simple.dependent.to_numpy(),
        simple.exog.to_numpy(),
        weights=simple.weights.to_numpy(),
    )
    assert_same_arrays(labelled, raw)


def test_inputs_that_are_not_present_are_not_compared(simple):
    IV2SLS(simple.dependent, None, simple.endog, simple.instruments)
    with pytest.raises(ValueError, match="row index of instruments"):
        IV2SLS(simple.dependent, None, simple.endog, reverse(simple.instruments))


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_ivdata_inputs(simple, cls):
    build(cls, simple, instruments=IVData(simple.instruments))
    with pytest.raises(ValueError, match="row index of instruments"):
        build(cls, simple, instruments=IVData(reverse(simple.instruments)))
    # IVData built from an array carries no labels
    build(cls, simple, instruments=IVData(simple.instruments.to_numpy()[::-1]))
    with pytest.raises(ValueError, match="row index of exog"):
        build(
            cls,
            simple,
            dependent=IVData(simple.dependent),
            exog=IVData(reverse(simple.exog)),
        )
    with pytest.raises(ValueError, match="row index of weights"):
        build(cls, simple, weights=IVData(reverse(simple.weights)))
    # IVData built from a pandas object with a default index carries no labels
    default = IVData(pd.DataFrame(simple.exog.to_numpy()[::-1], columns=["a", "b"]))
    build(cls, simple, exog=default)


def to_xarray(frame, coords=True):
    xr = pytest.importorskip("xarray")
    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    obs = {"obs": frame.index.to_numpy()} if coords else {}
    return xr.DataArray(
        frame.to_numpy(), dims=["obs", "var"], coords={**obs, "var": list(frame)}
    )


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_xarray_inputs(simple, cls):
    instruments = to_xarray(simple.instruments)
    build(cls, simple, instruments=instruments)
    # Reversing the rows of the DataArray reverses the coordinates
    reversed_rows = instruments.isel(obs=slice(None, None, -1))
    with pytest.raises(ValueError, match="row index of instruments"):
        build(cls, simple, instruments=reversed_rows)
    with pytest.raises(ValueError, match="row index of exog"):
        build(
            cls,
            simple,
            dependent=to_xarray(simple.dependent),
            exog=to_xarray(reverse(simple.exog)),
        )
    # A DataArray without coordinates has no labels
    no_coords = to_xarray(reverse(simple.instruments), coords=False)
    mod = build(cls, simple, instruments=no_coords)
    assert_allclose(mod.instruments.ndarray, simple.instruments.to_numpy()[::-1])
    # An xarray dependent sets the labels
    with pytest.raises(ValueError, match="row index of endog"):
        build(
            cls,
            simple,
            dependent=to_xarray(simple.dependent),
            endog=reverse(simple.endog),
        )
    build(cls, simple, dependent=to_xarray(simple.dependent), endog=simple.endog)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_missing_values_do_not_change_the_check(simple, cls):
    exog = simple.exog.copy()
    exog.iloc[3, 1] = np.nan
    weights = simple.weights.copy()
    weights.iloc[10] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = build(cls, simple, exog=exog, weights=weights)
    assert mod.notnull.sum() == NOBS - 2
    with pytest.raises(ValueError, match="row index of instruments"):
        build(
            cls,
            simple,
            exog=exog,
            weights=weights,
            instruments=reverse(simple.instruments),
        )


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_columns_are_not_compared(simple, cls):
    # The columns are variables. Only the rows are paired
    exog = simple.exog.copy()
    exog["group"] = pd.Categorical(np.arange(NOBS) % 3)
    mod = build(cls, simple, exog=exog)
    assert mod.exog.shape[1] == 4
    with pytest.raises(ValueError, match="row index of exog"):
        build(cls, simple, exog=reverse(exog))


# Clusters


@pytest.mark.parametrize("cls", WITH_CLUSTERS, ids=WITH_CLUSTERS_IDS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_clusters_in_fit(simple, cls, corruption):
    mod = build(cls, simple)
    res = mod.fit(cov_type="clustered", clusters=simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=simple.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    bad = CORRUPTIONS[corruption](simple.clusters)
    with pytest.raises(ValueError, match="row index of clusters does not match"):
        mod.fit(cov_type="clustered", clusters=bad)


@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_clusters_index_styles(data, corruption):
    mod = IV2SLS(*data.parts().values())
    res = mod.fit(cov_type="clustered", clusters=data.clusters)
    raw = mod.fit(cov_type="clustered", clusters=data.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    bad = CORRUPTIONS[corruption](data.clusters)
    with pytest.raises(ValueError, match="row index of clusters does not match"):
        mod.fit(cov_type="clustered", clusters=bad)


@pytest.mark.parametrize("cls", FAST, ids=FAST_IDS)
def test_clusters_use_the_labels_of_the_first_input_with_labels(simple, cls):
    y = simple.dependent.to_numpy()
    mod = cls(y, simple.exog, simple.endog, simple.instruments)
    res = mod.fit(cov_type="clustered", clusters=simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=simple.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    for corrupt in CORRUPTIONS.values():
        match = "row index of clusters does not match exog"
        with pytest.raises(ValueError, match=match):
            mod.fit(cov_type="clustered", clusters=corrupt(simple.clusters))
    # Without labels in any input there is nothing to compare
    mod = cls(*simple.raw())
    mod.fit(cov_type="clustered", clusters=reverse(simple.clusters))
    # weights can be the only input with labels
    weighted = cls(*simple.raw(), weights=simple.weights)
    weighted.fit(cov_type="clustered", clusters=simple.clusters)
    match = "row index of clusters does not match weights"
    with pytest.raises(ValueError, match=match):
        weighted.fit(cov_type="clustered", clusters=reverse(simple.clusters))


@pytest.mark.parametrize("cls", FAST, ids=FAST_IDS)
def test_clusters_and_missing_values_without_labels_in_dependent(simple, cls):
    exog = simple.exog.copy()
    exog.iloc[7, 1] = np.nan
    y = simple.dependent.to_numpy()
    with pytest.warns(MissingValueWarning):
        mod = cls(y, exog, simple.endog, simple.instruments)
    kept = simple.clusters[mod.notnull]
    assert len(kept) == NOBS - 1
    res = mod.fit(cov_type="clustered", clusters=kept)
    raw = mod.fit(cov_type="clustered", clusters=kept.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    with pytest.raises(ValueError, match="row index of clusters does not match exog"):
        mod.fit(cov_type="clustered", clusters=reverse(kept))


@pytest.mark.parametrize("cls", GMM, ids=GMM_IDS)
def test_weight_clusters_use_the_labels_of_the_first_input_with_labels(simple, cls):
    y = simple.dependent.to_numpy()
    args = (y, simple.exog, simple.endog, simple.instruments)
    clustered = {"weight_type": "clustered"}
    cls(*args, clusters=simple.clusters, **clustered)
    for corrupt in CORRUPTIONS.values():
        match = "row index of clusters does not match exog"
        with pytest.raises(ValueError, match=match):
            cls(*args, clusters=corrupt(simple.clusters), **clustered)
    # Arrays have no labels
    cls(*args, clusters=reverse(simple.clusters).to_numpy(), **clustered)


@pytest.mark.parametrize("cls", WITH_CLUSTERS, ids=WITH_CLUSTERS_IDS)
def test_clusters_numpy_is_paired_by_position(simple, cls):
    mod = build(cls, simple)
    values = simple.clusters.to_numpy()[::-1]
    raw = cls(*simple.raw()).fit(cov_type="clustered", clusters=values)
    res = mod.fit(cov_type="clustered", clusters=values)
    assert_allclose(res.std_errors, raw.std_errors)
    # A default index carries no labels
    frame = pd.DataFrame({"cl": values})
    res = mod.fit(cov_type="clustered", clusters=frame)
    assert_allclose(res.std_errors, raw.std_errors)
    res = mod.fit(cov_type="clustered", clusters=pd.Series(values))
    assert_allclose(res.std_errors, raw.std_errors)


@pytest.mark.parametrize("cls", [IV2SLS, IVLIML])
def test_two_way_clusters(simple, cls):
    rng = np.random.default_rng(11)
    second = pd.Series(rng.integers(0, 7, NOBS), index=simple.index, name="cl2")
    clusters = pd.concat([simple.clusters, second], axis=1)
    mod = build(cls, simple)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    raw = mod.fit(cov_type="clustered", clusters=clusters.to_numpy())
    assert_allclose(res.cov, raw.cov)
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=reverse(clusters))
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=relabel(clusters))
    # The columns are names, and do not need to match anything
    unnamed = pd.DataFrame(clusters.to_numpy(), columns=["a", "b"])
    res = mod.fit(cov_type="clustered", clusters=unnamed)
    assert_allclose(res.cov, raw.cov)


@pytest.mark.parametrize("cls", GMM, ids=GMM_IDS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_clusters_in_weight_config(simple, cls, corruption):
    clustered = {"weight_type": "clustered", "clusters": simple.clusters}
    labelled = cls(*simple.parts().values(), **clustered)
    raw_clusters = {**clustered, "clusters": simple.clusters.to_numpy()}
    raw = cls(*simple.raw(), **raw_clusters)
    assert_allclose(labelled.fit().params, raw.fit().params)
    bad = CORRUPTIONS[corruption](simple.clusters)
    with pytest.raises(ValueError, match="row index of clusters does not match"):
        cls(*simple.parts().values(), **{**clustered, "clusters": bad})
    # Arrays have no labels and are paired by position
    cls(*simple.parts().values(), **{**clustered, "clusters": bad.to_numpy()})
    # As are pandas objects with a default index
    cls(
        *simple.parts().values(),
        **{**clustered, "clusters": pd.Series(bad.to_numpy())},
    )


@pytest.mark.parametrize("cls", GMM, ids=GMM_IDS)
def test_clusters_in_weight_config_and_fit(simple, cls):
    clustered = {"weight_type": "clustered", "clusters": simple.clusters}
    mod = cls(*simple.parts().values(), **clustered)
    mod.fit(cov_type="clustered", clusters=simple.clusters)
    with pytest.raises(ValueError, match="row index of clusters does not match"):
        mod.fit(cov_type="clustered", clusters=reverse(simple.clusters))


@pytest.mark.parametrize("cls", GMM, ids=GMM_IDS)
def test_other_weight_config_is_unchanged(simple, cls):
    mod = build(cls, simple)
    assert mod._weight_type == "robust"
    mod = cls(*simple.parts().values(), weight_type="kernel", kernel="parzen")
    assert mod._weight_type == "kernel"
    assert mod._weight_config["kernel"] == "parzen"
    mod = cls(*simple.parts().values(), weight_type="unadjusted", debiased=True)
    assert mod._weight_config["debiased"]
    assert mod._weight_kwargs["debiased"]
    assert IVGMM(*simple.parts().values())._method == "IV-GMM"


@pytest.mark.parametrize("cls", WITH_CLUSTERS, ids=WITH_CLUSTERS_IDS)
def test_clusters_and_missing_values(simple, cls):
    # Missing values are dropped when the model is created, and so clusters
    # have to be provided for the observations that are used
    exog = simple.exog.copy()
    exog.iloc[7, 1] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = build(cls, simple, exog=exog)
    kept = simple.clusters[mod.notnull]
    assert len(kept) == NOBS - 1
    res = mod.fit(cov_type="clustered", clusters=kept)
    raw = mod.fit(cov_type="clustered", clusters=kept.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=reverse(kept))
    # Using all of the clusters is a length error, as before
    with pytest.raises(ValueError, match="clusters has the wrong nobs"):
        mod.fit(cov_type="clustered", clusters=simple.clusters)


@pytest.mark.parametrize("cls", GMM, ids=GMM_IDS)
def test_weight_clusters_and_missing_values(simple, cls):
    exog = simple.exog.copy()
    exog.iloc[7, 1] = np.nan
    mask = np.ones(NOBS, dtype=bool)
    mask[7] = False
    kept = simple.clusters[mask]
    parts = {**simple.parts(), "exog": exog}
    with pytest.warns(MissingValueWarning):
        mod = cls(*parts.values(), weight_type="clustered", clusters=kept)
    assert mod.fit().nobs == NOBS - 1
    with (
        pytest.warns(MissingValueWarning),
        pytest.raises(ValueError, match="row index of clusters"),
    ):
        cls(*parts.values(), weight_type="clustered", clusters=reverse(kept))


# Formulas use a single DataFrame


FORMULA = "y ~ 1 + w + [x ~ z1 + z2]"


@pytest.fixture
def frame(simple):
    return pd.concat(
        [
            simple.dependent,
            simple.exog[["w"]],
            simple.endog,
            simple.instruments,
            simple.weights,
        ],
        axis=1,
    )


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_from_formula(frame, cls):
    assert not isinstance(frame.index, pd.RangeIndex)
    plain = cls.from_formula(FORMULA, frame)
    assert plain.exog.shape == (NOBS, 2)
    weighted = cls.from_formula(FORMULA, frame, weights=frame["wt"])
    raw = cls.from_formula(FORMULA, frame, weights=frame["wt"].to_numpy())
    assert_same_arrays(weighted, raw)
    for corrupt in CORRUPTIONS.values():
        with pytest.raises(ValueError, match="row index of weights does not match"):
            cls.from_formula(FORMULA, frame, weights=corrupt(frame["wt"]))
    # The data has a default index, so there are no labels
    default = frame.reset_index(drop=True)
    cls.from_formula(FORMULA, default, weights=default["wt"])
    cls.from_formula(FORMULA, default, weights=default["wt"].to_numpy()[::-1])


@pytest.mark.parametrize("cls", FAST, ids=FAST_IDS)
def test_from_formula_matches_arrays(frame, cls):
    formula = cls.from_formula(FORMULA, frame, weights=frame["wt"]).fit()
    exog = np.c_[np.ones(NOBS), frame["w"].to_numpy()]
    arrays_ = cls(
        frame["y"].to_numpy(),
        exog,
        frame[["x"]].to_numpy(),
        frame[["z1", "z2"]].to_numpy(),
        weights=frame["wt"].to_numpy(),
    ).fit()
    assert_allclose(formula.params, arrays_.params)


# AbsorbingLS


class Absorbing:
    def __init__(self, style="str", nobs=NOBS, seed=5):
        rng = np.random.default_rng(seed)
        self.index = idx = make_index(style, nobs)
        self.exog = pd.DataFrame(
            {"const": 1.0, "w": rng.standard_normal(nobs)}, index=idx
        )
        self.cats = pd.DataFrame(
            {
                "g": pd.Categorical(rng.integers(0, 6, nobs)),
                "h": pd.Categorical(rng.integers(0, 4, nobs)),
            },
            index=idx,
        )
        self.cont = pd.DataFrame(
            rng.standard_normal((nobs, 2)), index=idx, columns=["c0", "c1"]
        )
        self.dependent = pd.Series(
            self.exog.w + self.cats["g"].cat.codes + rng.standard_normal(nobs),
            index=idx,
            name="y",
        )
        self.weights = pd.Series(
            np.exp(rng.standard_normal(nobs)), index=idx, name="wt"
        )
        self.clusters = pd.Series(rng.integers(0, 12, nobs), index=idx, name="cl")

    def model(self, **kwargs):
        return AbsorbingLS(self.dependent, self.exog, absorb=self.cats, **kwargs)


@pytest.fixture(params=INDEX_STYLES)
def absorbing(request):
    return Absorbing(request.param)


@pytest.fixture
def abs_simple():
    return Absorbing("str")


def test_absorbing_misaligned_rejected(absorbing):
    y, x, a, w = (
        absorbing.dependent,
        absorbing.exog,
        absorbing.cats,
        absorbing.weights,
    )
    AbsorbingLS(y, x, absorb=a, weights=w)
    for corrupt in CORRUPTIONS.values():
        with pytest.raises(ValueError, match="row index of absorb does not match"):
            AbsorbingLS(y, x, absorb=corrupt(a))
        with pytest.raises(ValueError, match="row index of exog does not match"):
            AbsorbingLS(y, corrupt(x), absorb=a)
        with pytest.raises(ValueError, match="row index of weights does not match"):
            AbsorbingLS(y, x, absorb=a, weights=corrupt(w))
        with pytest.raises(ValueError, match="row index of exog does not match"):
            AbsorbingLS(corrupt(y), x, absorb=a)


def test_absorbing_message(abs_simple):
    with pytest.raises(ValueError, match="does not match") as exc:
        AbsorbingLS(
            abs_simple.dependent, abs_simple.exog, absorb=reverse(abs_simple.cats)
        )
    message = str(exc.value)
    assert "absorb" in message
    assert "reindex" in message
    assert ".to_numpy()" in message


def test_absorbing_continuous_absorb(abs_simple):
    frame = pd.concat([abs_simple.cats, abs_simple.cont], axis=1)
    AbsorbingLS(abs_simple.dependent, abs_simple.exog, absorb=frame)
    with pytest.raises(ValueError, match="row index of absorb"):
        AbsorbingLS(abs_simple.dependent, abs_simple.exog, absorb=reverse(frame))


def test_absorbing_interaction_in_absorb(abs_simple):
    y, x = abs_simple.dependent, abs_simple.exog
    cat, cont = abs_simple.cats, abs_simple.cont
    AbsorbingLS(y, x, absorb=Interaction(cat=cat, cont=cont))
    AbsorbingLS(y, x, absorb=Interaction(cat=cat))
    AbsorbingLS(y, x, absorb=Interaction(cont=cont))
    with pytest.raises(ValueError, match="row index of absorb"):
        AbsorbingLS(y, x, absorb=Interaction(cat=reverse(cat), cont=reverse(cont)))
    with pytest.raises(ValueError, match="row index of absorb"):
        AbsorbingLS(y, x, absorb=Interaction(cat=reverse(cat)))
    with pytest.raises(ValueError, match="row index of absorb"):
        AbsorbingLS(y, x, absorb=Interaction(cont=reverse(cont)))
    # Arrays have no labels
    AbsorbingLS(y, x, absorb=Interaction(cat=cat.to_numpy()[::-1]))


def test_absorbing_interactions(abs_simple):
    y, x = abs_simple.dependent, abs_simple.exog
    cat, cont = abs_simple.cats, abs_simple.cont
    frame = pd.concat([cat[["g"]], cont[["c0"]]], axis=1)
    iact = Interaction(cat=cat[["h"]], cont=cont[["c1"]])
    flipped = Interaction(cat=reverse(cat[["h"]]), cont=reverse(cont[["c1"]]))
    AbsorbingLS(y, x, interactions=frame)
    AbsorbingLS(y, x, interactions=iact)
    AbsorbingLS(y, x, interactions=[frame, iact])
    with pytest.raises(ValueError, match="row index of interactions does not match"):
        AbsorbingLS(y, x, interactions=reverse(frame))
    with pytest.raises(ValueError, match="row index of interactions does not match"):
        AbsorbingLS(y, x, interactions=flipped)
    with pytest.raises(ValueError, match=r"row index of interactions\[0\]"):
        AbsorbingLS(y, x, interactions=[reverse(frame), iact])
    with pytest.raises(ValueError, match=r"row index of interactions\[1\]"):
        AbsorbingLS(y, x, interactions=[frame, relabel(frame)])
    with pytest.raises(ValueError, match=r"row index of interactions\[1\]"):
        AbsorbingLS(y, x, interactions=[frame, flipped])
    # Arrays have no labels
    arrays_ = Interaction(cat=cat[["h"]].to_numpy()[::-1], cont=cont[["c1"]])
    AbsorbingLS(y, x, interactions=[frame, arrays_])


def test_interaction_cat_and_cont_must_match(abs_simple):
    cat, cont = abs_simple.cats, abs_simple.cont
    Interaction(cat=cat, cont=cont)
    with pytest.raises(ValueError, match="row index of cont does not match cat"):
        Interaction(cat=cat, cont=reverse(cont))
    with pytest.raises(ValueError, match="row index of cont does not match cat"):
        Interaction(cat=cat, cont=relabel(cont))
    # There are no labels to compare
    Interaction(cat=cat, cont=cont.to_numpy()[::-1])
    Interaction(cat=cat.to_numpy()[::-1], cont=cont)
    default = pd.DataFrame(cont.to_numpy()[::-1], columns=cont.columns)
    Interaction(cat=cat, cont=default)


def test_absorbing_numpy_is_paired_by_position(abs_simple):
    y, x = abs_simple.dependent, abs_simple.exog
    cat, w = abs_simple.cats, abs_simple.weights
    unlabelled_cat = cat.reset_index(drop=True)
    # exog as an array in the opposite order
    flipped = x.to_numpy()[::-1]
    res = AbsorbingLS(y, flipped, absorb=cat).fit()
    raw = AbsorbingLS(y.to_numpy(), flipped, absorb=unlabelled_cat).fit()
    assert_allclose(res.params, raw.params)
    # weights as an array in the opposite order
    weights = w.to_numpy()[::-1]
    res = AbsorbingLS(y, x, absorb=cat, weights=weights).fit()
    raw = AbsorbingLS(
        y.to_numpy(), x.to_numpy(), absorb=unlabelled_cat, weights=weights
    ).fit()
    assert_allclose(res.params, raw.params)
    # pandas objects with a default index
    res = AbsorbingLS(
        y,
        pd.DataFrame(flipped, columns=x.columns),
        absorb=cat,
        weights=pd.Series(weights),
    ).fit()
    raw = AbsorbingLS(y.to_numpy(), flipped, absorb=unlabelled_cat, weights=weights)
    assert_allclose(res.params, raw.fit().params)


def test_absorbing_same_labels_match_numpy(absorbing):
    y, x, a, w = (
        absorbing.dependent,
        absorbing.exog,
        absorbing.cats,
        absorbing.weights,
    )
    res = AbsorbingLS(y, x, absorb=a, weights=w).fit()
    raw = AbsorbingLS(
        y.to_numpy(),
        x.to_numpy(),
        absorb=a.reset_index(drop=True),
        weights=w.to_numpy(),
    ).fit()
    assert_allclose(res.params, raw.params)
    assert_allclose(res.std_errors, raw.std_errors)


def test_absorbing_unlabelled_dependent_is_not_checked(abs_simple):
    y = abs_simple.dependent.to_numpy()
    AbsorbingLS(
        y,
        reverse(abs_simple.exog),
        absorb=reverse(abs_simple.cats),
        weights=reverse(abs_simple.weights),
    )


def test_absorbing_inputs_must_agree_when_dependent_has_no_labels(abs_simple):
    y = abs_simple.dependent.to_numpy()
    x, cats, w = abs_simple.exog, abs_simple.cats, abs_simple.weights
    AbsorbingLS(y, x, absorb=cats, weights=w)
    for corrupt in CORRUPTIONS.values():
        # exog sets the labels
        with pytest.raises(ValueError, match="row index of absorb does not match exog"):
            AbsorbingLS(y, x, absorb=corrupt(cats))
        with pytest.raises(
            ValueError, match="row index of weights does not match exog"
        ):
            AbsorbingLS(y, x, absorb=cats, weights=corrupt(w))
        with pytest.raises(ValueError, match="row index of absorb does not match exog"):
            AbsorbingLS(y, corrupt(x), absorb=cats)
        # exog is an array, so absorb sets the labels
        match = "row index of weights does not match absorb"
        with pytest.raises(ValueError, match=match):
            AbsorbingLS(y, x.to_numpy(), absorb=cats, weights=corrupt(w))
    # Only one input has labels
    unlabelled = reverse(cats).reset_index(drop=True)
    AbsorbingLS(y, x.to_numpy(), absorb=unlabelled, weights=reverse(w))
    AbsorbingLS(y, reverse(x), absorb=cats.reset_index(drop=True))


def test_absorbing_interactions_must_agree_when_dependent_has_no_labels(abs_simple):
    y = abs_simple.dependent.to_numpy()
    x = abs_simple.exog.to_numpy()
    cat, cont = abs_simple.cats, abs_simple.cont
    frame = pd.concat([cat[["g"]], cont[["c0"]]], axis=1)
    flipped = Interaction(cat=reverse(cat[["h"]]), cont=reverse(cont[["c1"]]))
    iact = Interaction(cat=cat[["h"]], cont=cont[["c1"]])
    AbsorbingLS(y, x, interactions=[frame, iact])
    match = r"row index of interactions\[1\] does not match interactions\[0\]"
    with pytest.raises(ValueError, match=match):
        AbsorbingLS(y, x, interactions=[frame, flipped])
    match = "row index of weights does not match interactions"
    with pytest.raises(ValueError, match=match):
        AbsorbingLS(y, x, interactions=iact, weights=reverse(abs_simple.weights))
    # An Interaction with arrays has no labels
    arrays_ = Interaction(cat=cat[["h"]].to_numpy()[::-1], cont=cont[["c1"]])
    AbsorbingLS(y, x, interactions=[frame, arrays_])


def test_absorbing_clusters_use_the_labels_of_the_first_input_with_labels(abs_simple):
    y = abs_simple.dependent.to_numpy()
    mod = AbsorbingLS(y, abs_simple.exog, absorb=abs_simple.cats)
    res = mod.fit(cov_type="clustered", clusters=abs_simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=abs_simple.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    for corrupt in CORRUPTIONS.values():
        match = "row index of clusters does not match exog"
        with pytest.raises(ValueError, match=match):
            mod.fit(cov_type="clustered", clusters=corrupt(abs_simple.clusters))
    # absorb sets the labels if exog has none
    mod = AbsorbingLS(y, abs_simple.exog.to_numpy(), absorb=abs_simple.cats)
    match = "row index of clusters does not match absorb"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=reverse(abs_simple.clusters))
    # The clusters are checked before the effects are absorbed
    assert mod._absorbed_dependent is None
    # Without labels in any input there is nothing to compare
    raw_model = AbsorbingLS(
        y, abs_simple.exog.to_numpy(), absorb=abs_simple.cats.reset_index(drop=True)
    )
    raw_model.fit(cov_type="clustered", clusters=reverse(abs_simple.clusters))


def test_absorbing_missing_values_without_labels_in_dependent(abs_simple):
    exog = abs_simple.exog.copy()
    exog.iloc[4, 1] = np.nan
    y = abs_simple.dependent.to_numpy()
    with pytest.warns(MissingValueWarning):
        mod = AbsorbingLS(y, exog, absorb=abs_simple.cats)
    kept = abs_simple.clusters[~np.asarray(exog.isnull().any(axis=1))]
    mod.fit(cov_type="clustered", clusters=kept)
    with pytest.raises(ValueError, match="row index of clusters does not match exog"):
        mod.fit(cov_type="clustered", clusters=reverse(kept))


def test_absorbing_clusters(abs_simple):
    mod = abs_simple.model()
    res = mod.fit(cov_type="clustered", clusters=abs_simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=abs_simple.clusters.to_numpy())
    assert_allclose(res.std_errors, raw.std_errors)
    for corrupt in CORRUPTIONS.values():
        with pytest.raises(ValueError, match="row index of clusters does not match"):
            mod.fit(cov_type="clustered", clusters=corrupt(abs_simple.clusters))
    values = abs_simple.clusters.to_numpy()[::-1]
    res = mod.fit(cov_type="clustered", clusters=values)
    raw = mod.fit(cov_type="clustered", clusters=pd.Series(values))
    assert_allclose(res.std_errors, raw.std_errors)
    # The clusters are checked before the effects are absorbed
    fresh = abs_simple.model()
    with pytest.raises(ValueError, match="row index of clusters"):
        fresh.fit(cov_type="clustered", clusters=reverse(abs_simple.clusters))
    assert fresh._absorbed_dependent is None


def test_absorbing_missing_values(abs_simple):
    exog = abs_simple.exog.copy()
    exog.iloc[4, 1] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = AbsorbingLS(abs_simple.dependent, exog, absorb=abs_simple.cats)
    kept = abs_simple.clusters[~np.asarray(exog.isnull().any(axis=1))]
    assert mod.dependent.shape[0] == NOBS - 1
    # clusters are for the observations that are used
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=reverse(kept))
    with pytest.raises(ValueError, match="row index of absorb"):
        AbsorbingLS(abs_simple.dependent, exog, absorb=reverse(abs_simple.cats))


def test_absorbing_xarray(abs_simple):
    exog = to_xarray(abs_simple.exog)
    AbsorbingLS(abs_simple.dependent, exog, absorb=abs_simple.cats)
    with pytest.raises(ValueError, match="row index of exog"):
        AbsorbingLS(
            abs_simple.dependent,
            exog.isel(obs=slice(None, None, -1)),
            absorb=abs_simple.cats,
        )
