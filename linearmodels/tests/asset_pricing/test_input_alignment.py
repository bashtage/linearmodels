"""
Factors that carry labels (pandas, xarray) have to use the labels of portfolios.

The factor models pair the rows of portfolios and factors by position: the
return on the test portfolios in a period is paired with the value of the
factors in the same row. Factors whose rows are in a different order from
portfolios would silently be attached to the wrong periods. These tests check
that this is rejected, that inputs that carry no labels (NumPy arrays and
pandas objects with a default index) are paired by position, and that inputs
that use the labels of portfolios are accepted.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv.data import IVData
from linearmodels.shared.exceptions import MissingValueWarning

NOBS = 120
NFACTOR = 2
NPORTFOLIO = 6
MODELS = [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
MODEL_IDS = [m.__name__ for m in MODELS]
FITTED = [TradedFactorModel, LinearFactorModel]
FITTED_IDS = [m.__name__ for m in FITTED]
INDEX_STYLES = ["datetime", "str", "int"]


def make_index(style, nobs=NOBS):
    rng = np.random.default_rng(77)
    if style == "str":
        return pd.Index([f"t{i}" for i in range(nobs)], name="period")
    if style == "datetime":
        return pd.date_range("2000-01-31", periods=nobs, freq="D", name="date")
    return pd.Index(rng.permutation(10 * nobs)[:nobs] + 5, name="id")


class Data:
    """Labelled returns. portfolios and factors share an index"""

    def __init__(self, style="datetime", nobs=NOBS, seed=2):
        rng = np.random.default_rng(seed)
        self.index = idx = make_index(style, nobs)
        self.factors = pd.DataFrame(
            0.5 + rng.standard_normal((nobs, NFACTOR)),
            index=idx,
            columns=[f"f{i}" for i in range(NFACTOR)],
        )
        loadings = rng.chisquare(2, (NFACTOR, NPORTFOLIO)) / 2
        noise = rng.standard_normal((nobs, NPORTFOLIO))
        self.portfolios = pd.DataFrame(
            self.factors.to_numpy() @ loadings + noise,
            index=idx,
            columns=[f"p{i}" for i in range(NPORTFOLIO)],
        )

    def frame(self):
        return pd.concat([self.portfolios, self.factors], axis=1)


@pytest.fixture(params=INDEX_STYLES)
def data(request):
    return Data(request.param)


@pytest.fixture
def simple():
    return Data("datetime")


def reverse(value):
    """The same labels in the opposite order"""
    return value.iloc[::-1]


def relabel(value):
    """The same values, with the labels attached to the wrong rows"""
    out = value.copy()
    out.index = value.index[np.roll(np.arange(len(value)), 1)]
    return out


CORRUPTIONS = {"reverse": reverse, "relabel": relabel}


def arrays(mod):
    return [mod.portfolios.ndarray, mod.factors.ndarray]


def assert_same_arrays(left, right):
    for a, b in zip(arrays(left), arrays(right), strict=True):
        assert_allclose(a, b)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_misaligned_factors_rejected(data, cls, corruption):
    cls(data.portfolios, data.factors)
    bad = CORRUPTIONS[corruption](data.factors)
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(data.portfolios, bad)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_message_names_input_and_remedy(simple, cls):
    with pytest.raises(ValueError, match="does not match") as exc:
        cls(simple.portfolios, reverse(simple.factors))
    message = str(exc.value)
    assert "factors" in message
    assert "does not match portfolios" in message
    assert "in the same order as portfolios" in message
    assert "reindex" in message
    assert ".to_numpy()" in message


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_misaligned_portfolios_rejected(simple, cls, corruption):
    # portfolios sets the labels, so it is factors that do not match
    bad = CORRUPTIONS[corruption](simple.portfolios)
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(bad, simple.factors)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_series_factor(simple, cls):
    # A single factor is a Series. The portfolios are used with their labels
    factor = simple.factors["f0"]
    portfolios = simple.portfolios.iloc[:, :3]
    cls(portfolios, factor)
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(portfolios, reverse(factor))
    mod = cls(portfolios, factor.to_numpy()[::-1])
    assert_allclose(mod.factors.ndarray, factor.to_numpy()[::-1, None])


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_same_labels_match_numpy(data, cls):
    labelled = cls(data.portfolios, data.factors)
    raw = cls(data.portfolios.to_numpy(), data.factors.to_numpy())
    assert_same_arrays(labelled, raw)


@pytest.mark.parametrize("cls", FITTED, ids=FITTED_IDS)
def test_same_labels_match_numpy_results(simple, cls):
    labelled = cls(simple.portfolios, simple.factors).fit()
    raw = cls(simple.portfolios.to_numpy(), simple.factors.to_numpy()).fit()
    assert_allclose(labelled.risk_premia, raw.risk_premia)
    assert_allclose(labelled.risk_premia_se, raw.risk_premia_se)
    assert_allclose(labelled.alphas, raw.alphas)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_numpy_factors_are_paired_by_position(simple, cls):
    values = simple.factors.to_numpy()[::-1]
    mod = cls(simple.portfolios, values)
    raw = cls(simple.portfolios.to_numpy(), values)
    assert_same_arrays(mod, raw)
    assert_allclose(mod.factors.ndarray, simple.factors.to_numpy()[::-1])


@pytest.mark.parametrize("cls", FITTED, ids=FITTED_IDS)
def test_numpy_factors_are_paired_by_position_results(simple, cls):
    values = simple.factors.to_numpy()[::-1]
    res = cls(simple.portfolios, values).fit()
    raw = cls(simple.portfolios.to_numpy(), values).fit()
    assert_allclose(res.risk_premia, raw.risk_premia)
    assert_allclose(res.alphas, raw.alphas)
    labelled = cls(simple.portfolios, simple.factors).fit()
    assert not np.allclose(res.alphas, labelled.alphas)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_numpy_portfolios_are_paired_by_position(simple, cls):
    # Without labels in portfolios there is nothing to compare to
    values = simple.portfolios.to_numpy()
    for corrupt in CORRUPTIONS.values():
        mod = cls(values, corrupt(simple.factors))
        assert_allclose(mod.factors.ndarray, corrupt(simple.factors).to_numpy())


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_default_index_is_paired_by_position(simple, cls):
    # pandas objects with a default index have no labels. This includes a
    # DataFrame that has named columns
    values = simple.factors.to_numpy()[::-1]
    factors = pd.DataFrame(values, columns=simple.factors.columns)
    assert isinstance(factors.index, pd.RangeIndex)
    mod = cls(simple.portfolios, factors)
    assert_allclose(mod.factors.ndarray, values)
    # Without labels in portfolios there is nothing to compare to
    portfolios = pd.DataFrame(simple.portfolios.to_numpy(), columns=list("abcdef"))
    mod = cls(portfolios, reverse(simple.factors))
    assert_allclose(mod.factors.ndarray, reverse(simple.factors).to_numpy())
    # The labels of a reversed range are not 0, 1, ...
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(simple.portfolios, factors.iloc[::-1])


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_everything_reordered_together_is_allowed(simple, cls):
    expected = cls(simple.portfolios, simple.factors)
    reordered = cls(reverse(simple.portfolios), reverse(simple.factors))
    for a, b in zip(arrays(expected), arrays(reordered), strict=True):
        assert_allclose(a, b[::-1])


@pytest.mark.parametrize("cls", FITTED, ids=FITTED_IDS)
def test_everything_reordered_together_results(simple, cls):
    expected = cls(simple.portfolios, simple.factors).fit()
    reordered = cls(reverse(simple.portfolios), reverse(simple.factors)).fit()
    assert_allclose(expected.risk_premia, reordered.risk_premia)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_index_names_are_ignored(simple, cls):
    factors = simple.factors.rename_axis("another name")
    mod = cls(simple.portfolios, factors)
    assert_allclose(mod.factors.ndarray, simple.factors.to_numpy())


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_different_length_reports_the_length(simple, cls):
    with pytest.raises(ValueError, match="number of observations in portfolios"):
        cls(simple.portfolios, simple.factors.iloc[:-1])


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_missing_values_do_not_change_the_check(simple, cls):
    portfolios = simple.portfolios.copy()
    portfolios.iloc[5, 2] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = cls(portfolios, simple.factors)
    assert mod.portfolios.shape[0] == NOBS - 1
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(portfolios, reverse(simple.factors))


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_ivdata_inputs(simple, cls):
    cls(IVData(simple.portfolios), IVData(simple.factors))
    with pytest.raises(ValueError, match="row index of factors"):
        cls(simple.portfolios, IVData(reverse(simple.factors)))
    with pytest.raises(ValueError, match="row index of factors"):
        cls(IVData(simple.portfolios), reverse(simple.factors))
    # IVData built from an array has no labels
    mod = cls(simple.portfolios, IVData(simple.factors.to_numpy()[::-1]))
    assert_allclose(mod.factors.ndarray, simple.factors.to_numpy()[::-1])


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_xarray_inputs(simple, cls):
    xr = pytest.importorskip("xarray")

    def to_xarray(frame, coords=True):
        obs = {"obs": frame.index.to_numpy()} if coords else {}
        return xr.DataArray(
            frame.to_numpy(), dims=["obs", "var"], coords={**obs, "var": list(frame)}
        )

    factors = to_xarray(simple.factors)
    mod = cls(simple.portfolios, factors)
    assert_allclose(mod.factors.ndarray, simple.factors.to_numpy())
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(simple.portfolios, factors.isel(obs=slice(None, None, -1)))
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(to_xarray(simple.portfolios), reverse(simple.factors))
    cls(to_xarray(simple.portfolios), factors)
    # No coordinates, no labels
    plain = to_xarray(reverse(simple.factors), coords=False)
    mod = cls(simple.portfolios, plain)
    assert_allclose(mod.factors.ndarray, reverse(simple.factors).to_numpy())


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_options_do_not_change_the_check(simple, cls):
    kwargs = {"risk_free": True}
    if cls is LinearFactorModel:
        kwargs["sigma"] = np.eye(NPORTFOLIO)
    if cls is TradedFactorModel:
        kwargs = {}
    cls(simple.portfolios, simple.factors, **kwargs)
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(simple.portfolios, reverse(simple.factors), **kwargs)


# Formulas use a single DataFrame


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_from_formula(simple, cls):
    frame = simple.frame()
    assert not isinstance(frame.index, pd.RangeIndex)
    formula = "p0 + p1 + p2 + p3 ~ f0 + f1"
    mod = cls.from_formula(formula, frame)
    raw = cls(simple.portfolios.iloc[:, :4].to_numpy(), simple.factors.to_numpy())
    assert_same_arrays(mod, raw)
    # Portfolios in a separate frame
    portfolios = simple.portfolios
    mod = cls.from_formula("f0 + f1", frame, portfolios=portfolios)
    assert_same_arrays(mod, cls(portfolios.to_numpy(), simple.factors.to_numpy()))
    for corrupt in CORRUPTIONS.values():
        with pytest.raises(ValueError, match="row index of factors does not match"):
            cls.from_formula("f0 + f1", frame, portfolios=corrupt(portfolios))
    # The data has a default index, so there are no labels
    default = frame.reset_index(drop=True)
    cls.from_formula(
        "f0 + f1", default, portfolios=reverse(default[portfolios.columns])
    )


@pytest.mark.parametrize("cls", FITTED, ids=FITTED_IDS)
def test_from_formula_matches_arrays(simple, cls):
    frame = simple.frame()
    res = cls.from_formula("p0 + p1 + p2 + p3 ~ f0 + f1", frame).fit()
    raw = cls(simple.portfolios.iloc[:, :4].to_numpy(), simple.factors.to_numpy()).fit()
    assert_allclose(res.risk_premia, raw.risk_premia)
