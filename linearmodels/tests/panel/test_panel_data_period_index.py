"""
Time labels that are a PeriodIndex

The periods are converted to the timestamps at their start with ``to_timestamp()``,
so that the results are those of a panel with a DatetimeIndex. A PeriodIndex was
rejected, with the error that the time labels must be numeric or date-like.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._data_checks import (
    EXOG,
    NPERIOD,
    PERIODS,
    assert_equal_results,
    make_frame,
)
from linearmodels.tests.panel._utility import access_attributes


def period_and_timestamp_frames(unbalanced=True):
    periods = make_frame(PERIODS, unbalanced=unbalanced)
    # The same panel with dates, written without to_timestamp
    dates = pd.date_range("2001-01-01", periods=NPERIOD, freq="MS")
    stamps = make_frame(dates, unbalanced=unbalanced)
    assert isinstance(periods.index.levels[1], pd.PeriodIndex)
    assert isinstance(stamps.index.levels[1], pd.DatetimeIndex)
    return periods, stamps


def test_period_index_in_panel_data():
    periods, stamps = period_and_timestamp_frames()
    original = periods.copy()
    pdata = PanelData(periods)
    expected = PanelData(stamps)
    # The periods are replaced by the timestamps at their start
    levels = pdata.index.levels[1]
    assert isinstance(levels, pd.DatetimeIndex)
    assert list(levels) == list(PERIODS.to_timestamp())
    assert pdata.time == expected.time
    assert pdata.index.equals(expected.index)
    assert pdata.shape == expected.shape
    assert_allclose(pdata.values3d, expected.values3d)
    # The frame of the caller is not changed
    assert isinstance(periods.index.levels[1], pd.PeriodIndex)
    pd.testing.assert_frame_equal(periods, original)


@pytest.mark.parametrize("freq", ["D", "M", "Q"])
def test_period_index_frequencies(freq):
    periods = pd.period_range("2001-01-01", periods=5, freq=freq)
    index = pd.MultiIndex.from_product([range(3), periods], names=["id", "period"])
    data = pd.DataFrame({"y": np.arange(15.0)}, index=index)
    time = PanelData(data).index.levels[1]
    assert list(time) == [p.to_timestamp() for p in periods]


def test_period_index_in_a_wide_frame():
    # Time by entity: the periods are the index of the frame
    rg = np.random.default_rng(2)
    values = rg.standard_normal((NPERIOD, 4))
    wide = pd.DataFrame(values, index=PERIODS, columns=list("abcd"))
    expected = pd.DataFrame(values, index=PERIODS.to_timestamp(), columns=list("abcd"))
    pdata = PanelData(wide)
    assert pdata.index.equals(PanelData(expected).index)
    assert list(pdata.time) == list(PERIODS.to_timestamp())


def test_other_time_labels_are_still_rejected():
    index = pd.MultiIndex.from_product([range(3), ["a", "b"]])
    data = pd.DataFrame({"y": np.arange(6.0)}, index=index)
    with pytest.raises(ValueError, match="must be either numeric or date-like"):
        PanelData(data)


MODEL_OPTIONS = {
    "PooledOLS": (PooledOLS, {}),
    "BetweenOLS": (BetweenOLS, {}),
    "FirstDifferenceOLS": (FirstDifferenceOLS, {}),
    "RandomEffects": (RandomEffects, {}),
    "FamaMacBeth": (FamaMacBeth, {}),
    "PanelOLS-entity": (PanelOLS, {"entity_effects": True}),
    "PanelOLS-time": (PanelOLS, {"time_effects": True}),
    "PanelOLS-both": (PanelOLS, {"entity_effects": True, "time_effects": True}),
}
COVARIANCE = {
    "unadjusted": {"cov_type": "unadjusted"},
    "robust": {"cov_type": "robust"},
    "clustered": {"cov_type": "clustered", "clusters": "cl"},
    "kernel": {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 2},
    "ac": {"cov_type": "ac", "kernel": "parzen", "bandwidth": 2},
}
# The covariance estimators that a model does not support
UNSUPPORTED = {
    "BetweenOLS": ("kernel", "ac"),
    "FamaMacBeth": ("clustered", "ac"),
}
CASES = [
    pytest.param(model, cov, id=f"{model}-{cov}")
    for model in MODEL_OPTIONS
    for cov in COVARIANCE
    if cov not in UNSUPPORTED.get(model.split("-")[0], ())
]


def fit_model(model, frame, cov):
    cls, options = MODEL_OPTIONS[model]
    exog = frame[EXOG] if cls is not FirstDifferenceOLS else frame[EXOG[1:]]
    mod = cls(frame["y"], exog, weights=frame["w"], **options)
    fit_options = dict(COVARIANCE[cov])
    if "clusters" in fit_options:
        # The column that has the clusters of the entities
        fit_options["clusters"] = frame[[fit_options["clusters"]]]
    return mod.fit(**fit_options)


@pytest.mark.parametrize(("model", "cov"), CASES)
@pytest.mark.parametrize("unbalanced", [False, True], ids=["balanced", "unbalanced"])
def test_period_index_equals_datetime_index(model, cov, unbalanced):
    periods, stamps = period_and_timestamp_frames(unbalanced)
    res = fit_model(model, periods, cov)
    expected = fit_model(model, stamps, cov)
    assert_equal_results(res, expected)
    access_attributes(res)


def test_period_index_with_clusters_weights_and_other_effects():
    periods, stamps = period_and_timestamp_frames()
    out = []
    for frame in (periods, stamps):
        mod = PanelOLS(
            frame["y"],
            frame[EXOG[1:]],
            weights=frame["w"],
            entity_effects=True,
            other_effects=frame[["g"]],
        )
        out.append(mod.fit(cov_type="clustered", clusters=frame[["g"]], use_lsdv=True))
    res, expected = out
    assert_equal_results(res, expected)


def test_period_index_and_timestamps_can_be_mixed():
    # Each input is converted, so the labels agree
    periods, stamps = period_and_timestamp_frames()
    res = PooledOLS(periods["y"], stamps[EXOG], weights=periods["w"]).fit()
    expected = PooledOLS(stamps["y"], stamps[EXOG], weights=stamps["w"]).fit()
    assert_equal_results(res, expected)


def test_period_index_formula():
    periods, stamps = period_and_timestamp_frames()
    formula = "y ~ 1 + x1 + x2 + EntityEffects"
    res = PanelOLS.from_formula(formula, periods, weights=periods["w"]).fit()
    expected = PanelOLS.from_formula(formula, stamps, weights=stamps["w"]).fit()
    assert_equal_results(res, expected)


def test_period_index_predict_in_sample():
    periods, stamps = period_and_timestamp_frames()
    res = PooledOLS(periods["y"], periods[EXOG]).fit()
    expected = PooledOLS(stamps["y"], stamps[EXOG]).fit()
    pd.testing.assert_frame_equal(
        res.predict(fitted=True, effects=True, idiosyncratic=True, missing=True),
        expected.predict(fitted=True, effects=True, idiosyncratic=True, missing=True),
    )
