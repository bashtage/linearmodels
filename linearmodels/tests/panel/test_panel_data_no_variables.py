"""
PanelData without variables

A DataFrame with a MultiIndex and no columns, or an array with 0 variables, is a panel
with no variables. It is the exog of a PanelOLS model that only has effects. The means,
the counts and the demeaned data have the index of the data and no variables. These
raised errors from pandas (concat of no objects) and NumPy (the maximum of an empty array).
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import xarray as xr

from linearmodels.panel.data import PanelData

NENTITY = 7
NPERIOD = 5


@pytest.fixture(scope="module")
def frame():
    rg = np.random.default_rng(31)
    index = pd.MultiIndex.from_product(
        [np.arange(NENTITY), np.arange(NPERIOD)], names=["firm", "year"]
    )
    nobs = index.shape[0]
    data = pd.DataFrame(
        {
            "w": rg.chisquare(4, nobs) / 4 + 0.2,
            "g": (np.arange(nobs) // 3) % 4,
        },
        index=index,
    )
    return data.iloc[2:]


@pytest.fixture(scope="module")
def empty(frame):
    return pd.DataFrame(index=frame.index)


def test_panel_data_without_variables(empty):
    pdata = PanelData(empty)
    assert pdata.shape == (0, NPERIOD, NENTITY)
    assert pdata.nvar == 0
    assert pdata.vars == []
    assert pdata.values2d.shape == (empty.shape[0], 0)
    assert pdata.values3d.shape == (0, NPERIOD, NENTITY)
    assert pdata.isnull.shape == (empty.shape[0],)
    assert not pdata.isnull.any()
    assert pdata.dataframe.index.equals(empty.index)
    assert PanelData(pdata).shape == (0, NPERIOD, NENTITY)
    assert pdata.copy().shape == (0, NPERIOD, NENTITY)


def test_panel_data_from_arrays_without_variables():
    pdata = PanelData(np.empty((0, NPERIOD, NENTITY)))
    assert pdata.shape == (0, NPERIOD, NENTITY)
    assert pdata.vars == []
    assert pdata.values3d.shape == (0, NPERIOD, NENTITY)
    assert pdata.dataframe.shape == (NPERIOD * NENTITY, 0)
    # The entities and the periods are those of an array with variables
    other = PanelData(np.zeros((1, NPERIOD, NENTITY)))
    assert pdata.dataframe.index.equals(other.dataframe.index)


def test_panel_data_from_data_array_without_variables():
    time = pd.date_range("2000-01-01", periods=NPERIOD, freq="MS")
    entities = [f"firm{i}" for i in range(NENTITY)]
    array = xr.DataArray(
        np.empty((0, NPERIOD, NENTITY)),
        coords={"vars": [], "time": time, "entities": entities},
        dims=["vars", "time", "entities"],
    )
    pdata = PanelData(array)
    assert pdata.shape == (0, NPERIOD, NENTITY)
    assert pdata.entities == entities


@pytest.mark.parametrize("group", ["entity", "time", "both"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_demean_without_variables(frame, empty, group, weighted):
    pdata = PanelData(empty)
    weights = PanelData(frame[["w"]]) if weighted else None
    out = pdata.demean(group, weights=weights)
    assert out.shape == (0, NPERIOD, NENTITY)
    assert out.dataframe.index.equals(empty.index)
    assert out.dataframe.shape == (empty.shape[0], 0)
    if group != "both":
        arr = pdata.demean(group, weights=weights, return_panel=False)
        assert arr.shape == (empty.shape[0], 0)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_demean_both_without_the_dummy_variables(frame, empty, weighted):
    pdata = PanelData(empty)
    weights = PanelData(frame[["w"]]) if weighted else None
    out = pdata.demean("both", weights=weights, low_memory=True)
    assert out.dataframe.shape == (empty.shape[0], 0)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_general_demean_without_variables(frame, empty, weighted):
    pdata = PanelData(empty)
    groups = PanelData(frame[["g"]], convert_dummies=False)
    two_groups = PanelData(
        pd.DataFrame(
            {"g": frame["g"], "h": np.arange(frame.shape[0]) % 3}, index=frame.index
        ),
        convert_dummies=False,
    )
    weights = PanelData(frame[["w"]]) if weighted else None
    for categories in (groups, two_groups):
        out = pdata.general_demean(categories, weights=weights)
        assert out.dataframe.shape == (empty.shape[0], 0)
        assert out.dataframe.index.equals(empty.index)


def test_means_and_counts_without_variables(frame, empty):
    pdata = PanelData(empty)
    weights = PanelData(frame[["w"]])
    for group, ngroup in (("entity", NENTITY), ("time", NPERIOD)):
        for weighted in (None, weights):
            mean = pdata.mean(group, weights=weighted)
            assert mean.shape == (ngroup, 0)
        assert pdata.count(group).shape == (ngroup, 0)
    assert list(pdata.mean("entity").index) == pdata.entities
    assert list(pdata.mean("time").index) == pdata.time


def test_variables_are_not_changed_by_the_checks(frame):
    # A panel with variables still means what it did
    pdata = PanelData(frame[["w"]])
    demeaned = pdata.demean("entity")
    expected = frame["w"] - frame["w"].groupby(level=0).transform("mean")
    assert_allclose(demeaned.values2d.ravel(), expected.to_numpy())
