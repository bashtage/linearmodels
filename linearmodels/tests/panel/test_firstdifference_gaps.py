"""
The periods that FirstDifferenceOLS differences

An observation is differenced with the observation of the same entity in the
preceding time label of the panel, which is the preceding one of the sorted labels
that occur for any entity. An entity has no difference for its first observation, or
for an observation that follows a period in which the entity is not observed.
Differencing across a gap, as plm does, would give a difference of several periods.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import FirstDifferenceOLS
from linearmodels.panel.data import PanelData


def make_frame(times, seed=7):
    """The data of entities with the periods in ``times``, which maps name to list"""
    rg = np.random.default_rng(seed)
    rows = [(name, t) for name, periods in times.items() for t in periods]
    index = pd.MultiIndex.from_tuples(rows, names=["entity", "period"])
    x1 = rg.standard_normal(len(rows))
    x2 = rg.standard_normal(len(rows))
    y = 0.5 * x1 - 0.3 * x2 + rg.standard_normal(len(rows))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=index)


def expected_differences(frame, pairs):
    """The differences of the (entity, period, preceding period) triples"""
    rows = []
    for entity, period, previous in pairs:
        current = frame.loc[(entity, period)]
        before = frame.loc[(entity, previous)]
        rows.append(current - before)
    index = pd.MultiIndex.from_tuples(
        [(e, p) for e, p, _ in pairs], names=frame.index.names
    )
    return pd.DataFrame(np.array(rows), index=index, columns=frame.columns)


# Entity B is not observed in the period 3, and entity C does not enter until 2 and
# is not observed in the period 5. The periods of the panel are 1, ..., 6
GAPS = {"A": [1, 2, 3, 4, 5], "B": [1, 2, 4, 5], "C": [2, 3, 4, 6]}
GAPS_PAIRS = [
    ("A", 2, 1),
    ("A", 3, 2),
    ("A", 4, 3),
    ("A", 5, 4),
    ("B", 2, 1),
    ("B", 5, 4),
    ("C", 3, 2),
    ("C", 4, 3),
]
# No entity is observed in the period 3, so the labels of the panel are 1, 2, 4, 5
# and the preceding label of 4 is 2
MISSING_LABEL = {"A": [1, 2, 4, 5], "B": [1, 2, 4, 5], "C": [2, 4, 5]}
MISSING_LABEL_PAIRS = [
    ("A", 2, 1),
    ("A", 4, 2),
    ("A", 5, 4),
    ("B", 2, 1),
    ("B", 4, 2),
    ("B", 5, 4),
    ("C", 4, 2),
    ("C", 5, 4),
]
CASES = [
    pytest.param(GAPS, GAPS_PAIRS, id="gaps"),
    pytest.param(MISSING_LABEL, MISSING_LABEL_PAIRS, id="label-missing-for-all"),
]


@pytest.mark.parametrize(("times", "pairs"), CASES)
def test_first_difference_of_panel_data(times, pairs):
    frame = make_frame(times)
    expected = expected_differences(frame, pairs)
    diffs = PanelData(frame).first_difference().dataframe
    assert diffs.index.equals(expected.index)
    assert_allclose(diffs.to_numpy(), expected.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize(("times", "pairs"), CASES)
def test_first_difference_ols_uses_adjacent_periods_only(times, pairs):
    frame = make_frame(times)
    expected = expected_differences(frame, pairs)
    res = FirstDifferenceOLS(frame["y"], frame[["x1", "x2"]]).fit()
    exog = expected[["x1", "x2"]].to_numpy()
    params = np.linalg.lstsq(exog, expected["y"].to_numpy(), rcond=None)[0]
    assert res.nobs == len(pairs)
    assert_allclose(res.params, params)
    assert res.resids.index.equals(expected.index)
    assert_allclose(res.resids, expected["y"].to_numpy() - exog @ params)
    assert res.df_resid == len(pairs) - 2


def test_differences_across_a_gap_are_not_used():
    # The difference of B between the periods 4 and 2 is not an observation
    frame = make_frame(GAPS)
    diffs = PanelData(frame).first_difference().dataframe
    assert ("B", 4) not in diffs.index
    assert ("C", 6) not in diffs.index
    # An entity has no difference for its first observation
    assert ("A", 1) not in diffs.index
    assert ("C", 2) not in diffs.index
    assert diffs.shape[0] == len(GAPS_PAIRS)
