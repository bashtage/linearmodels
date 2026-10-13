"""
Duplicated (entity, time) rows

A combination of an entity and a time period can appear once, as in plm and Stata.
The models accepted duplicated rows, except the first difference estimator, which
raised an error from pandas, and treated the rows as separate observations.
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
from linearmodels.tests.panel._data_checks import EXOG, PERIODS, make_frame


@pytest.fixture(scope="module")
def frame():
    return make_frame(PERIODS.to_timestamp())


def duplicated(frame, rows=(3, 9)):
    """The frame with the rows ``rows`` repeated at the end"""
    return pd.concat([frame, frame.iloc[list(rows)]])


def test_first_duplicated_entry_is_named():
    # The rows 3 and 4 repeat the rows 1 and 2. The first row that repeats an earlier
    # row is the fourth, the observation of firm 0 in year 1
    index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 0), (0, 1), (1, 0)], names=["firm", "year"]
    )
    data = pd.DataFrame({"y": np.arange(5.0)}, index=index)
    match = r"can appear only once.*2 row\(s\) repeat an earlier row.*firm=0, year=1\."
    with pytest.raises(ValueError, match=match):
        PanelData(data)
    with pytest.raises(ValueError, match=match):
        PanelData(data["y"])


def test_duplicated_entry_uses_the_default_names():
    index = pd.MultiIndex.from_tuples([("a", 2), ("a", 2)])
    with pytest.raises(ValueError, match=r"entity=a, time=2\."):
        PanelData(pd.DataFrame({"y": [1.0, 2.0]}, index=index))


def test_unique_index_is_accepted():
    index = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 2)])
    data = pd.DataFrame({"y": np.arange(4.0)}, index=index)
    assert PanelData(data).shape == (1, 3, 2)


def test_duplicated_labels_of_a_wide_frame():
    # Time by entity: repeated entity labels (columns) or time labels (rows)
    rg = np.random.default_rng(0)
    wide = pd.DataFrame(rg.standard_normal((4, 3)), columns=["a", "b", "c"])
    assert PanelData(wide).shape == (1, 4, 3)
    repeated = wide.set_axis(["a", "b", "a"], axis=1)
    with pytest.raises(ValueError, match=r"repeat an earlier row.*entity=a, time=0\."):
        PanelData(repeated)
    repeated = wide.set_axis([0, 1, 1, 2], axis=0)
    with pytest.raises(ValueError, match=r"the time label 1 appears more than once"):
        PanelData(repeated)


MODELS = [
    pytest.param(PooledOLS, {}, id="PooledOLS"),
    pytest.param(BetweenOLS, {}, id="BetweenOLS"),
    pytest.param(FirstDifferenceOLS, {}, id="FirstDifferenceOLS"),
    pytest.param(RandomEffects, {}, id="RandomEffects"),
    pytest.param(FamaMacBeth, {}, id="FamaMacBeth"),
    pytest.param(PanelOLS, {}, id="PanelOLS"),
    pytest.param(PanelOLS, {"entity_effects": True}, id="PanelOLS-entity"),
    pytest.param(
        PanelOLS, {"entity_effects": True, "time_effects": True}, id="PanelOLS-both"
    ),
]


@pytest.mark.parametrize(("cls", "options"), MODELS)
def test_models_reject_duplicated_rows(frame, cls, options):
    data = duplicated(frame)
    # The row 3 is repeated first
    entry = frame.index[3]
    match = rf"dependent: .*firm={entry[0]}, month={entry[1]}\."
    with pytest.raises(ValueError, match=match):
        cls(data["y"], data[EXOG], **options)


@pytest.mark.parametrize(("cls", "options"), MODELS)
def test_models_reject_duplicated_rows_of_exog(frame, cls, options):
    data = duplicated(frame)
    with pytest.raises(ValueError, match=r"exog: .*repeat an earlier row"):
        cls(frame["y"], data[EXOG], **options)


@pytest.mark.parametrize(("cls", "options"), MODELS)
def test_formula_models_reject_duplicated_rows(frame, cls, options):
    data = duplicated(frame)
    formula = "y ~ 1 + x1 + x2"
    if cls is PanelOLS and options:
        formula += " + EntityEffects"
    with pytest.raises(ValueError, match=r"data: .*repeat an earlier row"):
        cls.from_formula(formula, data)


def test_weights_clusters_and_effects_reject_duplicated_rows(frame):
    # The inputs have a value for each row of dependent, and two rows have the same
    # entity and time
    labels = list(frame.index)
    labels[10] = labels[6]
    repeated = pd.MultiIndex.from_tuples(labels, names=frame.index.names)
    entry = f"firm={labels[6][0]}, month={labels[6][1]}"
    for name in ("w", "g"):
        assert frame[name].set_axis(repeated).index.duplicated().sum() == 1
    with pytest.raises(ValueError, match=rf"repeat an earlier row.*{entry}"):
        PooledOLS(frame["y"], frame[EXOG], weights=frame["w"].set_axis(repeated))
    mod = PooledOLS(frame["y"], frame[EXOG])
    clusters = frame["g"].set_axis(repeated)
    with pytest.raises(
        ValueError, match=rf"clusters: .*repeat an earlier row.*{entry}"
    ):
        mod.fit(cov_type="clustered", clusters=clusters)
    effects = frame["g"].set_axis(repeated)
    with pytest.raises(
        ValueError, match=rf"other_effects: .*repeat an earlier row.*{entry}"
    ):
        PanelOLS(frame["y"], frame[["x1", "x2"]], other_effects=effects)


def test_removing_the_duplicates_gives_the_same_results(frame):
    data = duplicated(frame)
    unique = data.loc[~data.index.duplicated()]
    expected = PanelOLS(frame["y"], frame[EXOG[1:]], entity_effects=True).fit()
    res = PanelOLS(unique["y"], unique[EXOG[1:]], entity_effects=True).fit()
    assert_allclose(res.params, expected.params)
