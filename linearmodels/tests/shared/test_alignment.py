import numpy as np
import pandas as pd
import pytest

from linearmodels.iv.data import IVData
from linearmodels.panel.data import PanelData
from linearmodels.shared.alignment import (
    check_alignment,
    check_row_alignment,
    differs,
    is_default_index,
    is_labelled,
    labels_1d,
    misaligned,
    reference_labels,
    row_labels,
)

N = 6


@pytest.fixture
def index():
    return pd.Index(list("abcdef"), name="obs")


@pytest.fixture
def series(index):
    return pd.Series(np.arange(N, dtype=float), index=index, name="s")


@pytest.fixture
def frame(index):
    return pd.DataFrame(
        np.arange(2.0 * N).reshape(N, 2), index=index, columns=["a", "b"]
    )


@pytest.fixture
def default_frame():
    return pd.DataFrame(np.arange(2.0 * N).reshape(N, 2), columns=["a", "b"])


@pytest.fixture
def multi_index():
    return pd.MultiIndex.from_product([["x", "y"], [1, 2, 3]], names=["e", "t"])


def test_is_default_index():
    assert is_default_index(pd.RangeIndex(N))
    assert is_default_index(pd.Index(np.arange(N)))
    assert is_default_index(pd.RangeIndex(0))
    assert is_default_index(pd.RangeIndex(N, name="named"))
    assert not is_default_index(pd.RangeIndex(N)[::-1])
    assert not is_default_index(pd.RangeIndex(1, N + 1))
    assert not is_default_index(pd.Index(np.arange(N)[::-1]))
    assert not is_default_index(pd.Index(list("abcdef")))
    assert not is_default_index(pd.date_range("2000-01-01", periods=N))
    assert not is_default_index(pd.MultiIndex.from_product([[0, 1], [0, 1, 2]]))


def test_is_labelled_pandas(series, frame, default_frame, multi_index):
    assert is_labelled(series)
    assert not is_labelled(series.reset_index(drop=True))
    assert is_labelled(series.iloc[::-1].reset_index(drop=True)) is False
    assert is_labelled(frame)
    assert not is_labelled(default_frame)
    assert is_labelled(pd.Series(np.arange(N), index=multi_index))
    assert is_labelled(pd.DataFrame(np.ones((N, 2)), index=multi_index))
    # Reversed range
    assert is_labelled(default_frame.iloc[::-1])
    assert is_labelled(pd.Series(np.arange(N)).iloc[::-1])


def test_is_labelled_columns(default_frame):
    # Columns are variables, unless they identify entities
    assert not is_labelled(default_frame)
    assert not is_labelled(default_frame, columns=False)
    assert is_labelled(default_frame, columns=True)
    unnamed = pd.DataFrame(np.ones((N, 2)))
    assert not is_labelled(unnamed, columns=True)
    named_rows = default_frame.set_axis(list("abcdef"), axis=0)
    assert is_labelled(named_rows)
    assert is_labelled(named_rows, columns=True)


def test_is_labelled_other_types(series, multi_index):
    assert not is_labelled(None)
    assert not is_labelled(np.arange(N))
    assert not is_labelled(np.ones((N, 2)))
    assert not is_labelled(list(range(N)))
    assert not is_labelled(1.0)
    assert is_labelled(PanelData(pd.Series(np.arange(N), index=multi_index)))
    assert is_labelled(IVData(series))
    assert not is_labelled(IVData(series.reset_index(drop=True)))
    assert not is_labelled(IVData(series.to_numpy()))
    assert not is_labelled(IVData(None, nobs=N))
    # IVData built from IVData
    assert is_labelled(IVData(IVData(series)))
    assert not is_labelled(IVData(IVData(series.to_numpy())))


def test_is_labelled_xarray(frame):
    xr = pytest.importorskip("xarray")
    labelled = xr.DataArray(frame.to_numpy(), dims=["obs", "var"])
    # Any DataArray can carry labels. row_labels looks at the coordinates
    assert is_labelled(labelled)
    assert row_labels(labelled) is None
    coords = {"obs": list(frame.index), "var": list(frame.columns)}
    labelled = xr.DataArray(frame.to_numpy(), dims=["obs", "var"], coords=coords)
    assert row_labels(labelled).equals(frame.index)
    assert is_labelled(IVData(labelled))


def test_row_labels(series, frame, default_frame, index):
    assert row_labels(series).equals(index)
    assert row_labels(frame).equals(index)
    assert row_labels(series.reset_index(drop=True)) is None
    assert row_labels(default_frame) is None
    assert row_labels(None) is None
    assert row_labels(series.to_numpy()) is None
    assert row_labels([1, 2, 3]) is None
    assert row_labels(default_frame.iloc[::-1]).equals(pd.RangeIndex(N)[::-1])


def test_row_labels_ivdata(frame, index):
    ivd = IVData(frame)
    assert row_labels(ivd).equals(index)
    assert row_labels(IVData(frame.to_numpy())) is None
    assert row_labels(IVData(frame.reset_index(drop=True))) is None
    # The rows that are used are the ones that are not dropped
    drop = np.zeros(N, dtype=bool)
    drop[[1, 4]] = True
    ivd.drop(drop)
    assert row_labels(ivd).equals(index[~drop])
    # An array that has had rows dropped has no labels, even though its
    # remaining row numbers are not 0, 1, ...
    raw = IVData(frame.to_numpy())
    raw.drop(drop)
    assert row_labels(raw) is None
    assert row_labels(IVData(ivd)).equals(index[~drop])


def test_labels_1d(series, frame, index, multi_index):
    assert labels_1d(series).equals(index)
    assert labels_1d(frame).equals(index)
    assert labels_1d(IVData(frame)).equals(index)
    panel = PanelData(pd.Series(np.arange(N), index=multi_index))
    assert labels_1d(panel).equals(multi_index)


def test_labels_1d_xarray(frame, index):
    xr = pytest.importorskip("xarray")
    coords = {"obs": list(frame.index), "var": list(frame.columns)}
    array = xr.DataArray(frame.to_numpy(), dims=["obs", "var"], coords=coords)
    assert labels_1d(array).equals(index)
    one_d = xr.DataArray(
        frame.to_numpy()[:, 0], dims=["obs"], coords={"obs": list(frame.index)}
    )
    assert labels_1d(one_d).equals(index)


def test_differs(index):
    assert not differs(index, index)
    assert not differs(index.copy(), index)
    assert differs(index[::-1], index)
    assert differs(pd.Index(list("abcdeg")), index)
    # A default index has no labels
    assert not differs(pd.RangeIndex(N), index)
    assert differs(pd.RangeIndex(N)[::-1], index)
    # Names are not labels
    assert not differs(index.rename("another"), index)


def test_misaligned_message():
    exc = misaligned("exog", "The row index")
    assert isinstance(exc, ValueError)
    message = str(exc)
    assert message.startswith("The row index of exog does not match dependent.")
    assert "in the same order as dependent" in message
    assert "reindex exog to the index of dependent" in message
    assert ".to_numpy()" in message
    exc = misaligned("weights", "The time index", "portfolios")
    assert str(exc).startswith("The time index of weights does not match portfolios.")
    assert "to the index of portfolios" in str(exc)


def test_check_row_alignment(index):
    check_row_alignment("exog", index, index)
    check_row_alignment("exog", index.copy(), index)
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        check_row_alignment("exog", index[::-1], index)
    with pytest.raises(ValueError, match="row index of x does not match reference"):
        check_row_alignment("x", index[::-1], index, "reference")
    # Default index and a different number of rows are not compared
    check_row_alignment("exog", pd.RangeIndex(N), index)
    check_row_alignment("exog", index[:-1], index)
    check_row_alignment("exog", index[::-1][:-1], index)


def test_check_alignment(series, frame, default_frame, index):
    inputs = {"exog": frame, "weights": series, "clusters": None}
    check_alignment(index, inputs)
    check_alignment(None, {"exog": frame.iloc[::-1]})
    check_alignment(index, {})
    check_alignment(index, {"exog": default_frame.iloc[:, :1], "w": np.ones(N)})
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        check_alignment(index, {"exog": frame.iloc[::-1]})
    with pytest.raises(ValueError, match="row index of w does not match the dependent"):
        check_alignment(
            index,
            {"exog": frame, "w": series.iloc[::-1]},
            reference="the dependent",
        )
    # The first input that does not match is reported
    bad = {"a": frame, "b": frame.iloc[::-1], "c": series.iloc[::-1]}
    with pytest.raises(ValueError, match="row index of b"):
        check_alignment(index, bad)


def test_check_alignment_first_input_with_labels_is_the_reference(
    series, frame, default_frame, index
):
    # dependent has no labels, so the first input that has labels sets them
    check_alignment(None, {"exog": frame, "weights": series})
    reversed_frame = frame.iloc[::-1]
    with pytest.raises(ValueError, match="row index of weights does not match exog"):
        check_alignment(None, {"exog": frame, "weights": series.iloc[::-1]})
    with pytest.raises(ValueError, match="row index of weights does not match exog"):
        check_alignment(None, {"exog": reversed_frame, "weights": series})
    # Arrays, missing inputs and a default index do not set the labels
    inputs = {
        "none": None,
        "array": np.ones((N, 2)),
        "default": default_frame,
        "x": frame,
        "y": series,
    }
    check_alignment(None, inputs)
    with pytest.raises(ValueError, match="row index of y does not match x"):
        check_alignment(None, {**inputs, "y": series.iloc[::-1]})
    # The labels of dependent are used if there are any
    with pytest.raises(ValueError, match="row index of x does not match dependent"):
        check_alignment(index[::-1], inputs)
    # Inputs can be given in order as pairs, where a name can be used twice
    pairs = [("absorb", frame), ("absorb", series), ("weights", reversed_frame)]
    with pytest.raises(ValueError, match="row index of weights does not match absorb"):
        check_alignment(None, pairs)
    check_alignment(None, pairs[:2])


def test_check_alignment_reference_name(series, frame, index):
    # The name of dependent is used when it sets the labels
    with pytest.raises(ValueError, match="row index of x does not match the dependent"):
        check_alignment(index, {"x": frame.iloc[::-1]}, reference="the dependent")
    # Otherwise it is the first input
    with pytest.raises(ValueError, match="row index of y does not match x"):
        check_alignment(
            None, {"x": frame, "y": series.iloc[::-1]}, reference="the dependent"
        )
    with pytest.raises(ValueError, match=r"in the same order as x\. Either reindex y"):
        check_alignment(None, {"x": frame, "y": series.iloc[::-1]})


def test_reference_labels(series, frame, default_frame, index):
    labels, name = reference_labels(index, {"exog": frame.iloc[::-1]})
    assert labels is index
    assert name == "dependent"
    labels, name = reference_labels(index, {"exog": frame}, reference="y")
    assert labels is index
    assert name == "y"
    # The first input with labels is the reference without dependent
    inputs = {"a": None, "b": np.ones(N), "c": default_frame, "d": series, "e": frame}
    labels, name = reference_labels(None, inputs)
    assert labels is not None
    assert labels.equals(index)
    assert name == "d"
    labels, name = reference_labels(None, {"e": frame.iloc[::-1], "d": series})
    assert labels is not None
    assert labels.equals(index[::-1])
    assert name == "e"
    # pairs are used in order
    labels, name = reference_labels(None, [("x", np.ones(N)), ("x", series)])
    assert labels is not None
    assert name == "x"
    # There are no labels to use
    labels, name = reference_labels(
        None, {"a": None, "b": np.ones(N), "c": default_frame}
    )
    assert labels is None
    assert name == "dependent"
    labels, name = reference_labels(None, {}, reference="y")
    assert labels is None
    assert name == "y"
