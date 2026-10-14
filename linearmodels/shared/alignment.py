"""
Pairing the rows of inputs that carry labels.

Estimators pair the rows of their inputs by position: the first row of the
regressors belongs to the first row of the dependent variable, and so on. An
input that carries labels (a pandas Series or DataFrame with an index, an
xarray DataArray, or the data containers built from these) must use the same
labels, in the same order, as the dependent variable. Otherwise the values
would be silently attached to the wrong observations.

Inputs without labels, such as NumPy arrays and pandas objects that only have
a default index (0, 1, ..., n-1), are paired by position and only need to have
the right shape.

If the dependent variable does not carry labels, the first of the other inputs
that carries labels sets the labels instead, and the inputs that carry labels
after it must use them. Two inputs with labels in a different order would
otherwise be silently attached to the wrong observations in the same way.

Dates and time deltas are compared by value. The same dates are the same
labels when they are stored at different resolutions, e.g., ``datetime64[us]``
and ``datetime64[ns]``, which is what pandas 3 does. In pandas 2 these indexes
are not equal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)

# The resolutions of dates and time deltas in pandas 2, from the coarsest to the
# finest. Before pandas 2.0 the resolution is always nanoseconds, and indexes
# do not have a unit.
_UNITS = ("s", "ms", "us", "ns")


def is_default_index(index: Index) -> bool:
    """
    Whether an index only numbers the rows, i.e., it is 0, 1, ..., n-1.

    Parameters
    ----------
    index : Index
        The index to check.

    Returns
    -------
    bool
        True if the index is a default index, which does not have labels.
    """
    return not isinstance(index, MultiIndex) and index.equals(RangeIndex(len(index)))


def is_labelled(value: object, *, columns: bool = False) -> bool:
    """
    Whether the rows of an input are identified by labels.

    pandas, xarray, PanelData and IVData inputs can carry labels. Raw arrays do
    not, and are paired with the other inputs by position. The same is true of
    a pandas Series or DataFrame that has a default index, since there are no
    labels to compare.

    Parameters
    ----------
    value : object
        The input to check.
    columns : bool, optional
        Whether the columns of a DataFrame also carry labels, so that a
        DataFrame with a default index and non-default columns is labelled.
        This is the case when the columns identify the entities of a panel. The
        default is False, which is appropriate when the columns are variables.

    Returns
    -------
    bool
        True if the rows are identified by labels.
    """
    from linearmodels.iv.data import IVData  # noqa: PLC0415
    from linearmodels.panel.data import PanelData  # noqa: PLC0415

    if isinstance(value, Series):
        return not is_default_index(value.index)
    if isinstance(value, DataFrame):
        return not is_default_index(value.index) or (
            columns and not is_default_index(value.columns)
        )
    if isinstance(value, IVData):
        # IVData stores the object it was built from
        return is_labelled(value.original)
    return isinstance(value, PanelData) or _is_data_array(value)


def _is_data_array(value: object) -> bool:
    """Whether value is an xarray DataArray, if xarray is installed."""
    try:
        from xarray import DataArray  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(value, DataArray)


def labels_1d(value: Any) -> Index:
    """
    Row labels of a Series, DataFrame, DataArray, IVData or PanelData.

    Parameters
    ----------
    value : Series, DataFrame, DataArray, IVData or PanelData
        The input. A DataArray must have named dimensions, and the labels are
        taken from the first dimension.

    Returns
    -------
    Index
        The labels of the rows.
    """
    from linearmodels.iv.data import IVData  # noqa: PLC0415
    from linearmodels.panel.data import PanelData  # noqa: PLC0415

    if isinstance(value, (Series, DataFrame)):
        return value.index
    if isinstance(value, IVData):
        return value.pandas.index
    if isinstance(value, PanelData):
        return value.index
    return value.get_index(value.dims[0])


def row_labels(value: object) -> Index | None:
    """
    Row labels of an input, if it carries labels.

    Parameters
    ----------
    value : object
        The input, or None.

    Returns
    -------
    Index or None
        The labels of the rows, or None if the rows are not identified by
        labels. Only the index of the rows is considered, so a DataFrame
        with a default index is not labelled even if its columns are named.
    """
    if not is_labelled(value):
        return None
    labels = labels_1d(value)
    return None if is_default_index(labels) else labels


def _unit(index: Index) -> str | None:
    """Resolution of the dates or time deltas of an index, None if it has none."""
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        # Before pandas 2.0 an index does not have a unit
        unit = getattr(index, "unit", None)
        return unit if unit in _UNITS else None
    return None


def _finer_unit(left: Index, right: Index) -> str | None:
    """
    The finer resolution of two indexes that have different resolutions.

    Parameters
    ----------
    left : Index
        The first index.
    right : Index
        The second index.

    Returns
    -------
    str or None
        The finer of the resolutions if both are indexes of dates, or both are
        indexes of time deltas, that have different resolutions. None otherwise.
    """
    unit, other = _unit(left), _unit(right)
    if type(left) is not type(right) or unit is None or other is None or unit == other:
        return None
    return max(unit, other, key=_UNITS.index)


def _as_unit(left: Index, right: Index, unit: str) -> tuple[Index, Index] | None:
    """
    The dates or time deltas of two indexes at the same resolution.

    Parameters
    ----------
    left : DatetimeIndex or TimedeltaIndex
        The first index.
    right : DatetimeIndex or TimedeltaIndex
        The second index, which has the same type as the first.
    unit : str
        The resolution, which is the finer of the resolutions of the indexes.

    Returns
    -------
    tuple of Index or None
        The indexes at the resolution. None if a value cannot be represented at
        the resolution, e.g., a date in the year 3000 in nanoseconds. The index
        with the finer resolution cannot hold that value, and so the values
        are different.
    """
    try:
        return (
            left.as_unit(unit),  # type: ignore[attr-defined]
            right.as_unit(unit),  # type: ignore[attr-defined]
        )
    except ValueError:
        # OutOfBoundsDatetime and OutOfBoundsTimedelta, which are ValueErrors
        return None


def _same_multi_index(labels: MultiIndex, expected: MultiIndex) -> bool:
    """
    Whether two MultiIndexes are the same, comparing the dates in the levels by value.

    This is used for MultiIndexes that ``MultiIndex.equals`` reports as
    different, to find out if only the resolutions of their dates differ.
    """
    if labels.nlevels != expected.nlevels:
        return False
    pairs = zip(labels.levels, expected.levels, strict=True)
    if all(_finer_unit(left, right) is None for left, right in pairs):
        return False
    converted_labels: list[Index] = []
    converted_expected: list[Index] = []
    for i in range(labels.nlevels):
        # The values in the rows, since a level can have values that are not used
        left, right = labels.get_level_values(i), expected.get_level_values(i)
        unit = _finer_unit(left, right)
        if unit is not None:
            converted = _as_unit(left, right, unit)
            if converted is None:
                return False
            left, right = converted
        converted_labels.append(left)
        converted_expected.append(right)
    # The levels have the same type and resolution now, and so MultiIndex.equals
    # compares everything else as before
    return MultiIndex.from_arrays(converted_labels).equals(
        MultiIndex.from_arrays(converted_expected)
    )


def same_labels(labels: Index, expected: Index) -> bool:
    """
    Whether two sets of labels are the same, where dates are compared by value.

    The labels are the same if they have the same values in the same order.
    Dates and time deltas are compared by value, and so the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels. ``Index.equals`` treats them in this way in pandas 3, but
    not in pandas 2.

    Parameters
    ----------
    labels : Index
        The labels to check.
    expected : Index
        The expected labels.

    Returns
    -------
    bool
        True if labels has the same values in the same order as expected.

    Notes
    -----
    Dates and time deltas are compared at the finer of their resolutions. A
    value that cannot be represented at that resolution, e.g., a date in the
    year 3000 in nanoseconds, cannot be in the index that has the finer
    resolution, and so the labels are different. The levels of a ``MultiIndex``
    are compared in the same way. Everything else is compared as by
    ``Index.equals``, so that dates in different time zones, and dates with and
    without a time zone, are different labels as they are in pandas 3.
    """
    try:
        if labels.equals(expected):
            return True
    except ValueError:
        # pandas 3 raises an OutOfBoundsDatetime error when it compares a
        # MultiIndex that has dates which the resolution of the other cannot
        # hold. These are different labels, and the levels are compared below.
        pass
    if len(labels) != len(expected):
        return False
    if isinstance(labels, MultiIndex) and isinstance(expected, MultiIndex):
        return _same_multi_index(labels, expected)
    unit = _finer_unit(labels, expected)
    if unit is None:
        return False
    converted = _as_unit(labels, expected, unit)
    return converted is not None and converted[0].equals(converted[1])


def differs(labels: Index, expected: Index) -> bool:
    """
    Whether labels, other than a default index, are not the expected labels.

    Parameters
    ----------
    labels : Index
        The labels to check.
    expected : Index
        The expected labels.

    Returns
    -------
    bool
        True if labels is not a default index and is not the same as expected.
        Dates are compared by value, see ``same_labels``.
    """
    return not is_default_index(labels) and not same_labels(labels, expected)


def misaligned(name: str, what: str, reference: str = "dependent") -> ValueError:
    """
    Error to raise when the labels of an input are not those of the reference.

    Parameters
    ----------
    name : str
        Description of the input that does not match, which is used in the
        message.
    what : str
        The part of the labels that does not match, e.g., "The row index".
    reference : str, optional
        Description of the input that sets the labels. The default is
        "dependent".

    Returns
    -------
    ValueError
        The error, which explains how to resolve the mismatch.
    """
    return ValueError(
        f"{what} of {name} does not match {reference}. Rows are paired by "
        "position, so an input with labels (pandas or xarray) must use the "
        f"same labels in the same order as {reference}. Either reindex {name} "
        f"to the index of {reference}, or call .to_numpy() on it to pair the "
        "values by position."
    )


def check_row_alignment(
    name: str, labels: Index, expected: Index, reference: str = "dependent"
) -> None:
    """
    Reject labelled inputs whose row index is not the index of the reference.

    Estimation pairs rows by position. An input with the same values in a
    different order would attach those values to the wrong observation.

    Inputs with a different number of rows are not checked here since these
    are reported using a length-specific error message. A default index has
    no labels and is paired by position.

    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``,
    are the same labels. See ``same_labels``.

    Parameters
    ----------
    name : str
        Description of the input, which is used in the message.
    labels : Index
        Labels of the rows of the input.
    expected : Index
        Labels of the rows of the reference.
    reference : str, optional
        Description of the input that sets the labels. The default is
        "dependent".

    Raises
    ------
    ValueError
        If the labels differ from the expected labels.
    """
    if (
        len(labels) != len(expected)
        or is_default_index(labels)
        or same_labels(labels, expected)
    ):
        return
    raise misaligned(name, "The row index", reference)


def reference_labels(
    expected: Index | None,
    inputs: Mapping[str, object] | Iterable[tuple[str, object]],
    *,
    reference: str = "dependent",
) -> tuple[Index | None, str]:
    """
    The labels that the inputs have to use, and the input that they belong to.

    The dependent variable sets the labels if it carries labels. If it does not,
    the first of the inputs that carries labels sets them, so that two inputs
    that carry labels cannot be attached to the wrong observations either.

    Parameters
    ----------
    expected : Index or None
        Labels of the rows of the dependent variable. None if the dependent
        variable does not carry labels.
    inputs : dict of str to object or iterable of tuple of (str, object)
        The inputs in order, keyed by the name to use in the message. Inputs
        that are None or do not carry labels are not considered. The same name
        can be used more than once if the inputs are given as pairs.
    reference : str, optional
        Description of the dependent variable. The default is "dependent".

    Returns
    -------
    Index or None
        The labels of the rows. None if neither the dependent variable nor any
        of the inputs carries labels.
    str
        Description of the input that sets the labels, which is the name of the
        first input with labels when the dependent variable does not carry
        labels.
    """
    if expected is not None:
        return expected, reference
    items = inputs.items() if isinstance(inputs, Mapping) else inputs
    for name, value in items:
        labels = row_labels(value)
        if labels is not None:
            return labels, name
    return None, reference


def check_alignment(
    expected: Index | None,
    inputs: Mapping[str, object] | Iterable[tuple[str, object]],
    *,
    reference: str = "dependent",
) -> None:
    """
    Reject labelled inputs whose row index is not the expected index.

    Dates and time deltas are compared by value, so that the same dates at
    different resolutions are the same labels. See ``same_labels``.

    Parameters
    ----------
    expected : Index or None
        Labels of the rows of the reference, usually dependent. If None, the
        reference does not carry labels, and the first input that carries
        labels is the reference instead.
    inputs : dict of str to object or iterable of tuple of (str, object)
        The inputs to check, keyed by the name to use in the message, or pairs
        of the name and the input, in the order of the arguments. Inputs that
        are None or do not carry labels are not checked.
    reference : str, optional
        Description of the input that sets the labels. The default is
        "dependent".

    Raises
    ------
    ValueError
        If the labels of an input differ from the expected labels.

    See Also
    --------
    row_labels : Row labels of an input, if it carries labels.
    reference_labels : The labels that the inputs have to use.
    """
    items = list(inputs.items() if isinstance(inputs, Mapping) else inputs)
    expected, reference = reference_labels(expected, items, reference=reference)
    if expected is None:
        return
    for name, value in items:
        labels = row_labels(value)
        if labels is not None:
            check_row_alignment(name, labels, expected, reference)
