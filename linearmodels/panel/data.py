"""Data container that handles the formats of panel data."""

from __future__ import annotations

from linearmodels.compat.pandas import PD_GTE_21

from collections.abc import Sequence
from itertools import product
from typing import Literal, Union, cast, overload

import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    get_dummies,
)
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

from linearmodels.shared.exceptions import lower_option, unknown_option_error
from linearmodels.shared.inputs import as_float64, reject_complex
from linearmodels.shared.utility import ensure_unique_column, panel_to_frame
import linearmodels.typing.data

__all__ = ["PanelArrayLike", "PanelData", "PanelDataLike"]


class _Panel:
    """
    Convert a MI DataFrame to a 3-d structure where columns are items.

    This class is for internal use only and is a legacy shim related to the removed
    pandas Panel class.

    Parameters
    ----------
    df : DataFrame
        MultiIndex DataFrame containing floats.

    Notes
    -----
    Contains the logic needed to transform a MI DataFrame with 2 levels
    into a minimal pandas Panel-like object
    """

    def __init__(self, df: DataFrame):
        self._items = df.columns
        index = df.index
        assert isinstance(index, MultiIndex)
        # Time is required to be numeric or datetime. Order of first appearance
        # is not that order: a late-entering first entity puts later dates first,
        # and first_difference then subtracts non-adjacent periods.
        self._major_axis = Index(index.levels[1][index.codes[1]]).unique().sort_values()
        self._minor_axis = Index(index.levels[0][index.codes[0]]).unique()
        self._full_index = MultiIndex.from_product([self._minor_axis, self._major_axis])
        new_df = df.reindex(self._full_index)
        new_df.index.names = df.index.names
        self._frame = new_df
        i, j, k = len(self._items), len(self._major_axis), len(self.minor_axis)
        self._shape = (i, j, k)
        arr = np.asarray(new_df).copy().T.astype(np.float64)
        self._values = np.swapaxes(np.reshape(arr, (i, k, j)), 1, 2)

    @classmethod
    def from_array(
        cls,
        values: linearmodels.typing.NumericArray,
        items: Sequence[linearmodels.typing.Label],
        major_axis: Sequence[linearmodels.typing.Label],
        minor_axis: Sequence[linearmodels.typing.Label],
    ) -> _Panel:
        """
        Construct a panel from a 3-d array.

        Parameters
        ----------
        values : ndarray
            Array with shape (nitems, nmajor, nminor).
        items : sequence of Label
            The names of the items, which are the variables.
        major_axis : sequence of Label
            The labels of the major axis, which is time.
        minor_axis : sequence of Label
            The labels of the minor axis, which is the entity.

        Returns
        -------
        _Panel
            The panel with the data of the array.
        """
        index = list(product(minor_axis, major_axis))
        multi_index = MultiIndex.from_tuples(index)
        i, j, k = len(items), len(major_axis), len(minor_axis)
        values_flat: np.ndarray = np.swapaxes(values.copy(), 0, 2).ravel()
        values_flat = np.reshape(values_flat, ((j * k), i))
        # TODO: Remove Index when pandas-stubs is fixed
        df = DataFrame(values_flat, index=multi_index, columns=Index(items))
        return cls(df)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the panel, (nitems, nmajor, nminor)."""
        return self._shape

    @property
    def items(self) -> Index:
        """Names of the items, which are the variables."""
        return self._items

    @property
    def major_axis(self) -> Index:
        """Labels of the major axis, which is time."""
        return self._major_axis

    @property
    def minor_axis(self) -> Index:
        """Labels of the minor axis, which is the entity."""
        return self._minor_axis

    @property
    def values(self) -> linearmodels.typing.data.Float64Array:
        """Values of the panel with shape (nitems, nmajor, nminor)."""
        return self._values

    def to_frame(self) -> DataFrame:
        """
        Return the data as a DataFrame with a 2-level MultiIndex.

        Returns
        -------
        DataFrame
            The data with the entity and the time as the index levels, which
            has a row for each combination of an entity and a time period.
        """
        return self._frame


def convert_columns(s: Series, drop_first: bool) -> linearmodels.typing.AnyPandas:
    """
    Convert a string or categorical Series to dummy variables.

    Parameters
    ----------
    s : Series
        The data of one variable.
    drop_first : bool
        Flag indicating whether to drop the dummy of the first category.

    Returns
    -------
    Series or DataFrame
        A DataFrame of dummy variables, named ``name.category``, if s contains
        strings or is categorical. Otherwise, s.
    """
    if is_string_dtype(s.dtype) and s.map(lambda v: isinstance(v, str)).all():
        s = s.astype("category")

    if isinstance(s.dtype, pd.CategoricalDtype):
        out = get_dummies(s, drop_first=drop_first)
        # TODO: Remove once pandas typing fixed
        out.columns = Index([str(s.name) + "." + str(c) for c in out])
        return out
    return s


def expand_categoricals(x: DataFrame, drop_first: bool) -> DataFrame:
    """
    Convert the string and categorical columns of a DataFrame to dummy variables.

    Parameters
    ----------
    x : DataFrame
        The data.
    drop_first : bool
        Flag indicating whether to drop the dummy of the first category of each
        variable.

    Returns
    -------
    DataFrame
        The data with dummy variables in place of the string and categorical
        columns.
    """
    if x.shape[1] == 0:
        # Nothing to convert, and concat needs an object
        return x
    return concat(
        [convert_columns(x[c], drop_first) for c in x.columns], axis=1, sort=False
    )


class PanelData:
    """
    Abstraction to handle alternative formats for panel data.

    Parameters
    ----------
    x : DataFrame, Series, ndarray, DataArray or PanelData
        Input data. A list or a tuple raises a ``TypeError``, since the layout
        of the data would not be known. A ``TypeError`` is also raised if the
        data are complex or cannot be converted to float64.
    var_name : str, optional
        Variable name to use when naming variables in NumPy arrays or
        xarray DataArrays. The default is ``"x"``.
    convert_dummies : bool, optional
        Flag indicating whether pandas categoricals or string input data
        should be converted to dummy variables. The default is True.
    drop_first : bool, optional
        Flag indicating to drop first dummy category when converting. The
        default is True.
    copy : bool, optional
        Flag indicating whether to copy the input. Only has an effect when
        x is a DataFrame with a MultiIndex. The default is True.

    Raises
    ------
    TypeError
        If the input type is not supported, or if it is complex or cannot be
        converted to float64.
    ValueError
        If the input has the wrong number of dimensions or a MultiIndex
        DataFrame does not have 2 levels, if the time labels are neither
        numbers nor dates, or if a combination of an entity and a time
        period appears more than once.

    Notes
    -----
    Data can be either 2- or 3-dimensional. The three key dimensions are

    * nvar - number of variables
    * nobs - number of time periods
    * nentity - number of entities

    All 3-d inputs should be in the form (nvar, nobs, nentity). With one
    exception, 2-d inputs are treated as (nobs, nentity) so that the input
    can be treated as-if being (1, nobs, nentity).

    If the 2-d input is a pandas DataFrame with a 2-level MultiIndex then the
    input is treated differently.  Index level 0 is assumed to be entity.
    Index level 1 is time.  The columns are the variables.  MultiIndex Series
    are also accepted and treated as single column MultiIndex DataFrames.

    The data can have no variables, a DataFrame with a MultiIndex and no columns
    or an array with 0 variables. It is the ``exog`` of a
    :class:`~linearmodels.panel.model.PanelOLS` model that only has effects. The
    means, counts and demeaned data have no variables either.

    Missing values (NaN) in the entity or time index are not supported. The
    methods that group by the index raise a ValueError if the level that they
    use has a missing label.

    The time labels must be numbers or dates. A pandas ``PeriodIndex`` is
    accepted: it is converted with ``to_timestamp()``, so that the time labels
    are the timestamps at the start of the periods, and the data are the same
    as those of a panel that is indexed by timestamps.

    Each combination of an entity and a time period can appear only once, since
    an observation is identified by them.

    .. versionchanged:: 7.1

       Duplicated (entity, time) rows raise a ``ValueError``. Earlier versions
       accepted them, except for the first difference estimator, which raised an
       error from pandas, and the other estimators treated them as separate
       observations. A ``PeriodIndex`` of time labels is accepted. Earlier
       versions raised a ``ValueError`` that the time labels must be numeric or
       date-like.
    """

    def __init__(
        self,
        x: PanelDataLike,
        var_name: str = "x",
        convert_dummies: bool = True,
        drop_first: bool = True,
        copy: bool = True,
    ):
        self._var_name = var_name
        self._convert_dummies = convert_dummies
        self._drop_first = drop_first
        self._panel: _Panel | None = None
        self._shape: tuple[int, int, int] | None = None
        index_names = ["entity", "time"]
        if isinstance(x, PanelData):
            x = x.dataframe
        self._original = x
        # Complex values would be converted to float64 without their imaginary part
        reject_complex(x, var_name)

        if not isinstance(x, (Series, DataFrame, np.ndarray)):
            try:
                from xarray import DataArray  # noqa: PLC0415

                if isinstance(x, DataArray):
                    if x.ndim not in (2, 3):
                        raise ValueError("Only 2-d or 3-d DataArrays are supported")
                    if x.ndim == 2:
                        x = x.to_pandas()
                    else:
                        # The indexes keep the type of the coordinates. tolist() turns
                        # datetime64[ns] values into integers since a datetime cannot
                        # hold nanoseconds.
                        items = x.get_index(x.dims[0]).rename(None)
                        major = x.get_index(x.dims[1]).rename(None)
                        minor = x.get_index(x.dims[2]).rename(None)
                        values = x.values
                        x = panel_to_frame(values, items, major, minor, True)
            except ImportError:
                pass

        if isinstance(x, Series) and isinstance(x.index, MultiIndex):
            x = DataFrame(x)
        elif isinstance(x, Series):
            raise ValueError("Series can only be used with a 2-level MultiIndex")

        if isinstance(x, DataFrame):
            if isinstance(x.index, MultiIndex):
                if len(x.index.levels) != 2:
                    raise ValueError(
                        "DataFrame input must have a MultiIndex with 2 levels"
                    )
                if isinstance(self._original, (DataFrame, PanelData, Series)):
                    for i in range(2):
                        index_names[i] = x.index.levels[i].name or index_names[i]
                self._frame = x
                if copy:
                    self._frame = self._frame.copy()
            else:
                if not x.index.is_unique:
                    label = x.index[int(np.argmax(np.asarray(x.index.duplicated())))]
                    raise ValueError(
                        "Each combination of an entity and a time period can appear "
                        f"only once, but the time label {label} appears more than "
                        "once in the index of a DataFrame with time as the index and "
                        "entities as the columns, so that each entity has more than "
                        "one observation at that time."
                    )
                options = {"future_stack": True} if PD_GTE_21 else {"dropna": False}
                self._frame = DataFrame({var_name: x.T.stack(**options)})
        elif isinstance(x, np.ndarray):
            if x.ndim not in (2, 3):
                raise ValueError(
                    "2 or 3-d array required for numpy input. The array has "
                    f"{x.ndim} dimension(s)."
                )
            if x.ndim == 2:
                x = x[None, :, :]

            k, t, n = x.shape
            # An array without variables has no names, and no digits to pad
            ndigits = int(np.log10(k) + 0.01) if k > 0 else 0
            var_str = var_name + ".{0:0>" + str(ndigits) + "}"
            variables = [var_name] if k == 1 else [var_str.format(i) for i in range(k)]
            entity_str = "entity.{0:0>" + str(int(np.log10(n) + 0.01)) + "}"
            entities = [entity_str.format(i) for i in range(n)]
            time = list(range(t))
            assert isinstance(x, np.ndarray)
            if x.dtype != np.float64:
                x = as_float64(x, var_name)
            panel = _Panel.from_array(
                x, items=variables, major_axis=time, minor_axis=entities
            )
            self._fake_panel = panel
            self._frame = panel.to_frame()
        else:
            raise TypeError(
                "Only ndarrays, DataFrames or DataArrays are supported. A DataFrame "
                "or Series needs a MultiIndex with 2 levels (entity and time), or "
                "time as the index and entities as the columns if it is a DataFrame. "
                f"Got {type(x).__name__}."
            )
        if convert_dummies:
            self._frame = expand_categoricals(self._frame, drop_first)
            try:
                self._frame = self._frame.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{var_name} must contain numbers, strings or categories, but "
                    f"it cannot be converted to float64: {exc}"
                ) from exc

        time_level = self.index.levels[1]
        if isinstance(time_level, pd.PeriodIndex):
            # A period is identified by its start, as a date is in a DatetimeIndex. The
            # frame is not changed in place since it can be the frame of the caller.
            timestamps = self.index.set_levels(time_level.to_timestamp(), level=1)
            self._frame = self._frame.set_axis(timestamps, axis=0)
        time_index = Series(self.index.levels[1])
        if not (
            is_numeric_dtype(time_index.dtype)
            or is_datetime64_any_dtype(time_index.dtype)
        ):
            raise ValueError(
                "The index on the time dimension must be either numeric or date-like"
            )
        self._check_unique(index_names)
        # self._k, self._t, self._n = self.panel.shape
        self._k, self._t, self._n = self.shape
        self._frame.index = self._frame.index.set_names(index_names)

    def _check_unique(self, index_names: Sequence[str]) -> None:
        """
        Raise a ValueError if an (entity, time) combination appears more than once

        Parameters
        ----------
        index_names : sequence of str
            The names of the entity and time levels, which are used in the message.

        Raises
        ------
        ValueError
            If the index has duplicated entries. The message names the first of
            them, the first row that repeats an earlier row.
        """
        index = self.index
        if index.is_unique:
            return
        duplicated = np.asarray(index.duplicated())
        first = index[int(np.argmax(duplicated))]
        entity, time = index_names
        raise ValueError(
            "Each combination of an entity and a time period can appear only once, "
            f"but {int(duplicated.sum())} row(s) repeat an earlier row. The first "
            f"duplicated entry is {entity}={first[0]}, {time}={first[1]}."
        )

    @property
    def panel(self) -> _Panel:
        """Panel view of the data, a legacy shim of the pandas Panel."""
        if self._panel is None:
            self._check_index(0, 1)
            self._panel = _Panel(self._frame)
        assert self._panel is not None
        return self._panel

    @property
    def dataframe(self) -> DataFrame:
        """DataFrame view of the data."""
        return self._frame

    @property
    def values2d(self) -> linearmodels.typing.data.AnyArray:
        """NumPy ndarray view of the DataFrame."""
        return np.require(self._frame, requirements="W")

    @property
    def values3d(self) -> linearmodels.typing.data.AnyArray:
        """NumPy ndarray view of the panel with shape (nvar, nobs, nentity)."""
        return self.panel.values

    def drop(self, locs: pd.Series | linearmodels.typing.data.BoolArray) -> None:
        """
        Drop observations from the panel.

        Parameters
        ----------
        locs : Series or ndarray
            Boolean array indicating observations to drop with reference to
            the dataframe view of the data.
        """
        if isinstance(locs, Series):
            locs = np.asarray(locs)
        self._frame = self._frame.loc[~locs.ravel()]
        self._frame = self._minimize_multiindex(self._frame)
        # Reset panel and shape after a drop
        self._panel = self._shape = None
        self._k, self._t, self._n = self.shape

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the panel view of the data, (nvar, nobs, nentity)."""
        if self._shape is None:
            k = self._frame.shape[1]
            index: Index = self._frame.index
            t = index.get_level_values(1).unique().shape[0]
            n = index.get_level_values(0).unique().shape[0]
            self._shape = k, t, n
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the panel view of the data."""
        return 3

    @property
    def isnull(self) -> Series:
        """Locations with missing observations."""
        return self._frame.isnull().any(axis=1)

    @property
    def nobs(self) -> int:
        """Number of time observations."""
        return self._t

    @property
    def nvar(self) -> int:
        """Number of variables."""
        return self._k

    @property
    def nentity(self) -> int:
        """Number of entities."""
        return self._n

    @property
    def vars(self) -> list[linearmodels.typing.Label]:
        """List of variable names."""
        return list(self._frame.columns)

    @property
    def time(self) -> list[linearmodels.typing.Label]:
        """List of time index labels."""
        index = self.index
        return list(index.levels[1][index.codes[1]].unique())

    @property
    def entities(self) -> list[linearmodels.typing.Label]:
        """List of entity index labels."""
        index = self.index
        return list(index.levels[0][index.codes[0]].unique())

    def _check_index(self, *levels: int) -> None:
        """
        Raise a ValueError if a level of the index has missing values (NaN)

        Parameters
        ----------
        *levels : int
            Levels to check, 0 for the entity and 1 for the time

        Notes
        -----
        pandas encodes a missing label as -1 in the codes of the level. The
        operations that group by a level have to use this to reject it since
        a groupby drops the rows that have a missing label, and the codes
        select the last label.
        """
        for level in levels:
            if np.any(self.index.codes[level] < 0):
                name = "entity" if level == 0 else "time"
                raise ValueError(
                    f"The {name} index contains missing values (NaN), which are not "
                    f"supported and are encoded as -1 in the {name} ids."
                )

    @property
    def entity_ids(self) -> linearmodels.typing.data.AnyIntArray:
        """
        Get array containing entity group membership information.

        Returns
        -------
        ndarray
            2d array containing entity ids corresponding to the dataframe view.
        """
        self._check_index(0)
        return np.asarray(self.index.codes[0])[:, None]

    @property
    def time_ids(self) -> linearmodels.typing.data.AnyIntArray:
        """
        Get array containing time membership information.

        Returns
        -------
        ndarray
            2d array containing time ids corresponding to the dataframe view.
        """
        self._check_index(1)
        return np.asarray(self.index.codes[1])[:, None]

    def _demean_both_low_mem(self, weights: PanelData | None) -> PanelData:
        groups = PanelData(
            DataFrame(np.c_[self.entity_ids, self.time_ids], index=self._frame.index),
            convert_dummies=False,
            copy=False,
        )
        return self.general_demean(groups, weights=weights)

    def _demean_both(self, weights: PanelData | None) -> PanelData:
        """
        Entity and time demean.

        Parameters
        ----------
        weights : PanelData or None
            Weights to use in demeaning.

        Returns
        -------
        PanelData
            The data demeaned by entity and by time.
        """
        group: Literal["entity", "time"]
        if self.nentity > self.nobs:
            group = "entity"
            dummy = "time"
        else:
            group = "time"
            dummy = "entity"
        e = self.demean(group, weights=weights)
        assert isinstance(e, PanelData)
        d = self.dummies(dummy, drop_first=True)
        if d.shape[1] == 0:
            # A single entity or period: there are no dummy variables to remove
            return e
        d.index = e.index
        d_pd = PanelData(d).demean(group, weights=weights)
        d_arr = d_pd.values2d
        e_arr = e.values2d
        resid_arr = e_arr - d_arr @ lstsq(d_arr, e_arr, rcond=None)[0]
        resid = DataFrame(
            resid_arr, index=self._frame.index, columns=self._frame.columns
        )

        return PanelData(resid)

    def general_demean(
        self, groups: PanelDataLike, weights: PanelData | None = None
    ) -> PanelData:
        """
        Multi-way demeaning using only groupby.

        Parameters
        ----------
        groups : PanelData or array_like
            Arrays with the same size containing group identifiers. Input that
            is not a PanelData is converted with ``PanelData``.
        weights : PanelData, optional
            Weights to use in the weighted demeaning. If not provided, the
            observations have equal weights.

        Returns
        -------
        PanelData
            Weighted, demeaned data according to groups.

        Notes
        -----
        Iterates until convergence.
        """
        if not isinstance(groups, PanelData):
            groups = PanelData(groups)
        if self._frame.shape[1] == 0:
            # There are no variables to demean
            return PanelData(self._frame.copy())
        if weights is None:
            weights = PanelData(
                DataFrame(
                    np.ones((self._frame.shape[0], 1)),
                    index=self.index,
                    columns=["weights"],
                )
            )
        groups = groups.values2d.astype(np.int64, copy=False)

        weight_sum: dict[int, Series | DataFrame] = {}

        def weighted_group_mean(
            df: pd.DataFrame,
            weights: pd.DataFrame,
            root_w: linearmodels.typing.data.Float64Array,
            level: int,
        ) -> linearmodels.typing.data.Float64Array:
            scaled_df = cast("DataFrame", root_w * df)
            num = scaled_df.groupby(level=level).transform("sum")
            if level in weight_sum:
                denom = np.asarray(weight_sum[level])
            else:
                denom_df = weights.groupby(level=level).transform("sum")
                weight_sum[level] = denom_df
                denom = np.asarray(denom_df)
            return np.asarray(num) / denom

        def demean_pass(
            frame: pd.DataFrame,
            weights: pd.DataFrame,
            root_w: linearmodels.typing.data.Float64Array,
        ) -> DataFrame:
            levels = groups.shape[1]
            for level in range(levels):
                mu = weighted_group_mean(frame, weights, root_w, level)
                if level == 0:
                    frame = frame - root_w * mu
                else:
                    frame -= root_w * mu

            return frame

        # Swap out the index for better performance
        init_index = DataFrame(groups)
        init_index = init_index.set_index(list(init_index.columns))

        root_w = cast(
            "linearmodels.typing.data.Float64Array", np.sqrt(weights.values2d)
        )
        weights_df = DataFrame(weights.values2d, index=init_index.index)
        wframe = cast("DataFrame", root_w * self._frame)
        wframe.index = init_index.index

        previous = wframe
        current: pd.DataFrame = demean_pass(previous, weights_df, root_w)
        if groups.shape[1] == 1:
            current.index = self._frame.index
            return PanelData(current)

        exclude = np.ptp(np.asarray(self._frame), 0) == 0
        max_rmse = np.sqrt(np.asarray(self._frame).var(0).max())
        scale = np.require(self._frame.std(), requirements="W")
        exclude = exclude | (scale < 1e-14 * max_rmse)
        replacement = cast(
            "linearmodels.typing.data.Float64Array", np.maximum(scale, 1)
        )
        scale[exclude] = replacement[exclude]
        scale = scale[None, :]

        while np.max(np.abs(np.asarray(current) - np.asarray(previous)) / scale) > 1e-8:
            previous = current
            current = demean_pass(previous, weights_df, root_w)
        current.index = self._frame.index

        return PanelData(current)

    @overload
    def demean(  # numpydoc ignore=GL08
        self,
        group: Literal["entity", "time", "both"],
        *,
        return_panel: Literal[False],
    ) -> linearmodels.typing.data.Float64Array: ...

    @overload
    def demean(  # numpydoc ignore=GL08
        self,
        group: Literal["entity", "time", "both"] = ...,
        weights: PanelData | None = ...,
        return_panel: Literal[True] = ...,
        low_memory: bool = ...,
    ) -> PanelData: ...

    @overload
    def demean(  # numpydoc ignore=GL08
        self,
        group: Literal["entity", "time", "both"],
        weights: PanelData | None,
        return_panel: Literal[False],
    ) -> linearmodels.typing.data.Float64Array: ...

    def demean(
        self,
        group: Literal["entity", "time", "both"] = "entity",
        weights: PanelData | None = None,
        return_panel: bool = True,
        low_memory: bool = False,
    ) -> PanelData | linearmodels.typing.data.Float64Array:
        """
        Demean data by either entity or time group.

        Parameters
        ----------
        group : {"entity", "time", "both"}, optional
            Group to use in demeaning. The default is "entity". The names are not
            case sensitive.
        weights : PanelData, optional
            Weights to implement weighted averaging. If not provided, the
            observations have equal weights.
        return_panel : bool, optional
            Flag indicating to return a PanelData object. If False, a 2-d
            NumPy representation of the panel is returned. The default is
            True.
        low_memory : bool, optional
            Flag indicating whether to use a low memory implementation
            that avoids constructing dummy variables. Only relevant when
            group is "both". The default is False.

        Returns
        -------
        PanelData or ndarray
            Demeaned data according to type. An ndarray if `return_panel` is
            False.

        Raises
        ------
        ValueError
            If `group` is not one of the supported values, or if the level of
            the index that is used has a missing label.

        Notes
        -----
        If weights are provided, the values returned will be scaled by
        the square root of the weights so that they can be used in WLS
        estimation.
        """
        group = lower_option(group)
        if group not in ("entity", "time", "both"):
            raise unknown_option_error("group", group, ("entity", "time", "both"))
        self._check_index(*{"entity": (0,), "time": (1,), "both": (0, 1)}[group])
        if group == "both":
            if not low_memory:
                return self._demean_both(weights)
            else:
                return self._demean_both_low_mem(weights)

        level = 0 if group == "entity" else 1
        if weights is None or self._frame.shape[1] == 0:
            # The weights have no variables to weight if the frame has no columns
            group_mu = self._frame.groupby(level=level).transform("mean")
            out = self._frame - group_mu
            if not return_panel:
                return np.asarray(out)
            return PanelData(out)
        else:
            w = weights.values2d
            frame: DataFrame = self._frame.copy()
            frame = cast("DataFrame", w * frame)
            weighted_sum: DataFrame = frame.groupby(level=level).transform("sum")
            frame.iloc[:, :] = w
            sum_weights: DataFrame = frame.groupby(level=level).transform("sum")
            group_mu = weighted_sum / sum_weights
            out_df = np.sqrt(w) * (self._frame - group_mu)
            if not return_panel:
                return np.asarray(out_df)
            return PanelData(out_df)

    def __str__(self) -> str:
        return self.__class__.__name__ + "\n" + str(self._frame)

    def __repr__(self) -> str:
        return (
            self.__str__()
            + "\n"
            + self.__class__.__name__
            + " object, id: "
            + hex(id(self))
        )

    def _repr_html_(self) -> str:
        html = self._frame._repr_html_()  # type: ignore[operator]
        return self.__class__.__name__ + "<br/>" + html

    def count(self, group: str = "entity") -> DataFrame:
        """
        Count number of observations by entity or time.

        Parameters
        ----------
        group : {"entity", "time"}, optional
            Group to count. The default is "entity". The names are not case
            sensitive.

        Returns
        -------
        DataFrame
            Counts according to type. Either (entity by var) or (time by var).

        Raises
        ------
        UnknownOptionError
            If `group` is not one of the supported values. It is also a
            ``ValueError``.
        ValueError
            If the level of the index that is used has a missing label.

        Notes
        -----
        .. versionchanged:: 7.1

           An unknown `group` raises an error. It was treated as ``"time"``.
        """
        group = lower_option(group)
        if group not in ("entity", "time"):
            raise unknown_option_error("group", group, ("entity", "time"))
        level = 0 if group == "entity" else 1
        self._check_index(level)
        reindex = self.entities if group == "entity" else self.time
        out = self._frame.groupby(level=level).count()

        return out.reindex(reindex)

    @property
    def index(self) -> MultiIndex:
        """Return the index of the multi-index dataframe view."""
        index = self._frame.index
        assert isinstance(index, MultiIndex)
        return index

    def copy(self) -> PanelData:
        """
        Return a deep copy.

        Returns
        -------
        PanelData
            A copy of the data with the same settings.
        """
        return PanelData(
            self._frame.copy(),
            var_name=self._var_name,
            convert_dummies=self._convert_dummies,
            drop_first=self._drop_first,
        )

    def mean(
        self, group: str = "entity", weights: PanelData | None = None
    ) -> pd.DataFrame:
        """
        Compute data mean by either entity or time group.

        Parameters
        ----------
        group : {"entity", "time"}, optional
            Group to use in computing the means. The default is "entity". The names
            are not case sensitive.
        weights : PanelData, optional
            Weights to implement weighted averaging. If not provided, the
            observations have equal weights.

        Returns
        -------
        DataFrame
            Data mean according to type. Either (entity by var) or (time by var).

        Raises
        ------
        UnknownOptionError
            If `group` is not one of the supported values. It is also a
            ``ValueError``.
        ValueError
            If the level of the index that is used has a missing label.

        Notes
        -----
        .. versionchanged:: 7.1

           An unknown `group` raises an error. It was treated as ``"time"``.
        """
        group = lower_option(group)
        if group not in ("entity", "time"):
            raise unknown_option_error("group", group, ("entity", "time"))
        level = 0 if group == "entity" else 1
        self._check_index(level)
        if weights is None or self._frame.shape[1] == 0:
            # There are no variables to weight if the frame has no columns
            mu = self._frame.groupby(level=level).mean()
        else:
            w = weights.values2d
            frame = self._frame.copy()
            frame = cast("DataFrame", w * frame)
            weighted_sum = frame.groupby(level=level).sum()
            frame.iloc[:, :] = w
            sum_weights = frame.groupby(level=level).sum()
            mu = weighted_sum / sum_weights

        reindex = self.entities if group == "entity" else self.time
        out = mu.reindex(reindex)

        return out

    def first_difference(self) -> PanelData:
        """
        Compute first differences of variables.

        Returns
        -------
        PanelData
            Differenced values.

        Notes
        -----
        An observation is differenced with the observation of the same entity in
        the preceding time label of the panel, which is the preceding one of the
        distinct time labels that occur for any entity. If the entity has no
        observation in the preceding label, there is no difference for the
        observation. This is the case for the first observation of an entity and
        for an observation that follows a gap. The differences are never taken
        across a gap, and so they are all differences between adjacent periods.
        """
        diffs = self.panel.values
        diffs = diffs[:, 1:] - diffs[:, :-1]
        diffs_frame = panel_to_frame(
            diffs,
            self.panel.items,
            self.panel.major_axis[1:],
            self.panel.minor_axis,
            True,
        )
        diffs_frame = diffs_frame.reindex(self._frame.index).dropna(how="any")
        return PanelData(diffs_frame)

    @staticmethod
    def _minimize_multiindex(df: pd.DataFrame) -> DataFrame:
        index_cols = list(df.index.names)
        orig_names = index_cols[:]
        for i, _col in enumerate(index_cols):
            col = ensure_unique_column(_col, df)
            index_cols[i] = col
        df.index.names = index_cols
        df = df.reset_index()
        df = df.set_index(index_cols)
        df.index.names = orig_names
        return df

    def dummies(self, group: str = "entity", drop_first: bool = False) -> DataFrame:
        """
        Generate entity or time dummies.

        Parameters
        ----------
        group : {"entity", "time"}, optional
            Type of dummies to generate. The default is "entity". The names are not
            case sensitive.
        drop_first : bool, optional
            Flag indicating that the dummy column corresponding to the first
            entity or time period should be dropped. The default is False.

        Returns
        -------
        DataFrame
            Dummy variables.

        Raises
        ------
        ValueError
            If `group` is not one of the supported values, or if the level of
            the index that is used has a missing label.
        """
        group = lower_option(group)
        if group not in ("entity", "time"):
            raise unknown_option_error("group", group, ("entity", "time"))
        axis = 0 if group == "entity" else 1
        self._check_index(axis)
        labels = self.index.codes
        levels = self.index.levels
        cat = Categorical(levels[axis][labels[axis]])
        dummies = get_dummies(cat, drop_first=drop_first)
        cols = self.entities if group == "entity" else self.time
        # TODO: Incorrect typing in pandas-stubs not handling Hashable | None
        dummy_cols = [c for c in cols if c in dummies]
        return dummies[dummy_cols].astype(np.float64)


# The inputs that PanelData accepts. A list or a tuple does not have a layout.
PanelDataLike = Union[PanelData, linearmodels.typing.data.DataContainer]
# The inputs that are array_like, which also include a list or a tuple
PanelArrayLike = Union[PanelData, linearmodels.typing.data.ArrayLike]
