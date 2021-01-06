from itertools import product
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.linalg import lstsq
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    get_dummies,
)
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from linearmodels.shared.utility import ensure_unique_column, panel_to_frame
from linearmodels.typing import AnyPandas, ArrayLike, Label, NDArray

__all__ = ["PanelData", "PanelDataLike"]


class _Panel(object):
    """
    Convert a MI DataFrame to a 3-d structure where columns are items

    Parameters
    ----------
    df : DataFrame
        MultiIndex DataFrame containing floats

    Notes
    -----
    Contains the logic needed to transform a MI DataFrame with 2 levels
    into a minimal pandas Panel-like object
    """

    def __init__(self, df: DataFrame):
        self._items = df.columns
        index = df.index
        assert isinstance(index, MultiIndex)
        self._major_axis = Index(index.levels[1][index.codes[1]]).unique()
        self._minor_axis = Index(index.levels[0][index.codes[0]]).unique()
        self._full_index = MultiIndex.from_product([self._minor_axis, self._major_axis])
        new_df = df.reindex(self._full_index)
        new_df.index.names = df.index.names
        self._frame = new_df
        i, j, k = len(self._items), len(self._major_axis), len(self.minor_axis)
        self._shape = (i, j, k)
        self._values = np.swapaxes(
            np.reshape(np.asarray(new_df).copy().T, (i, k, j)), 1, 2
        )

    @classmethod
    def from_array(
        cls,
        values: NDArray,
        items: Sequence[Label],
        major_axis: Sequence[Label],
        minor_axis: Sequence[Label],
    ) -> "_Panel":
        index = list(product(minor_axis, major_axis))
        index = MultiIndex.from_tuples(index)
        i, j, k = len(items), len(major_axis), len(minor_axis)
        values = np.swapaxes(values.copy(), 0, 2).ravel()
        values = np.reshape(values, ((j * k), i))

        df = DataFrame(values, index=index, columns=items)
        return cls(df)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def items(self) -> Index:
        return self._items

    @property
    def major_axis(self) -> Index:
        return self._major_axis

    @property
    def minor_axis(self) -> Index:
        return self._minor_axis

    @property
    def values(self) -> NDArray:
        return self._values

    def to_frame(self) -> DataFrame:
        return self._frame


def convert_columns(s: Series, drop_first: bool) -> AnyPandas:
    if is_string_dtype(s.dtype) and s.map(lambda v: isinstance(v, str)).all():
        s = s.astype("category")

    if is_categorical_dtype(s):
        out = get_dummies(s, drop_first=drop_first)
        out.columns = [str(s.name) + "." + str(c) for c in out]
        return out
    return s


def expand_categoricals(x: DataFrame, drop_first: bool) -> DataFrame:
    return concat(
        [convert_columns(x[c], drop_first) for c in x.columns], axis=1, sort=False
    )


class PanelData(object):
    """
    Abstraction to handle alternative formats for panel data

    Parameters
    ----------
    x : {ndarray, Series, DataFrame, DataArray}
       Input data
    var_name : str, optional
        Variable name to use when naming variables in NumPy arrays or
        xarray DataArrays
    convert_dummies : bool, optional
        Flat indicating whether pandas categoricals or string input data
        should be converted to dummy variables
    drop_first : bool, optional
        Flag indicating to drop first dummy category when converting
    copy: bool, optional
        Flag indicating whether to copy the input. Only has an effect when
        x is a DataFrame

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
    input is treated differently.  Index level 0 is assumed ot be entity.
    Index level 1 is time.  The columns are the variables.  MultiIndex Series
    are also accepted and treated as single column MultiIndex DataFrames.

    Raises
    ------
    TypeError
        If the input type is not supported
    ValueError
        If the input has the wrong number of dimensions or a MultiIndex
        DataFrame does not have 2 levels
    """

    def __init__(
        self,
        x: "PanelDataLike",
        var_name: str = "x",
        convert_dummies: bool = True,
        drop_first: bool = True,
        copy: bool = True,
    ):
        self._var_name = var_name
        self._convert_dummies = convert_dummies
        self._drop_first = drop_first
        self._panel: Optional[_Panel] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        index_names = ["entity", "time"]
        if isinstance(x, PanelData):
            x = x.dataframe
        self._original = x

        if not isinstance(x, (Series, DataFrame, np.ndarray)):
            try:
                from xarray import DataArray

                if isinstance(x, DataArray):
                    if x.ndim not in (2, 3):
                        raise ValueError("Only 2-d or 3-d DataArrays are supported")
                    if x.ndim == 2:
                        x = x.to_pandas()
                    else:
                        items: List[Hashable] = np.asarray(x.coords[x.dims[0]]).tolist()
                        major: List[Hashable] = np.asarray(x.coords[x.dims[1]]).tolist()
                        minor: List[Hashable] = np.asarray(x.coords[x.dims[2]]).tolist()
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
                        "DataFrame input must have a " "MultiIndex with 2 levels"
                    )
                if isinstance(self._original, (DataFrame, PanelData, Series)):
                    for i in range(2):
                        index_names[i] = x.index.levels[i].name or index_names[i]
                self._frame = x
                if copy:
                    self._frame = self._frame.copy()
            else:
                self._frame = DataFrame({var_name: x.T.stack(dropna=False)})
        elif isinstance(x, np.ndarray):
            if x.ndim not in (2, 3):
                raise ValueError("2 or 3-d array required for numpy input")
            if x.ndim == 2:
                x = x[None, :, :]

            k, t, n = x.shape
            var_str = var_name + ".{0:0>" + str(int(np.log10(k) + 0.01)) + "}"
            variables = [var_name] if k == 1 else [var_str.format(i) for i in range(k)]
            entity_str = "entity.{0:0>" + str(int(np.log10(n) + 0.01)) + "}"
            entities = [entity_str.format(i) for i in range(n)]
            time = list(range(t))
            assert isinstance(x, np.ndarray)
            x = x.astype(np.float64, copy=False)
            panel = _Panel.from_array(
                x, items=variables, major_axis=time, minor_axis=entities
            )
            self._fake_panel = panel
            self._frame = panel.to_frame()
        else:
            raise TypeError("Only ndarrays, DataFrames or DataArrays are " "supported")
        if convert_dummies:
            self._frame = expand_categoricals(self._frame, drop_first)
            self._frame = self._frame.astype(np.float64, copy=False)

        time_index = Series(self.index.levels[1])
        if not (
            is_numeric_dtype(time_index.dtype)
            or is_datetime64_any_dtype(time_index.dtype)
        ):
            raise ValueError(
                "The index on the time dimension must be either " "numeric or date-like"
            )
        # self._k, self._t, self._n = self.panel.shape
        self._k, self._t, self._n = self.shape
        self._frame.index.set_names(index_names, inplace=True)

    @property
    def panel(self) -> _Panel:
        """pandas Panel view of data"""
        if self._panel is None:
            self._panel = _Panel(self._frame)
        assert self._panel is not None
        return self._panel

    @property
    def dataframe(self) -> DataFrame:
        """pandas DataFrame view of data"""
        return self._frame

    @property
    def values2d(self) -> NDArray:
        """NumPy ndarray view of dataframe"""
        return np.asarray(self._frame)

    @property
    def values3d(self) -> NDArray:
        """NumPy ndarray view of panel"""
        return self.panel.values

    def drop(self, locs: Union[Series, NDArray]) -> None:
        """
        Drop observations from the panel.

        Parameters
        ----------
        locs : ndarray
            Boolean array indicating observations to drop with reference to
            the dataframe view of the data
        """
        self._frame = self._frame.loc[~locs.ravel()]
        self._frame = self._minimize_multiindex(self._frame)
        # Reset panel and shape after a drop
        self._panel = self._shape = None
        self._k, self._t, self._n = self.shape

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of panel view of data"""
        if self._shape is None:
            k = self._frame.shape[1]
            index: Index = self._frame.index
            t = index.get_level_values(1).unique().shape[0]
            n = index.get_level_values(0).unique().shape[0]
            self._shape = k, t, n
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of panel view of data"""
        return 3

    @property
    def isnull(self) -> Series:
        """Locations with missing observations"""
        return self._frame.isnull().any(axis=1)

    @property
    def nobs(self) -> int:
        """Number of time observations"""
        return self._t

    @property
    def nvar(self) -> int:
        """Number of variables"""
        return self._k

    @property
    def nentity(self) -> int:
        """Number of entities"""
        return self._n

    @property
    def vars(self) -> List[Label]:
        """List of variable names"""
        return list(self._frame.columns)

    @property
    def time(self) -> List[Label]:
        """List of time index names"""
        index = self.index
        return list(index.levels[1][index.codes[1]].unique())

    @property
    def entities(self) -> List[Label]:
        """List of entity index names"""
        index = self.index
        return list(index.levels[0][index.codes[0]].unique())

    @property
    def entity_ids(self) -> NDArray:
        """
        Get array containing entity group membership information

        Returns
        -------
        ndarray
            2d array containing entity ids corresponding dataframe view
        """
        index = self.index
        return np.asarray(index.codes[0])[:, None]

    @property
    def time_ids(self) -> NDArray:
        """
        Get array containing time membership information

        Returns
        -------
        ndarray
            2d array containing time ids corresponding dataframe view
        """
        index = self.index
        return np.asarray(index.codes[1])[:, None]

    def _demean_both_low_mem(self, weights: Optional["PanelData"]) -> "PanelData":
        groups = PanelData(
            DataFrame(np.c_[self.entity_ids, self.time_ids], index=self._frame.index),
            convert_dummies=False,
            copy=False,
        )
        return self.general_demean(groups, weights=weights)

    def _demean_both(self, weights: Optional["PanelData"]) -> "PanelData":
        """
        Entity and time demean

        Parameters
        ----------
        weights : PanelData, optional
             Weights to use in demeaning
        """
        if self.nentity > self.nobs:
            group = "entity"
            dummy = "time"
        else:
            group = "time"
            dummy = "entity"
        e = self.demean(group, weights=weights)
        assert isinstance(e, PanelData)
        d = self.dummies(dummy, drop_first=True)
        d.index = e.index
        d = PanelData(d).demean(group, weights=weights)
        d = d.values2d
        e = e.values2d
        resid = e - d @ lstsq(d, e, rcond=None)[0]
        resid = DataFrame(resid, index=self._frame.index, columns=self._frame.columns)

        return PanelData(resid)

    def general_demean(
        self, groups: "PanelDataLike", weights: Optional["PanelData"] = None
    ) -> "PanelData":
        """
        Multi-way demeaning using only groupby

        Parameters
        ----------
        groups : PanelData
            Arrays with the same size containing group identifiers
        weights : PanelData, optional
            Weights to use in the weighted demeaning

        Returns
        -------
        PanelData
            Weighted, demeaned data according to groups

        Notes
        -----
        Iterates until convergence
        """
        if not isinstance(groups, PanelData):
            groups = PanelData(groups)
        if weights is None:
            weights = PanelData(
                DataFrame(
                    np.ones((self._frame.shape[0], 1)),
                    index=self.index,
                    columns=["weights"],
                )
            )
        groups = groups.values2d.astype(np.int64, copy=False)

        weight_sum: Dict[int, Series] = {}

        def weighted_group_mean(
            df: DataFrame, weights: DataFrame, root_w: NDArray, level: int
        ) -> NDArray:
            num = (root_w * df).groupby(level=level).transform("sum")
            if level in weight_sum:
                denom = weight_sum[level]
            else:
                denom = weights.groupby(level=level).transform("sum")
                weight_sum[level] = denom
            return np.asarray(num) / np.asarray(denom)

        def demean_pass(
            frame: DataFrame, weights: DataFrame, root_w: NDArray
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
        init_index.set_index(list(init_index.columns), inplace=True)

        root_w = cast(NDArray, np.sqrt(weights.values2d))
        weights_df = DataFrame(weights.values2d, index=init_index.index)
        wframe = root_w * self._frame
        wframe.index = init_index.index

        previous = wframe
        current = demean_pass(previous, weights_df, root_w)
        if groups.shape[1] == 1:
            current.index = self._frame.index
            return PanelData(current)

        exclude = np.ptp(np.asarray(self._frame), 0) == 0
        max_rmse = np.sqrt(np.asarray(self._frame).var(0).max())
        scale = np.asarray(self._frame.std())
        exclude = exclude | (scale < 1e-14 * max_rmse)
        replacement = cast(NDArray, np.maximum(scale, 1))
        scale[exclude] = replacement[exclude]
        scale = scale[None, :]

        while np.max(np.abs(np.asarray(current) - np.asarray(previous)) / scale) > 1e-8:
            previous = current
            current = demean_pass(previous, weights_df, root_w)
        current.index = self._frame.index

        return PanelData(current)

    def demean(
        self,
        group: str = "entity",
        weights: Optional["PanelData"] = None,
        return_panel: bool = True,
        low_memory: bool = False,
    ) -> Union["PanelData", np.ndarray]:
        """
        Demeans data by either entity or time group

        Parameters
        ----------
        group : {'entity', 'time', 'both'}
            Group to use in demeaning
        weights : PanelData, optional
            Weights to implement weighted averaging
        return_panel : bool
            Flag indicating to return a PanelData object. If False, a 2-d
            NumPy representation of the panel is returned
        low_memory : bool
            Flag indicating whether to use a low memory implementation
            that avoids constructing dummy variables. Only relevant when
            group is 'both'

        Returns
        -------
        PanelData
            Demeaned data according to type

        Notes
        -----
        If weights are provided, the values returned will be scaled by
        the square root of the weights so that they can be used in WLS
        estimation.
        """
        if group not in ("entity", "time", "both"):
            raise ValueError
        if group == "both":
            if not low_memory:
                return self._demean_both(weights)
            else:
                return self._demean_both_low_mem(weights)

        level = 0 if group == "entity" else 1
        if weights is None:
            group_mu = self._frame.groupby(level=level).transform("mean")
            out = self._frame - group_mu
            if not return_panel:
                return np.asarray(out)
            return PanelData(out)
        else:
            w = weights.values2d
            frame = self._frame.copy()
            frame = w * frame
            weighted_sum = frame.groupby(level=level).transform("sum")
            frame.iloc[:, :] = w
            sum_weights = frame.groupby(level=level).transform("sum")
            group_mu = weighted_sum / sum_weights
            out = np.sqrt(w) * (self._frame - group_mu)
            if not return_panel:
                return np.asarray(out)
            return PanelData(out)

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
        return self.__class__.__name__ + "<br/>" + self._frame._repr_html_()

    def count(self, group: str = "entity") -> DataFrame:
        """
        Count number of observations by entity or time

        Parameters
        ----------
        group : {'entity', 'time'}
            Group to count

        Returns
        -------
        DataFrame
            Counts according to type. Either (entity by var) or (time by var)
        """
        level = 0 if group == "entity" else 1
        reindex = self.entities if group == "entity" else self.time
        out = self._frame.groupby(level=level).count()

        return out.reindex(reindex)

    @property
    def index(self) -> MultiIndex:
        """Return the index of the multi-index dataframe view"""
        index = self._frame.index
        assert isinstance(index, MultiIndex)
        return index

    def copy(self) -> "PanelData":
        """Return a deep copy"""
        return PanelData(
            self._frame.copy(),
            var_name=self._var_name,
            convert_dummies=self._convert_dummies,
            drop_first=self._drop_first,
        )

    def mean(
        self, group: str = "entity", weights: Optional["PanelData"] = None
    ) -> DataFrame:
        """
        Compute data mean by either entity or time group

        Parameters
        ----------
        group : {'entity', 'time'}
            Group to use in demeaning
        weights : PanelData, optional
            Weights to implement weighted averaging

        Returns
        -------
        DataFrame
            Data mean according to type. Either (entity by var) or (time by var)
        """
        level = 0 if group == "entity" else 1
        if weights is None:
            mu = self._frame.groupby(level=level).mean()
        else:
            w = weights.values2d
            frame = self._frame.copy()
            frame = w * frame
            weighted_sum = frame.groupby(level=level).sum()
            frame.iloc[:, :] = w
            sum_weights = frame.groupby(level=level).sum()
            mu = weighted_sum / sum_weights

        reindex = self.entities if group == "entity" else self.time
        out = mu.reindex(reindex)

        return out

    def first_difference(self) -> "PanelData":
        """
        Compute first differences of variables

        Returns
        -------
        PanelData
            Differenced values
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
    def _minimize_multiindex(df: DataFrame) -> DataFrame:
        index_cols = list(df.index.names)
        orig_names = index_cols[:]
        for i, col in enumerate(index_cols):
            col = ensure_unique_column(col, df)
            index_cols[i] = col
        df.index.names = index_cols
        df = df.reset_index()
        df = df.set_index(index_cols)
        df.index.names = orig_names
        return df

    def dummies(self, group: str = "entity", drop_first: bool = False) -> DataFrame:
        """
        Generate entity or time dummies

        Parameters
        ----------
        group : {'entity', 'time'}, optional
            Type of dummies to generate
        drop_first : bool, optional
            Flag indicating that the dummy column corresponding to the first
            entity or time period should be dropped

        Returns
        -------
        DataFrame
            Dummy variables
        """
        if group not in ("entity", "time"):
            raise ValueError
        axis = 0 if group == "entity" else 1
        labels = self.index.codes
        levels = self.index.levels
        cat = Categorical(levels[axis][labels[axis]])
        dummies = get_dummies(cat, drop_first=drop_first)
        cols = self.entities if group == "entity" else self.time
        return dummies[[c for c in cols if c in dummies]].astype(np.float64, copy=False)


PanelDataLike = Union[PanelData, ArrayLike]
