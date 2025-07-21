from __future__ import annotations

from linearmodels.iv.data import IVData
from linearmodels.panel.data import PanelData
from linearmodels.typing import AnyPandas, ArrayLike, NumericArray, Float64Array
import numpy as np
from numpy import cast
from numpy.linalg import lstsq
import pandas as pd
from pandas import DataFrame, Series
from typing import Any, Literal, Union, overload


class IVPanelData(PanelData, IVData):
    def __init__(
            self,
            x: IVPanelDataLike,
            var_name: str = "x",
            convert_dummies: bool = True,
            drop_first: bool = True,
            copy: bool = True,
    ):
        super().__init__(x, var_name, convert_dummies, drop_first, copy)
        self.original = self._original

    @property
    def pandas(self) -> pd.DataFrame:
        """DataFrame view of data"""
        return self.dataframe

    @property
    def ndarray(self) -> NumericArray:
        """ndarray view of data, always 2d"""
        return self.values2d

    @property
    def labels(self) -> dict[int, Any]:
        """Dictionary containing row and column labels keyed by axis"""
        return {0: self._row_labels, 1: self._col_labels}

    def _demean_both_low_mem(self, weights: IVPanelData | None) -> IVPanelData:
        groups = IVPanelData(
            DataFrame(np.c_[self.entity_ids, self.time_ids], index=self._frame.index),
            convert_dummies=False,
            copy=False,
        )
        return self.general_demean(groups, weights=weights)

    def _demean_both(self, weights: IVPanelData | None) -> IVPanelData:
        """
        Entity and time demean

        Parameters
        ----------
        weights : IVPanelData
             Weights to use in demeaning
        """
        group: Literal["entity", "time"]
        if self.nentity > self.nobs:
            group = "entity"
            dummy = "time"
        else:
            group = "time"
            dummy = "entity"
        e = self.demean(group, weights=weights)
        assert isinstance(e, IVPanelData)
        d = self.dummies(dummy, drop_first=True)
        d.index = e.index
        d_pd = IVPanelData(d).demean(group, weights=weights)
        d_arr = d_pd.values2d
        e_arr = e.values2d
        resid_arr = e_arr - d_arr @ lstsq(d_arr, e_arr, rcond=None)[0]
        resid = DataFrame(
            resid_arr, index=self._frame.index, columns=self._frame.columns
        )

        return IVPanelData(resid)

    def general_demean(
        self, groups: IVPanelDataLike, weights: IVPanelData | None = None
    ) -> IVPanelData:
        """
        Multi-way demeaning using only groupby

        Parameters
        ----------
        groups : IVPanelData
            Arrays with the same size containing group identifiers
        weights : IVPanelData
            Weights to use in the weighted demeaning

        Returns
        -------
        IVPanelData
            Weighted, demeaned data according to groups

        Notes
        -----
        Iterates until convergence
        """
        if not isinstance(groups, IVPanelData):
            groups = IVPanelData(groups)
        if weights is None:
            weights = IVPanelData(
                DataFrame(
                    np.ones((self._frame.shape[0], 1)),
                    index=self.index,
                    columns=["weights"],
                )
            )
        groups = groups.values2d.astype(np.int64, copy=False)

        weight_sum: dict[int, Series | DataFrame] = {}

        def weighted_group_mean(
            df: DataFrame, weights: DataFrame, root_w: Float64Array, level: int
        ) -> Float64Array:
            scaled_df = cast(DataFrame, root_w * df)
            num = scaled_df.groupby(level=level).transform("sum")
            if level in weight_sum:
                denom = np.asarray(weight_sum[level])
            else:
                denom_df = weights.groupby(level=level).transform("sum")
                weight_sum[level] = denom_df
                denom = np.asarray(denom_df)
            return np.asarray(num) / denom

        def demean_pass(
            frame: DataFrame, weights: DataFrame, root_w: Float64Array
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

        root_w = cast(Float64Array, np.sqrt(weights.values2d))
        weights_df = DataFrame(weights.values2d, index=init_index.index)
        wframe = cast(DataFrame, root_w * self._frame)
        wframe.index = init_index.index

        previous = wframe
        current: DataFrame = demean_pass(previous, weights_df, root_w)
        if groups.shape[1] == 1:
            current.index = self._frame.index
            return IVPanelData(current)

        exclude = np.ptp(np.asarray(self._frame), 0) == 0
        max_rmse = np.sqrt(np.asarray(self._frame).var(0).max())
        scale = np.require(self._frame.std(), requirements="W")
        exclude = exclude | (scale < 1e-14 * max_rmse)
        replacement = cast(Float64Array, np.maximum(scale, 1))
        scale[exclude] = replacement[exclude]
        scale = scale[None, :]

        while np.max(np.abs(np.asarray(current) - np.asarray(previous)) / scale) > 1e-8:
            previous = current
            current = demean_pass(previous, weights_df, root_w)
        current.index = self._frame.index

        return IVPanelData(current)

    @overload
    def demean(
        self,
        group: Literal["entity", "time", "both"],
        *,
        return_panel: Literal[False],
    ) -> Float64Array:
        ...

    @overload
    def demean(
        self,
        group: Literal["entity", "time", "both"] = ...,
        weights: IVPanelData | None = ...,
        return_panel: Literal[True] = ...,
        low_memory: bool = ...,
    ) -> IVPanelData:
        ...

    @overload
    def demean(
        self,
        group: Literal["entity", "time", "both"],
        weights: IVPanelData | None,
        return_panel: Literal[False],
    ) -> Float64Array:
        ...

    def demean(
        self,
        group: Literal["entity", "time", "both"] = "entity",
        weights: IVPanelData | None = None,
        return_panel: bool = True,
        low_memory: bool = False,
    ) -> IVPanelData | Float64Array:
        """
        Demeans data by either entity or time group

        Parameters
        ----------
        group : {"entity", "time", "both"}
            Group to use in demeaning
        weights : IVPanelData
            Weights to implement weighted averaging
        return_panel : bool
            Flag indicating to return a IVPanelData object. If False, a 2-d
            NumPy representation of the panel is returned
        low_memory : bool
            Flag indicating whether to use a low memory implementation
            that avoids constructing dummy variables. Only relevant when
            group is "both"

        Returns
        -------
        IVPanelData
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
            return IVPanelData(out)
        else:
            w = weights.values2d
            frame: DataFrame = self._frame.copy()
            frame = w * frame
            weighted_sum: DataFrame = frame.groupby(level=level).transform("sum")
            frame.iloc[:, :] = w
            sum_weights: DataFrame = frame.groupby(level=level).transform("sum")
            group_mu = weighted_sum / sum_weights
            out_df = np.sqrt(w) * (self._frame - group_mu)
            if not return_panel:
                return np.asarray(out_df)
            return IVPanelData(out_df)



IVPanelDataLike = Union[IVPanelData, ArrayLike]
