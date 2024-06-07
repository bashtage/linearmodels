from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formulaic import model_matrix
from formulaic.formula import Formula
from formulaic.materializers.types import NAAction as fNAAction
from formulaic.utils.context import capture_context
import numpy as np
import pandas
from pandas import DataFrame

import linearmodels.typing.data

from ..compat.formulaic import monkey_patch_materializers

# Monkey patch parsers if needed, remove once formulaic updated
monkey_patch_materializers()

PARSING_ERROR = """
Conversion of formula blocks to DataFrames failed.
The formula blocks used for conversion were:

dependent: {0}
exogenous: {1}
endogenous: {2}
instruments: {3}

The original error was:
"""


def proj(
    y: linearmodels.typing.data.Float64Array, x: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    """
    Projection of y on x from y

    Parameters
    ----------
    y : ndarray
        Array to project (nobs by nseries)
    x : ndarray
        Array to project onto (nobs by nvar)

    Returns
    -------
    ndarray
        Projected values of y (nobs by nseries)
    """
    if x.shape[1] == 0:
        return np.zeros_like(y)
    return x @ (np.linalg.pinv(x) @ y)


def annihilate(
    y: linearmodels.typing.data.Float64Array, x: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    """
    Remove projection of y on x from y

    Parameters
    ----------
    y : ndarray
        Array to project (nobs by nseries)
    x : ndarray
        Array to project onto (nobs by nvar)

    Returns
    -------
    ndarray
        Residuals values of y minus y projected on x (nobs by nseries)
    """
    return y - proj(y, x)


class IVFormulaParser:
    """
    Parse formulas for OLS and IV models

    Parameters
    ----------
    formula : str
        String formula object.
    data : DataFrame
        Frame containing values for variables used in formula
    eval_env : int
        Stack depth to use when evaluating formulas

    Notes
    -----
    The general structure of a formula is `dep ~ exog + [endog ~ instr]`
    """

    def __init__(
        self,
        formula: str,
        data: pandas.DataFrame,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
    ):
        self._formula = formula
        self._data = data
        self._eval_env = eval_env
        if not context:
            self._context = capture_context(context=self._eval_env)
        else:
            self._context = context
        self._components: dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        blocks = self._formula.strip().split("~")
        if len(blocks) == 2:
            dep = blocks[0].strip()
            exog = blocks[1].strip()
            endog = "0"
            instr = "0"
        elif len(blocks) == 3:
            blocks = [bl.strip() for bl in blocks]
            if "[" not in blocks[1] or "]" not in blocks[2]:
                raise ValueError(
                    "formula not understood. Endogenous variables and "
                    "instruments must be segregated in a block that "
                    "starts with [ and ends with ]."
                )
            dep = blocks[0].strip()
            exog, endog = (bl.strip() for bl in blocks[1].split("["))
            instr, exog2 = (bl.strip() for bl in blocks[2].split("]"))
            if endog[0] == "+" or endog[-1] == "+":
                raise ValueError(
                    "endogenous block must not start or end with +. This block "
                    "was: {}".format(endog)
                )
            if instr[0] == "+" or instr[-1] == "+":
                raise ValueError(
                    "instrument block must not start or end with +. This "
                    "block was: {}".format(instr)
                )
            if exog:
                exog = exog[:-1].strip() if exog[-1] == "+" else exog
            if exog2:
                exog += exog2
            exog = "0" if not exog else "0 + " + exog
        else:
            raise ValueError("formula contains more then 2 separators (~)")
        comp = {
            "dependent": "0 + " + dep,
            "exog": exog,
            "endog": endog,
            "instruments": instr,
        }
        self._components = comp

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth"""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:
        self._eval_env = value

    @property
    def data(
        self,
    ) -> tuple[DataFrame, DataFrame | None, DataFrame | None, DataFrame | None]:
        """Returns a tuple containing the dependent, exog, endog and instruments"""
        self._eval_env += 1
        out = self.dependent, self.exog, self.endog, self.instruments
        self._eval_env -= 1
        return out

    @property
    def dependent(self) -> DataFrame:
        """Dependent variable"""
        dep_fmla = self.components["dependent"]
        dep = model_matrix(
            dep_fmla,
            self._data,
            context=self._context,
            ensure_full_rank=False,
            na_action=fNAAction("raise"),
        )
        return DataFrame(dep)

    @property
    def exog(self) -> DataFrame | None:
        """Exogenous variables"""
        exog_fmla = self.components["exog"]
        exog = Formula(exog_fmla).get_model_matrix(
            self._data,
            context=self._context,
            ensure_full_rank=True,
            na_action=fNAAction("ignore"),
        )
        return self._empty_check(DataFrame(exog))

    @property
    def endog(self) -> DataFrame | None:
        """Endogenous variables"""
        endog_fmla = "0 +" + self.components["endog"]
        endog = Formula(endog_fmla).get_model_matrix(
            self._data,
            context=self._context,
            ensure_full_rank=False,
            na_action=fNAAction("raise"),
        )
        return self._empty_check(DataFrame(endog))

    @property
    def instruments(self) -> DataFrame | None:
        """Instruments"""
        instr_fmla = "0 +" + self.components["instruments"]
        instr = Formula(instr_fmla).get_model_matrix(
            self._data,
            context=self._context,
            ensure_full_rank=False,
            na_action=fNAAction("raise"),
        )
        return self._empty_check(DataFrame(instr))

    @property
    def components(self) -> dict[str, str]:
        """Dictionary containing the string components of the formula"""
        return self._components

    @staticmethod
    def _empty_check(arr: pandas.DataFrame) -> DataFrame | None:
        return None if arr.shape[1] == 0 else arr
