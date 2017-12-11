import numpy as np
from patsy.highlevel import dmatrix
from patsy.missing import NAAction

PARSING_ERROR = """
Conversion of formula blocks to DataFrames using patsy failed.
The formula blocks used for conversion were:

dependent: {0}
exogenous: {1}
endogenous: {2}
instruments: {3}

The original Patsy error was:
"""


def proj(y, x):
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
    yhat : ndarray
        Projected values of y (nobs by nseries)
    """
    return x @ np.linalg.pinv(x) @ y


def annihilate(y, x):
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
    eps : ndarray
        Residuals values of y minus y projected on x (nobs by nseries)
    """
    return y - proj(y, x)


class IVFormulaParser(object):
    """
    Parse formulas for OLS and IV models

    Parameters
    ----------
    formula : str
        String formula object.
    data : DataFrame
        Frame containing values for variables used in formula
    eval_env : int
        Stack depth to use when evaluating Patsy formulas

    Notes
    -----
    The general structure of a formula is `dep ~ exog + [endog ~ instr]`
    """

    def __init__(self, formula, data, eval_env=2):
        self._formula = formula
        self._data = data
        self._na_action = NAAction(on_NA='raise', NA_types=[])
        self._eval_env = eval_env
        self._components = {}
        self._parse()

    def _parse(self):
        blocks = self._formula.strip().split('~')
        if len(blocks) == 2:
            dep = blocks[0].strip()
            exog = blocks[1].strip()
            endog = '0'
            instr = '0'
        elif len(blocks) == 3:
            blocks = [bl.strip() for bl in blocks]
            if '[' not in blocks[1] or ']' not in blocks[2]:
                raise ValueError('formula not understood. Endogenous variables and '
                                 'instruments must be segregated in a block that '
                                 'starts with [ and ends with ].')
            dep = blocks[0].strip()
            exog, endog = [bl.strip() for bl in blocks[1].split('[')]
            instr, exog2 = [bl.strip() for bl in blocks[2].split(']')]
            if endog[0] == '+' or endog[1] == '+':
                raise ValueError('endogenous block must not start or end with +. This block '
                                 'was: {0}'.format(endog))
            if instr[0] == '+' or instr[1] == '+':
                raise ValueError('instrument block must not start or end with +. This '
                                 'block was: {0}'.format(instr))
            if exog2:
                exog += exog2
            if exog:
                exog = exog[:-1].strip() if exog[-1] == '+' else exog
            exog = '0' if not exog else '0 + ' + exog
        else:
            raise ValueError('formula contains more then 2 separators (~)')
        comp = {'dependent': '0 + ' + dep,
                'exog': exog,
                'endog': endog,
                'instruments': instr}
        self._components = comp

    @property
    def eval_env(self):
        """Set or get the eval env depth"""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value):
        self._eval_env = value

    @property
    def data(self):
        """Returns a tuple containing the dependent, exog, endog and instruments"""
        self._eval_env += 1
        out = self.dependent, self.exog, self.endog, self.instruments
        self._eval_env -= 1
        return out

    @property
    def dependent(self):
        """Dependent variable"""
        dep = self.components['dependent']
        dep = dmatrix('0 + ' + dep, self._data, eval_env=self._eval_env,
                      return_type='dataframe', NA_action=self._na_action)
        return dep

    @property
    def exog(self):
        """Exogenous variables"""
        exog = self.components['exog']
        exog = dmatrix(exog, self._data, eval_env=self._eval_env,
                       return_type='dataframe', NA_action=self._na_action)
        return exog

    @property
    def endog(self):
        """Endogenous variables"""
        endog = self.components['endog']
        endog = dmatrix('0 + ' + endog, self._data, eval_env=self._eval_env,
                        return_type='dataframe', NA_action=self._na_action)
        return endog

    @property
    def instruments(self):
        """Instruments"""
        instr = self.components['instruments']
        instr = dmatrix('0 + ' + instr, self._data, eval_env=self._eval_env,
                        return_type='dataframe', NA_action=self._na_action)
        return instr

    @property
    def components(self):
        """Dictionary containing the string components of the formula"""
        return self._components
