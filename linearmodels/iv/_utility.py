import numpy as np
from patsy.highlevel import dmatrices, dmatrix
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


def parse_formula(formula, data):
    """Parse an IV model forumla"""
    na_action = NAAction(on_NA='raise', NA_types=[])
    if formula.count('~') == 1:
        dep, exog = dmatrices(formula, data, return_type='dataframe', NA_action=na_action)
        endog = instr = None
        return dep, exog, endog, instr

    elif formula.count('~') > 2:
        raise ValueError('formula not understood.  Must have 1 or 2 '
                         'occurrences of ~')

    blocks = [bl.strip() for bl in formula.strip().split('~')]
    if '[' not in blocks[1] or ']' not in blocks[2]:
        raise ValueError('formula not understood. Endogenous variables and '
                         'instruments must be segregated in a block that '
                         'starts with [ and ends with ].')

    dep = blocks[0].strip()
    exog, endog = [bl.strip() for bl in blocks[1].split('[')]
    instr, exog2 = [bl.strip() for bl in blocks[2].split(']')]
    if endog[0] == '+' or endog[1] == '+':
        raise ValueError(
            'endogenous block must not start or end with +. This block was: {0}'.format(endog))
    if instr[0] == '+' or instr[1] == '+':
        raise ValueError(
            'instrument block must not start or end with +. This block was: {0}'.format(instr))
    if exog2:
        exog += exog2

    if exog:
        exog = exog[:-1].strip() if exog[-1] == '+' else exog

    try:
        dep = dmatrix('0 + ' + dep, data, eval_env=2,
                      return_type='dataframe', NA_action=na_action)
        exog = '0 + ' + exog if exog else '0'
        exog = dmatrix(exog, data, eval_env=2,
                       return_type='dataframe', NA_action=na_action)
        endog = dmatrix('0 + ' + endog, data, eval_env=2,
                        return_type='dataframe', NA_action=na_action)
        instr = dmatrix('0 + ' + instr, data, eval_env=2,
                        return_type='dataframe', NA_action=na_action)
    except Exception as e:
        raise type(e)(PARSING_ERROR.format(dep, exog, endog, instr) + e.msg, e.args[1])

    return dep, exog, endog, instr
