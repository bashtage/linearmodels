import numpy as np
from patsy.highlevel import dmatrices, dmatrix
from patsy.missing import NAAction


def clean_patsy_endogenous(formula: str):
    formula = ' '.join(formula.strip().split())
    tokens = []
    st = 0
    lparen = 0
    for i, c in enumerate(formula):
        if c == ' ' and lparen == 0:
            tokens.append(formula[st:i])
            st = i
        elif c == '(':
            lparen += 1
        elif c == ')':
            lparen -= 1
    tokens.append(formula[st:].strip())
    tokens = [t for t in tokens if t.strip()]
    return ' + '.join(tokens)


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
    if formula.count('~') == 1:
        dep, exog = dmatrices(formula, data, return_type='dataframe')
        endog = instr = None
        return dep, exog, endog, instr

    elif formula.count('~') > 2:
        raise ValueError('formula not understood.  Must have 1 or 2 '
                         'occurrences of ~')

    # Manual formula parse
    formula = formula.strip()
    components = formula.split('~')
    components = [c.strip() for c in components]
    if ((components[1].count('(') - components[1].count(')')) != 1 or
            (components[2].count(')') - components[2].count('(')) != 1):
        raise ValueError('formula not understood. Please see the examples.')

    dep = components[0].strip()
    rparen = 0
    for i, c in enumerate(components[1][::-1]):
        if c == ')':
            rparen += 1
        elif c == '(':
            rparen -= 1
            if rparen < 0:
                break
    endog_start = len(components[1]) - i - 1
    exog = components[1][:endog_start].strip()
    endog = components[1][endog_start + 1:].strip()
    lparen = 0
    for i, c in enumerate(components[2]):
        if c == '(':
            lparen += 1
        elif c == ')':
            lparen -= 1
            if lparen < 0:
                break
    instr_end = i
    instr = components[2][:instr_end]
    if instr_end + 1 != len(components[2]):
        exog += ' ' + components[2][instr_end + 1:]
    endog = clean_patsy_endogenous(endog)
    if exog[-1] == '+':
        exog = exog[:-1]
    na_action = NAAction(on_NA='raise', NA_types=[])
    dep = dmatrix('0 + ' + dep, data, eval_env=2, return_type='dataframe', NA_action=na_action)
    exog = dmatrix('0 + ' + exog, data, eval_env=2, return_type='dataframe', NA_action=na_action)
    endog = dmatrix('0 + ' + endog, data, eval_env=2, return_type='dataframe', NA_action=na_action)
    instr = dmatrix('0 + ' + instr, data, eval_env=2, return_type='dataframe', NA_action=na_action)

    return dep, exog, endog, instr
