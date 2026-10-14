"""
Formatting of the tables in the summaries of results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.iolib.summary import SimpleTable, fmt_params


def _str(v: float) -> str:
    """Preferred basic formatter."""
    if np.isnan(v):
        return "        "
    if np.isinf(v):
        return f"{v:8.4g}"
    av = abs(v)
    digits = 0
    if av != 0:
        digits = int(np.ceil(np.log10(av)))
    if digits > 4 or digits <= -4:
        return f"{v:8.4g}"

    if digits > 0:
        d = int(5 - digits)
    else:
        d = 4

    format_str = "{0:" + f"0.{d}f" + "}"
    return format_str.format(v)


def pval_format(v: float) -> str:
    """
    Format a probability.

    Parameters
    ----------
    v : float
        The value to format, which is in [0, 1] or NaN.

    Returns
    -------
    str
        The value with 4 digits after the decimal point, or blanks if the value
        is NaN.
    """
    if np.isnan(v):
        return "        "
    return f"{v:4.4f}"


# TODO: typing for Any
def param_table(results: Any, title: str, pad_bottom: bool = False) -> SimpleTable:
    """
    Create the table of parameter estimates of a model.

    Parameters
    ----------
    results : object
        Results of a model, with ``params``, ``std_errors``, ``tstats`` and
        ``pvalues`` that are Series and a ``conf_int`` method.
    title : str
        Title of the table.
    pad_bottom : bool, optional
        Whether to add a blank row below the parameters. The default is False.

    Returns
    -------
    SimpleTable
        The table, with a row for each parameter and columns for the estimate,
        the standard error, the t-statistic, the p-value and the confidence
        interval.
    """
    param_data = np.c_[
        np.asarray(results.params)[:, None],
        np.asarray(results.std_errors)[:, None],
        np.asarray(results.tstats)[:, None],
        np.asarray(results.pvalues)[:, None],
        results.conf_int(),
    ]
    data = []
    for row in param_data:
        txt_row = []
        for i, v in enumerate(row):
            func = _str
            if i == 3:
                func = pval_format
            txt_row.append(func(v))
        data.append(txt_row)
    header = ["Parameter", "Std. Err.", "T-stat", "P-value", "Lower CI", "Upper CI"]
    table_stubs = list(results.params.index)
    if pad_bottom:
        # Append blank row for spacing
        data.append([""] * 6)
        table_stubs += [""]

    return SimpleTable(
        data, stubs=table_stubs, txt_fmt=fmt_params, headers=header, title=title
    )


def format_wide(s: list[str] | tuple[str, ...], cols: int) -> list[list[str]]:
    """
    Format a list of strings.

    Parameters
    ----------
    s : list of str or tuple of str
        List of strings to format.
    cols : int
        Maximum number of characters in a line of the output.

    Returns
    -------
    list of list of str
        The lines. Each line is a list with a single string, which joins the
        strings that fit in the line with commas.
    """
    lines = []
    line = ""
    for i, val in enumerate(s):
        if line == "":
            line = val
            if i + 1 != len(s):
                line += ", "
        else:
            temp = line + val
            if i + 1 != len(s):
                temp += ", "
            if len(temp) > cols:
                lines.append([line])
                line = val
                if i + 1 != len(s):
                    line += ", "
            else:
                line = temp
    lines.append([line])
    return lines


def add_star(value: str, pvalue: float, star: bool) -> str:
    """
    Add 1, 2 or 3 stars to a string based on the p-value.

    Adds 1 star if the p-value is at most 10%, 2 if at most 5% and 3 if at most
    1%.

    Parameters
    ----------
    value : str
        The formatted parameter value as a string.
    pvalue : float
        The p-value of the parameter.
    star : bool
        Flag indicating whether the star should be added.

    Returns
    -------
    str
        The value, followed by the stars if ``star`` is True.
    """
    if not star:
        return value
    return value + "*" * sum(pvalue <= c for c in (0.01, 0.05, 0.1))
