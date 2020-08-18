from typing import Any, List, Sequence

import numpy as np
from statsmodels.iolib.summary import SimpleTable, fmt_params


def _str(v: float) -> str:
    """Preferred basic formatter"""
    if np.isnan(v):
        return "        "
    av = abs(v)
    digits = 0
    if av != 0:
        digits = int(np.ceil(np.log10(av)))
    if digits > 4 or digits <= -4:
        return "{0:8.4g}".format(v)

    if digits > 0:
        d = int(5 - digits)
    else:
        d = int(4)

    format_str = "{0:" + "0.{0}f".format(d) + "}"
    return format_str.format(v)


def pval_format(v: float) -> str:
    """Preferred formatting for x in [0,1]"""
    if np.isnan(v):
        return "        "
    return "{0:4.4f}".format(v)


# TODO: typing for Any
def param_table(results: Any, title: str, pad_bottom: bool = False) -> SimpleTable:
    """Formatted standard parameter table"""
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


def format_wide(s: Sequence[str], cols: int) -> List[List[str]]:
    """
    Format a list of strings.

    Parameters
    ----------
    s : List[str]
        List of strings to format
    cols : int
        Number of columns in output

    Returns
    -------
    List[List[str]]
        The joined list.
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
    Add 1, 2 or 3 stars to a string base on the p-value

    Adds 1 star if the pvalue is less than 10%, 2 if less than 5% and 3 is
    less than 1%.

    Parameters
    ----------
    value : str
        The formatted parameter value as a string.
    pvalue : float
        The p-value of the parameter
    star : bool
        Flag indicating whether the star should be added
    """
    if not star:
        return value
    return value + "*" * sum([pvalue <= c for c in (0.01, 0.05, 0.1)])
