from linearmodels.compat.statsmodels import Summary

import numpy as np
import pandas as pd
from statsmodels.iolib import summary
from statsmodels.iolib.summary import SimpleTable

from linearmodels.asset_pricing import LinearFactorModel
from linearmodels.iv import IV2SLS
from linearmodels.shared.base import _SummaryStr


def test_sumary_str():
    ss = _SummaryStr()
    assert isinstance(ss.summary, Summary)


def make_summary(extra_txt=None):
    smry = Summary()
    smry.tables.append(SimpleTable([["1.0"]], headers=["Parameter"], stubs=["x"]))
    if extra_txt is not None:
        smry.add_extra_txt(extra_txt)
    return smry


def test_summary_latex_escapes_extra_txt():
    extra_txt = [
        "Endogenous: my_x",
        "Instruments: my_z1, my_z2",
        "50% & $5 #1 {a} ~ ^ \\",
    ]
    smry = make_summary(extra_txt)
    expected = r"""Endogenous: my\_x \newline
 Instruments: my\_z1, my\_z2 \newline
 50\% \& \$5 \#1 \{a\} \textasciitilde{} \textasciicircum{} \textbackslash{}"""
    assert smry.as_latex().endswith("\n\n" + expected)
    # Escaping must not change the stored text or the other formats
    assert smry.extra_txt == "\n".join(extra_txt)
    assert "my_x" in smry.as_text()
    assert "my_x" in smry.as_html()


def test_summary_latex_unchanged_without_special_characters():
    smry = make_summary(["Endogenous: x", "Instruments: z1, z2"])
    assert smry.as_latex() == summary.Summary.as_latex(smry)
    smry = make_summary()
    assert smry.as_latex() == summary.Summary.as_latex(smry)


def test_summary_latex_end_to_end():
    rng = np.random.default_rng(12345)
    nobs = 200
    instruments = pd.DataFrame(
        rng.standard_normal((nobs, 2)), columns=["my_z1", "my_z2"]
    )
    endog = pd.DataFrame({"my_x": instruments.sum(axis=1) + rng.standard_normal(nobs)})
    dependent = pd.Series(
        endog["my_x"] + rng.standard_normal(nobs), name="my_y", dtype=float
    )
    exog = pd.DataFrame({"my_const": np.ones(nobs)})
    latex = IV2SLS(dependent, exog, endog, instruments).fit().summary.as_latex()
    assert r"Endogenous: my\_x" in latex
    assert r"Instruments: my\_z1, my\_z2" in latex
    assert "my_x" not in latex

    factors = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["f1", "f2"])
    portfolios = pd.DataFrame(
        rng.standard_normal((nobs, 4)) + factors.sum(axis=1).to_numpy()[:, None],
        columns=["p1", "p2", "p3", "p4"],
    )
    latex = LinearFactorModel(portfolios, factors).fit().summary.as_latex()
    assert r"See full\_summary for complete results" in latex
    assert "full_summary" not in latex
