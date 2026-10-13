"""Panels and comparisons of results that the tests of the checks of inputs share"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

NENTITY = 14
NPERIOD = 7
PERIODS = pd.period_range("2001-01", periods=NPERIOD, freq="M")
EXOG = ["const", "x1", "x2"]


def make_frame(times, unbalanced=True, seed=11):
    """A panel indexed by (firm, month) with the columns y, const, x1, x2, w, g, cl"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(NENTITY), times], names=["firm", "month"]
    )
    nobs = index.shape[0]
    entity = index.codes[0]
    x1 = rg.standard_normal(nobs) + 0.3 * (entity % 3)
    x2 = rg.standard_normal(nobs)
    y = 1 + 0.5 * x1 - 0.3 * x2 + rg.standard_normal(NENTITY)[entity]
    y = y + (0.5 + np.abs(x1)) * rg.standard_normal(nobs)
    frame = pd.DataFrame(
        {
            "y": y,
            "const": 1.0,
            "x1": x1,
            "x2": x2,
            "w": rg.chisquare(4, nobs) / 4 + 0.2,
            "g": (np.arange(nobs) // 5) % 4,
            # Clusters of the entities
            "cl": entity // 3,
        },
        index=index,
    )
    if unbalanced:
        keep = rg.random((NENTITY, len(times))) > 0.25
        for i in range(NENTITY):
            keep[i, rg.choice(len(times), 4, replace=False)] = True
        frame = frame.loc[keep.ravel()]
    return frame


def assert_equal_results(res, expected):
    """Every number of two results is the same, and so is the index"""
    assert type(res) is type(expected)
    skip = {"model", "summary", "wald_test", "hausman", "predict", "conf_int"}
    skip |= {"compare", "pesaran_cd"}
    for name in dir(res):
        if name.startswith("_") or name in skip:
            continue
        value, other = getattr(res, name), getattr(expected, name)
        if isinstance(value, (pd.Series, pd.DataFrame)):
            assert value.index.equals(other.index), name
            if isinstance(value, pd.DataFrame):
                assert value.columns.equals(other.columns), name
            assert_allclose(
                np.asarray(value, dtype=float),
                np.asarray(other, dtype=float),
                rtol=1e-10,
                atol=1e-12,
                err_msg=name,
            )
        elif hasattr(value, "stat"):
            assert_allclose(value.stat, other.stat, rtol=1e-10, err_msg=name)
            assert_allclose(value.pval, other.pval, rtol=1e-10, err_msg=name)
        elif isinstance(value, (int, float, np.number)):
            assert_allclose(value, other, rtol=1e-10, err_msg=name)
    assert_allclose(res.conf_int(), expected.conf_int(), rtol=1e-10, atol=1e-12)
    assert res.pesaran_cd.__class__ is expected.pesaran_cd.__class__
