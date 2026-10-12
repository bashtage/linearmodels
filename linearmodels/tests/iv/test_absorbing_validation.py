"""
Validation and messages of AbsorbingLS and the IV models.

* AbsorbingLS.fit accepts the methods "auto", "hdfe" and "lsmr", whatever their case,
  and nothing else. It used LSMR for any other name.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import linearmodels.iv.absorbing
from linearmodels.iv.absorbing import AbsorbingLS

NOBS = 200

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.standard_normal((NOBS, 2)), columns=["x0", "x1"])
    cats = pd.DataFrame(
        {
            "g0": pd.Categorical(rng.integers(0, 9, NOBS)),
            "g1": pd.Categorical(rng.integers(0, 4, NOBS)),
        }
    )
    y = pd.Series(x.sum(axis=1) + rng.standard_normal(NOBS), name="y")
    return y, x, cats


@pytest.fixture
def spy_lsmr(monkeypatch):
    """Record the calls to the function that absorbs variables using LSMR"""
    calls = []
    original = linearmodels.iv.absorbing.lsmr_annihilate

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(linearmodels.iv.absorbing, "lsmr_annihilate", spy)
    return calls


@pytest.mark.parametrize(
    "method", ["hdfee", "foo", "", "hdf", "lsmr ", None, 1, ["hdfe"]]
)
def test_unknown_method(data, method):
    y, x, cats = data
    mod = AbsorbingLS(y, x, absorb=cats)
    with pytest.raises(ValueError, match=r'one of "auto", "hdfe" or "lsmr"') as info:
        mod.fit(method=method)
    assert repr(method) in str(info.value)
    # The model is not affected by the failed call
    assert_allclose(mod.fit().params, mod.fit(method="auto").params)


@pytest.mark.parametrize("method", ["auto", "Auto", "AUTO", "hdfe", "HDFE", "Hdfe"])
def test_hdfe_by_name(data, spy_lsmr, method):
    y, x, cats = data
    res = AbsorbingLS(y, x, absorb=cats).fit(method=method)
    assert not spy_lsmr
    expected = AbsorbingLS(y, x, absorb=cats).fit(method="hdfe")
    assert_allclose(res.params, expected.params, rtol=1e-12)


@pytest.mark.parametrize("method", ["lsmr", "LSMR", "Lsmr"])
def test_lsmr_by_name(data, spy_lsmr, method):
    y, x, cats = data
    res = AbsorbingLS(y, x, absorb=cats).fit(method=method)
    assert len(spy_lsmr) == 1
    expected = AbsorbingLS(y, x, absorb=cats).fit(method="lsmr")
    assert_allclose(res.params, expected.params, rtol=1e-12)


def test_hdfe_name_is_checked_against_the_model(data):
    # A case that is not the lower case is the same method, and so it is checked in the
    # same way: it cannot be used with continuous variables to absorb
    y, x, cats = data
    absorb = pd.concat([cats, x[["x1"]].rename(columns={"x1": "z"})], axis=1)
    with pytest.raises(RuntimeError, match="HDFE has been set"):
        AbsorbingLS(y, x[["x0"]], absorb=absorb).fit(method="HDFE")


def test_method_with_another_case_does_not_absorb_again(data, monkeypatch):
    y, x, cats = data
    mod = AbsorbingLS(y, x, absorb=cats)
    calls = []
    original = AbsorbingLS._first_time_fit

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AbsorbingLS, "_first_time_fit", spy)
    mod.fit(method="HDFE")
    mod.fit(method="hdfe")
    mod.fit(method="Hdfe", cov_type="unadjusted")
    assert len(calls) == 1
    mod.fit(method="LSMR")
    assert len(calls) == 2
