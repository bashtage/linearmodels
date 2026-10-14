"""
Validation and messages of AbsorbingLS and the IV models.

* AbsorbingLS.fit accepts the methods "auto", "hdfe" and "lsmr", whatever their case,
  and nothing else. It used LSMR for any other name.
* The FutureWarning of lsmr_options says what to use instead, which needs a method on
  models that can use HDFE.
* Interaction.drop modifies the instance, and nobs must be the number of observations
  that remain.
* The message about the length of clusters names only what the models have.
* The message about the estimation of kappa is grammatical.
"""

import re

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVLIML
import linearmodels.iv.absorbing
from linearmodels.iv.absorbing import AbsorbingLS, Interaction
from linearmodels.iv.covariance import CLUSTER_ERR
from linearmodels.iv.model import _IVModelBase
from linearmodels.shared.exceptions import MissingValueWarning

NOBS = 200


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


def test_lsmr_options_warning_says_how_to_migrate(data, spy_lsmr):
    y, x, cats = data
    options = {"atol": 1e-9, "btol": 1e-9}
    with pytest.warns(FutureWarning, match=r'absorb_options with method="lsmr"') as rec:
        old = AbsorbingLS(y, x, absorb=cats).fit(lsmr_options=options)
    assert len([w for w in rec if issubclass(w.category, FutureWarning)]) == 1
    assert len(spy_lsmr) == 1

    # The advice of the warning gives the same results without a warning. The model
    # can be absorbed using HDFE, and then the options of LSMR need the method.
    new = AbsorbingLS(y, x, absorb=cats).fit(method="lsmr", absorb_options=options)
    assert_allclose(new.params, old.params, rtol=1e-12)
    assert_allclose(new.std_errors, old.std_errors, rtol=1e-12)
    with pytest.raises(ValueError, match=r'method="lsmr"'):
        AbsorbingLS(y, x, absorb=cats).fit(absorb_options=options)


def test_deprecation_note_in_docstring_names_the_method():
    doc = AbsorbingLS.fit.__doc__
    assert 'Use absorb_options with method="lsmr"' in " ".join(doc.split())


@pytest.mark.parametrize("kind", ["both", "cat", "cont", "none", "frame"])
def test_interaction_drop(kind):
    rng = np.random.default_rng(1)
    cat = pd.DataFrame({"c": pd.Categorical(rng.integers(0, 4, 50))})
    cont = pd.DataFrame(rng.standard_normal((50, 2)), columns=["a", "b"])
    if kind == "both":
        inter = Interaction(cat, cont)
    elif kind == "cat":
        inter = Interaction(cat)
    elif kind == "cont":
        inter = Interaction(cont=cont)
    elif kind == "none":
        inter = Interaction(nobs=50)
    else:
        inter = Interaction.from_frame(pd.concat([cat, cont], axis=1))
    assert inter.nobs == 50
    locs = np.zeros(50, dtype=bool)
    locs[[0, 7, 49]] = True
    cat_before, cont_before = inter.cat.copy(), inter.cont.copy()

    assert inter.drop(locs) is None
    assert inter.nobs == 47
    assert inter.cat.shape[0] == inter.cont.shape[0] == 47
    assert inter.sparse.shape[0] == 47
    assert inter.isnull.shape[0] == 47
    pd.testing.assert_frame_equal(inter.cat, cat_before.loc[~locs])
    pd.testing.assert_frame_equal(inter.cont, cont_before.loc[~locs])

    # A second call drops from the remaining observations
    more = np.zeros(47, dtype=bool)
    more[:5] = True
    inter.drop(more)
    assert inter.nobs == 42
    assert inter.sparse.shape[0] == 42


def test_interaction_drop_is_documented():
    doc = Interaction.drop.__doc__
    assert doc is not None
    flat = " ".join(doc.split())
    assert "in place" in flat
    assert "nothing is returned" in flat
    assert Interaction.nobs.__doc__


def test_nobs_of_the_interactions_of_a_model_with_missing_values(data):
    y, x, cats = data
    x = x.copy()
    x.iloc[[3, 50], 0] = np.nan
    cont = pd.DataFrame({"k": np.arange(NOBS, dtype=float)})
    interaction = Interaction(cats[["g0"]], cont)
    with pytest.warns(MissingValueWarning):
        mod = AbsorbingLS(y, x, absorb=cats, interactions=interaction)
    assert mod.dependent.shape[0] == NOBS - 2
    assert mod._absorb_inter.nobs == NOBS - 2
    assert [i.nobs for i in mod._interaction_list] == [NOBS - 2]
    # The interaction of the caller is not modified
    assert interaction.nobs == NOBS


def test_cluster_message_names_only_what_the_models_have(data):
    y, x, cats = data
    y = y.copy()
    y.iloc[3] = np.nan
    clusters = np.arange(NOBS) % 7  # One row for each observation before the drop
    z = pd.DataFrame(
        np.random.default_rng(2).standard_normal((NOBS, 2)), columns=["z0", "z1"]
    )
    endog = (z.sum(axis=1) + x["x0"]).to_frame("e")
    with pytest.warns(MissingValueWarning, match="missing values"):
        models = {
            "IV2SLS": IV2SLS(y, x, endog, z),
            "IVLIML": IVLIML(y, x, endog, z),
            "AbsorbingLS": AbsorbingLS(y, x, absorb=cats),
        }
    for name, mod in models.items():
        with pytest.raises(ValueError, match="clusters has the wrong nobs") as info:
            mod.fit(cov_type="clustered", clusters=clusters)
        message = str(info.value)
        assert f"Expected {NOBS - 1}, got {NOBS}." in message
        assert "have have" not in message
        assert "missing" in message
        for attribute in re.findall(r"`([^`]+)`", message):
            assert hasattr(mod, attribute), f"{name} does not have {attribute}"
    # The shared text does not name any attribute of any model
    assert "`" not in CLUSTER_ERR


def test_clusters_that_are_dropped_with_the_data_are_accepted(data):
    # The advice of the message works
    y, x, cats = data
    y = y.copy()
    y.iloc[3] = np.nan
    clusters = np.arange(NOBS) % 7
    keep = np.ones(NOBS, dtype=bool)
    keep[3] = False
    mod = AbsorbingLS(y[keep], x[keep], absorb=cats[keep])
    res = mod.fit(cov_type="clustered", clusters=clusters[keep])
    assert res.nobs == NOBS - 1
    # The rows are the rows of the model that drops them
    with pytest.warns(MissingValueWarning, match="missing values"):
        auto = AbsorbingLS(y, x, absorb=cats)
    assert_allclose(
        auto.fit(cov_type="clustered", clusters=clusters[keep]).params, res.params
    )


def test_liml_kappa_message(data, monkeypatch):
    y, x, _ = data
    rng = np.random.default_rng(3)
    z = pd.DataFrame(rng.standard_normal((NOBS, 2)), columns=["z0", "z1"])
    endog = (z.sum(axis=1) + rng.standard_normal(NOBS)).to_frame("e")

    def fail(self):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(_IVModelBase, "_estimate_kappa", fail)
    with pytest.raises(ValueError, match="Unable to estimate kappa") as info:
        IVLIML(y, x, endog, z).fit()
    message = str(info.value)
    assert (
        "This most likely occurs if the instrument matrix is rank deficient." in message
    )
    assert "This is most likely" not in message
    assert "Eigenvalues did not converge" in message
