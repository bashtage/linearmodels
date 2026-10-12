"""
AbsorbingLS.fit with options of LSMR and the default method

The default method, "auto", absorbs the variables using HDFE when the model
can be absorbed using it, which is when it is not weighted and all of the
absorbed variables are categorical, and LSMR otherwise. The options are passed
to the algorithm that is used, either pyhdfe.create or scipy.sparse.linalg.lsmr.
Options of LSMR that are passed with the default method and a model that uses
HDFE reached pyhdfe, which raised

    TypeError: create() got an unexpected keyword argument 'atol'

The deprecated lsmr_options are options of LSMR, and so they select LSMR if the
method is "auto". Options of LSMR in absorb_options, which does not say which
algorithm it is for, are an error that says what to do.
"""

import functools
import re

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import absorbing
from linearmodels.iv.absorbing import AbsorbingLS, clear_cache
from linearmodels.shared.utility import AttrDict

NOBS = 400
KINDS = ["hdfe", "weighted", "mixed"]
# LSMR stops after 1 iteration, which is far from the solution that HDFE finds
LIMITED = {"maxiter": 1}
LSMR_OPTIONS = [
    {"atol": 1e-8},
    {"maxiter": 5},
    {"maxiter": 5, "damp": 0.1},
    {"btol": 1e-6, "conlim": 1e6},
    {"show": False},
]
HDFE_OPTIONS = [
    {"residualize_method": "map"},
    {"compute_degrees": False},
    {"residualize_method": "map", "drop_singletons": False},
]
LSMR_MESSAGE = re.escape("options of scipy.sparse.linalg.lsmr")
HDFE_MESSAGE = re.escape("options of pyhdfe.create")


@pytest.fixture(autouse=True)
def empty_cache():
    # The cache is shared by all models and so by all tests
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261011)
    cat = pd.DataFrame(
        {
            "g": pd.Categorical(rng.integers(0, 25, NOBS)),
            "h": pd.Categorical(rng.integers(0, 20, NOBS)),
        }
    )
    cont = pd.DataFrame({"z": rng.standard_normal(NOBS)})
    x = pd.DataFrame(rng.standard_normal((NOBS, 2)), columns=["x0", "x1"])
    x["x0"] += 0.05 * cat["g"].cat.codes + 0.05 * cat["h"].cat.codes
    y = pd.Series(rng.standard_normal(NOBS) + x.sum(axis=1), name="y")
    weights = pd.Series(np.exp(0.5 * rng.standard_normal(NOBS)), name="w")
    return AttrDict(y=y, x=x, cat=cat, cont=cont, weights=weights)


def make_model(data, kind="hdfe"):
    """HDFE can absorb the first, and only LSMR can absorb the others"""
    if kind == "hdfe":
        return AbsorbingLS(data.y, data.x, absorb=data.cat)
    if kind == "weighted":
        return AbsorbingLS(data.y, data.x, absorb=data.cat, weights=data.weights)
    return AbsorbingLS(data.y, data.x, absorb=pd.concat([data.cat, data.cont], axis=1))


def assert_same(res, expected):
    assert res.df_model == expected.df_model
    for name in ("params", "std_errors", "resids", "absorbed_effects"):
        assert_allclose(
            getattr(res, name).to_numpy(),
            getattr(expected, name).to_numpy(),
            rtol=1e-10,
            atol=1e-12,
            err_msg=name,
        )


def assert_different(res, other):
    assert not np.allclose(res.params, other.params, rtol=1e-6, atol=0)


@pytest.fixture
def count_absorptions(monkeypatch):
    """Record the calls that absorb variables"""
    calls = []
    original = AbsorbingLS._first_time_fit

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AbsorbingLS, "_first_time_fit", spy)
    return calls


@pytest.fixture
def algorithms(monkeypatch):
    """Record which of LSMR and HDFE absorb variables"""
    import pyhdfe  # noqa: PLC0415

    used = []
    lsmr_annihilate = absorbing.lsmr_annihilate
    create = pyhdfe.create

    # wraps keeps the signature of pyhdfe.create, which names its options
    @functools.wraps(lsmr_annihilate)
    def spy_lsmr(*args, **kwargs):
        used.append("lsmr")
        return lsmr_annihilate(*args, **kwargs)

    @functools.wraps(create)
    def spy_create(*args, **kwargs):
        used.append("hdfe")
        return create(*args, **kwargs)

    monkeypatch.setattr(absorbing, "lsmr_annihilate", spy_lsmr)
    monkeypatch.setattr(pyhdfe, "create", spy_create)
    return used


# ----------------------------------------------------------------------------
# lsmr_options, which are deprecated
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("options", LSMR_OPTIONS)
def test_lsmr_options_with_the_default_method(data, options, algorithms):
    # A model that HDFE can absorb raised a TypeError from pyhdfe
    mod = make_model(data)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        res = mod.fit(lsmr_options=options)
    assert algorithms == ["lsmr"]
    expected = make_model(data).fit(method="lsmr", absorb_options=options)
    assert_same(res, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_lsmr_options_are_used(data, kind):
    # With 1 iteration LSMR is far from the HDFE solution, so that it is clear that
    # the options and LSMR were used
    mod = make_model(data, kind)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        res = mod.fit(lsmr_options=LIMITED)
    assert_same(res, make_model(data, kind).fit(method="lsmr", absorb_options=LIMITED))
    assert_different(res, make_model(data, kind).fit())


@pytest.mark.parametrize("kind", KINDS)
def test_lsmr_options_with_lsmr(data, kind):
    mod = make_model(data, kind)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        res = mod.fit(method="lsmr", lsmr_options=LIMITED)
    assert_same(res, make_model(data, kind).fit(method="lsmr", absorb_options=LIMITED))


@pytest.mark.parametrize("kind", KINDS)
def test_lsmr_options_with_hdfe(data, kind):
    mod = make_model(data, kind)
    first = mod.fit()
    with pytest.raises(ValueError, match="lsmr_options") as exc_info:
        mod.fit(method="hdfe", lsmr_options=LIMITED)
    assert exc_info.type is ValueError
    message = str(exc_info.value)
    assert 'method="hdfe"' in message
    assert 'method="lsmr"' in message
    # The model can still be used
    assert_same(mod.fit(), first)
    assert_same(mod.fit(), make_model(data, kind).fit())


def test_lsmr_options_and_absorb_options(data):
    mod = make_model(data)
    with pytest.raises(ValueError, match="absorb_options cannot be used"):
        mod.fit(lsmr_options=LIMITED, absorb_options=LIMITED)
    with pytest.raises(ValueError, match="absorb_options cannot be used"):
        mod.fit(method="hdfe", lsmr_options=LIMITED, absorb_options=LIMITED)


@pytest.mark.parametrize("method", ["auto", "lsmr", "hdfe"])
def test_empty_lsmr_options_do_not_select_lsmr(data, method, algorithms):
    # There are no options to pass, so the model is absorbed as without them
    mod = make_model(data)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        res = mod.fit(method=method, lsmr_options={})
    assert algorithms == ["lsmr" if method == "lsmr" else "hdfe"]
    assert_same(res, make_model(data).fit(method=method))


def test_lsmr_options_reuse_the_absorbed_variables(data, count_absorptions):
    # fit compares the arguments that absorb the variables with the previous ones
    mod = make_model(data)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        first = mod.fit(lsmr_options=LIMITED)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        mod.fit(lsmr_options=dict(LIMITED), cov_type="unadjusted")
    # The same options for LSMR in the other way
    mod.fit(method="lsmr", absorb_options=LIMITED)
    assert len(count_absorptions) == 1
    # HDFE absorbs the variables again, and the options are not the same
    default = mod.fit()
    assert len(count_absorptions) == 2
    assert_different(first, default)
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        again = mod.fit(lsmr_options=LIMITED)
    assert len(count_absorptions) == 3
    assert_same(again, first)


# ----------------------------------------------------------------------------
# absorb_options, which are the options of the algorithm that is used
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["auto", "hdfe"])
@pytest.mark.parametrize("options", LSMR_OPTIONS)
def test_lsmr_absorb_options_with_hdfe(data, options, method, algorithms):
    # TypeError: create() got an unexpected keyword argument ...
    mod = make_model(data)
    with pytest.raises(ValueError, match=LSMR_MESSAGE) as exc_info:
        mod.fit(method=method, absorb_options=options)
    assert exc_info.type is ValueError
    message = str(exc_info.value)
    for name in options:
        assert f"'{name}'" in message
    assert 'method="lsmr"' in message
    assert f'method is "{method}"' in message
    # The options were not passed to the algorithm
    assert algorithms == []
    # And the model can be used, with LSMR if that is what was meant
    assert_same(mod.fit(), make_model(data).fit())
    res = mod.fit(method="lsmr", absorb_options=options)
    assert_same(res, make_model(data).fit(method="lsmr", absorb_options=options))


@pytest.mark.parametrize("options", LSMR_OPTIONS)
@pytest.mark.parametrize("kind", ["weighted", "mixed"])
def test_lsmr_absorb_options_that_lsmr_uses(data, kind, options, algorithms):
    # LSMR absorbs weighted models and those with continuous variables
    res = make_model(data, kind).fit(absorb_options=options)
    assert algorithms == ["lsmr"]
    expected = make_model(data, kind).fit(method="lsmr", absorb_options=options)
    assert_same(res, expected)


@pytest.mark.parametrize("options", HDFE_OPTIONS)
@pytest.mark.parametrize(
    ("kind", "method"), [("weighted", "auto"), ("mixed", "auto"), ("hdfe", "lsmr")]
)
def test_hdfe_absorb_options_with_lsmr(data, kind, method, options, algorithms):
    # TypeError: lsmr() got an unexpected keyword argument ...
    mod = make_model(data, kind)
    with pytest.raises(ValueError, match=HDFE_MESSAGE) as exc_info:
        mod.fit(method=method, absorb_options=options)
    assert exc_info.type is ValueError
    message = str(exc_info.value)
    for name in options:
        assert f"'{name}'" in message
    assert algorithms == []
    assert_same(mod.fit(method="lsmr"), make_model(data, kind).fit(method="lsmr"))


@pytest.mark.parametrize("method", ["auto", "hdfe"])
def test_hdfe_options_that_hdfe_uses(data, method, algorithms):
    options = {"residualize_method": "map", "options": {"iteration_limit": 5000}}
    res = make_model(data).fit(method=method, absorb_options=options)
    assert algorithms == ["hdfe"]
    assert_allclose(res.params, make_model(data).fit().params)


@pytest.mark.parametrize("kind", KINDS)
def test_options_that_are_not_options_of_either_algorithm(data, kind):
    # These are still an error from the algorithm that is used
    with pytest.raises(TypeError, match="not_an_option"):
        make_model(data, kind).fit(absorb_options={"not_an_option": 1})


def test_options_are_ignored_without_absorbed_variables(data):
    # There is no algorithm, so there is nothing that the options could be for
    mod = AbsorbingLS(data.y, data.x)
    res = mod.fit(absorb_options=LIMITED)
    assert_same(res, AbsorbingLS(data.y, data.x).fit())
    res = mod.fit(absorb_options={"residualize_method": "map"})
    assert_same(res, AbsorbingLS(data.y, data.x).fit())
