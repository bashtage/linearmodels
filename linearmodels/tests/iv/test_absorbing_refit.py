"""
AbsorbingLS.fit uses the options that it is called with

The variables are absorbed when fit is called for the first time. They were then
kept, and the method, absorb_options, use_cache and lsmr_options of every later
call were ignored. The cache that is shared by all models identified the
residuals by the regressors and the variable only, so a model that used other
LSMR options could get the residuals of a model that used different options.
"""

from linearmodels.compat.scipy import csc_array

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import absorbing
from linearmodels.iv.absorbing import (
    _VARIABLE_CACHE,
    AbsorbingLS,
    clear_cache,
    lsmr_annihilate,
)
from linearmodels.panel.utility import AbsorbingEffectWarning
from linearmodels.shared.utility import AttrDict

NOBS = 1000
KINDS = ["hdfe", "weighted", "mixed"]
# Options with an effect on the estimates: LSMR stops after one iteration
LIMITED = {"method": "lsmr", "absorb_options": {"maxiter": 1}}
REFIT_ARGS = [
    LIMITED,
    {"method": "lsmr", "absorb_options": {"maxiter": 2}, "use_cache": False},
    {"method": "lsmr", "absorb_options": {"maxiter": 3, "damp": 0.1}},
]


@pytest.fixture(autouse=True)
def empty_cache():
    # The cache is shared by all models and so by all tests
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    cat = pd.DataFrame(
        {
            "g": pd.Categorical(rng.integers(0, 40, NOBS)),
            "h": pd.Categorical(rng.integers(0, 30, NOBS)),
        }
    )
    cont = pd.DataFrame({"z": rng.standard_normal(NOBS)})
    x = pd.DataFrame(rng.standard_normal((NOBS, 2)), columns=["x0", "x1"])
    x["x0"] += 0.05 * cat["g"].cat.codes + 0.05 * cat["h"].cat.codes
    y = pd.Series(rng.standard_normal(NOBS) + x.sum(axis=1), name="y")
    weights = pd.Series(np.exp(0.5 * rng.standard_normal(NOBS)), name="w")
    clusters = rng.integers(0, 25, NOBS)
    return AttrDict(y=y, x=x, cat=cat, cont=cont, weights=weights, clusters=clusters)


def make_model(data, kind, **kwargs):
    """HDFE can absorb the first, and only LSMR can absorb the others"""
    if kind == "hdfe":
        return AbsorbingLS(data.y, data.x, absorb=data.cat, **kwargs)
    if kind == "weighted":
        return AbsorbingLS(
            data.y, data.x, absorb=data.cat, weights=data.weights, **kwargs
        )
    absorb = pd.concat([data.cat, data.cont], axis=1)
    return AbsorbingLS(data.y, data.x, absorb=absorb, **kwargs)


def assert_same(res, expected):
    assert res.df_model == expected.df_model
    assert list(res.params.index) == list(expected.params.index)
    series = ["params", "std_errors", "cov", "resids", "fitted_values"]
    series += ["absorbed_effects"]
    for name in series:
        assert_allclose(
            getattr(res, name).to_numpy(),
            getattr(expected, name).to_numpy(),
            rtol=1e-10,
            atol=1e-12,
            err_msg=name,
        )
    for name in ("rsquared", "absorbed_rsquared", "resid_ss", "s2"):
        assert_allclose(getattr(res, name), getattr(expected, name), rtol=1e-10)
    assert_allclose(res.f_statistic.stat, expected.f_statistic.stat, rtol=1e-10)


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


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("args", REFIT_ARGS)
def test_refit_matches_new_model(data, kind, args):
    mod = make_model(data, kind)
    first = mod.fit()
    refit = mod.fit(**args)
    expected = make_model(data, kind).fit(**args)
    assert_same(refit, expected)
    assert_different(refit, first)
    # The first results are not modified by the second fit
    assert_same(first, make_model(data, kind).fit())


@pytest.mark.parametrize("kind", KINDS)
def test_refit_to_defaults(data, kind):
    mod = make_model(data, kind)
    limited = mod.fit(**LIMITED)
    default = mod.fit()
    assert_same(default, make_model(data, kind).fit())
    assert_different(default, limited)
    # And back to the limited options
    assert_same(mod.fit(**LIMITED), limited)


@pytest.mark.parametrize("kind", KINDS)
def test_refit_does_not_change_degrees_of_freedom(data, kind):
    mod = make_model(data, kind)
    df_model = mod.fit().df_model
    for args in REFIT_ARGS + [{}, LIMITED]:
        assert mod.fit(**args).df_model == df_model


def test_same_arguments_use_absorbed_variables(data, count_absorptions):
    mod = make_model(data, "weighted")
    mod.fit()
    absorbed = mod.absorbed_dependent
    assert len(count_absorptions) == 1
    # The covariance estimator does not change what is absorbed
    other_arguments = [
        {"cov_type": "unadjusted"},
        {"debiased": True},
        {"cov_type": "clustered", "clusters": data.clusters},
        {"method": "auto"},
        {"use_cache": True},
        {"absorb_options": {}},
        {"absorb_options": None},
    ]
    for args in other_arguments:
        mod.fit(**args)
        assert len(count_absorptions) == 1
        assert mod.absorbed_dependent is absorbed

    mod.fit(absorb_options={"atol": 1e-7})
    assert len(count_absorptions) == 2
    # Equal options in a different dict, and an option that only controls output
    mod.fit(absorb_options={"atol": 1e-7})
    mod.fit(absorb_options={"atol": 1e-7, "show": False})
    assert len(count_absorptions) == 2

    mod.fit(absorb_options={"atol": 1e-7}, use_cache=False)
    assert len(count_absorptions) == 3
    mod.fit(absorb_options={"atol": 1e-7}, use_cache=False)
    assert len(count_absorptions) == 3
    mod.fit(absorb_options={"atol": 1e-7})
    assert len(count_absorptions) == 4


def test_options_changed_by_caller(data):
    mod = make_model(data, "weighted")
    options = {"maxiter": 1}
    first = mod.fit(absorb_options=options)
    options["maxiter"] = 3
    second = mod.fit(absorb_options=options)
    assert_same(first, make_model(data, "weighted").fit(absorb_options={"maxiter": 1}))
    assert_same(second, make_model(data, "weighted").fit(absorb_options={"maxiter": 3}))
    assert_different(first, second)


def test_lsmr_options(data):
    mod = make_model(data, "weighted")
    first = mod.fit()
    with pytest.warns(FutureWarning, match="lsmr_options is deprecated"):
        refit = mod.fit(lsmr_options={"maxiter": 1})
    expected = make_model(data, "weighted").fit(absorb_options={"maxiter": 1})
    assert_same(refit, expected)
    assert_different(refit, first)
    with pytest.raises(ValueError, match="cannot be used with lsmr_options"):
        mod.fit(lsmr_options={"maxiter": 2}, absorb_options={"maxiter": 2})
    # The error does not change what was absorbed
    assert_same(mod.fit(absorb_options={"maxiter": 1}), expected)


def test_hdfe_options(data, count_absorptions):
    mod = make_model(data, "hdfe")
    first = mod.fit()
    # The options of the algorithm are in a dict, which is part of the options
    options = {"options": {"iteration_limit": 5000}}
    refit = mod.fit(absorb_options=options)
    assert len(count_absorptions) == 2
    # The estimates are the same since both converge
    assert_allclose(refit.params, first.params)
    # The same options in another dict, and then other options
    mod.fit(absorb_options={"options": {"iteration_limit": 5000}})
    assert len(count_absorptions) == 2
    mod.fit(absorb_options={"options": {"iteration_limit": 4000}})
    assert len(count_absorptions) == 3
    assert_same(refit, make_model(data, "hdfe").fit(absorb_options=options))
    # A limit that is too low is an error in a model that has been fit
    with pytest.raises(RuntimeError, match="converge"):
        mod.fit(absorb_options={"options": {"iteration_limit": 1}})


def test_method_that_cannot_be_used(data):
    # A new model raises, and so must a model that has been fit
    mod = make_model(data, "weighted")
    first = mod.fit()
    with pytest.raises(RuntimeError, match="HDFE has been set as the method"):
        mod.fit(method="hdfe")
    # The model can still be used
    assert_same(mod.fit(), first)


def test_failed_absorption_does_not_change_the_model(data):
    mod = make_model(data, "weighted")
    first = mod.fit()
    with pytest.raises(TypeError, match="not_an_option"):
        mod.fit(absorb_options={"not_an_option": 1})
    assert_same(mod.fit(), first)
    assert_same(mod.fit(**LIMITED), make_model(data, "weighted").fit(**LIMITED))


def test_failed_first_absorption_does_not_change_the_model(data):
    # Degrees of freedom of the model used to include those of the failed fit
    mod = make_model(data, "weighted")
    with pytest.raises(TypeError, match="not_an_option"):
        mod.fit(absorb_options={"not_an_option": 1})
    with pytest.raises(RuntimeError, match="fit must be called once"):
        _ = mod.absorbed_dependent
    assert_same(mod.fit(), make_model(data, "weighted").fit())


def test_drop_absorbed_refit(data):
    # An effect that absorbs a regressor, which is dropped each time. It is the
    # first column, so that the positions of the columns that are retained are
    # not the positions of the columns that are left after the first fit.
    dummy = pd.get_dummies(data.cat["g"], dtype=float).iloc[:, [3]]
    x = pd.concat([dummy.add_prefix("g"), data.x], axis=1)

    def model():
        return AbsorbingLS(data.y, x, absorb=data.cat[["g"]], drop_absorbed=True)

    mod = model()
    with pytest.warns(AbsorbingEffectWarning):
        first = mod.fit()
    assert list(first.params.index) == ["x0", "x1"]
    for args in (LIMITED, {}, {"method": "lsmr"}):
        with pytest.warns(AbsorbingEffectWarning):
            refit = mod.fit(**args)
        with pytest.warns(AbsorbingEffectWarning):
            expected = model().fit(**args)
        assert list(refit.params.index) == ["x0", "x1"]
        assert_same(refit, expected)
    with pytest.warns(AbsorbingEffectWarning):
        assert_same(mod.fit(), first)


def test_cache_identifies_options(data, monkeypatch):
    calls = []
    original = absorbing.lsmr

    def counting_lsmr(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(absorbing, "lsmr", counting_lsmr)
    default = make_model(data, "weighted").fit()
    ncalls = len(calls)
    assert ncalls > 0
    assert len(_VARIABLE_CACHE) == 1

    # Not the residuals of the default options, which are in the cache
    limited = make_model(data, "weighted").fit(absorb_options={"maxiter": 1})
    assert len(calls) == 2 * ncalls
    assert len(_VARIABLE_CACHE) == 2
    reference = make_model(data, "weighted").fit(
        absorb_options={"maxiter": 1}, use_cache=False
    )
    assert_same(limited, reference)
    assert_different(limited, default)

    # The residuals of both are in the cache
    ncalls = len(calls)
    assert_same(make_model(data, "weighted").fit(), default)
    assert_same(
        make_model(data, "weighted").fit(absorb_options={"maxiter": 1}), limited
    )
    assert len(calls) == ncalls
    assert len(_VARIABLE_CACHE) == 2
    # Writing the default as an option does not change the key
    make_model(data, "weighted").fit(absorb_options={"atol": 1e-8, "btol": 1e-8})
    assert len(calls) == ncalls
    assert len(_VARIABLE_CACHE) == 2


def test_cache_is_not_used_when_not_requested(data):
    make_model(data, "weighted").fit(use_cache=False)
    assert len(_VARIABLE_CACHE) == 0
    make_model(data, "weighted").fit(use_cache=True)
    assert len(_VARIABLE_CACHE) == 1


@pytest.mark.parametrize("use_cache", [True, False])
def test_lsmr_annihilate_cache_with_array_options(monkeypatch, use_cache):
    calls = []
    original = absorbing.lsmr

    def counting_lsmr(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(absorbing, "lsmr", counting_lsmr)
    rng = np.random.default_rng(1)
    nobs, ncol = 100, 8
    x = csc_array(np.eye(ncol)[rng.integers(0, ncol, nobs)])
    y = rng.standard_normal((nobs, 2))

    def annihilate(**options):
        """The residuals and the number of times that lsmr was used"""
        before = len(calls)
        resids = lsmr_annihilate(x, y, use_cache, "x-hash", **options)
        return resids, len(calls) - before

    # One use of lsmr for each column of y, unless the cache has the residuals
    computed = 2
    cached = 0 if use_cache else computed
    zeros = np.zeros(ncol)
    first, ncalls = annihilate(x0=zeros)
    assert ncalls == computed
    # The same values in another array
    assert annihilate(x0=np.zeros(ncol))[1] == cached
    # Other values
    assert annihilate(x0=np.arange(ncol) / 100.0)[1] == computed
    again, ncalls = annihilate(x0=zeros)
    assert ncalls == cached
    assert_allclose(again, first)
    assert len(_VARIABLE_CACHE) == (2 if use_cache else 0)


def test_options_key():
    from linearmodels.iv.absorbing import _options_key  # noqa: PLC0415

    key = _options_key(
        {"show": True, "options": {"b": np.arange(3), "a": [1, 2]}, "atol": 1e-8}
    )
    # Hashable, and independent of order, of the type of the array and of show
    assert hash(key) == hash(
        _options_key(
            {"atol": 1e-8, "options": {"a": (1, 2), "b": np.arange(3)}, "show": False}
        )
    )
    assert key == _options_key(
        {"atol": 1e-8, "options": {"a": (1, 2), "b": np.arange(3)}}
    )
    assert key != _options_key({"atol": 1e-8, "options": {"a": [1, 2], "b": [0, 1, 3]}})
    assert key != _options_key({"atol": 1e-9, "options": {"a": [1, 2], "b": [0, 1, 2]}})
    assert _options_key({}) == ()
