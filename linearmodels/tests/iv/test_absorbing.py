from linearmodels.compat.statsmodels import Summary

from itertools import product
import struct
from typing import Optional

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
import scipy.sparse as sp
from scipy.sparse import csc_matrix

from linearmodels.iv._utility import annihilate
from linearmodels.iv.absorbing import (
    _VARIABLE_CACHE,
    AbsorbingLS,
    AbsorbingRegressor,
    Interaction,
    category_continuous_interaction,
    category_interaction,
    category_product,
    clear_cache,
    lsmr_annihilate,
)
from linearmodels.iv.model import _OLS
from linearmodels.iv.results import AbsorbingLSResults, OLSResults
from linearmodels.panel.utility import (
    AbsorbingEffectError,
    AbsorbingEffectWarning,
    dummy_matrix,
)
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.shared.utility import AttrDict

NOBS = 100
pytestmark = pytest.mark.filterwarnings(
    "ignore:the matrix subclass:PendingDeprecationWarning"
)


class Hasher:
    @property
    def hash_func(self):
        try:
            import xxhash

            return xxhash.xxh64()
        except ImportError:
            import hashlib

            return hashlib.sha256()

    def single(self, value):
        h = self.hash_func
        h.update(np.ascontiguousarray(value))
        return h.hexdigest()


hasher = Hasher()


@pytest.fixture(scope="function")
def random_gen(request):
    return np.random.RandomState(12345678)


def random_cat(ncat, size, frame=False, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    series = pd.Series(pd.Categorical(rs.randint(0, ncat, size, dtype=np.int8)))
    if frame:
        return pd.DataFrame(series)
    return series


def random_cont(size, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    series = pd.Series(rs.standard_normal(size))
    return pd.DataFrame(series)


@pytest.fixture(scope="module", params=[1, 2, 3])
def cat(request):
    rs = np.random.RandomState(0)
    return pd.DataFrame(
        {str(i): random_cat(4, NOBS, rs=rs) for i in range(request.param)}
    )


@pytest.fixture(scope="module", params=[1, 2])
def cont(request):
    rs = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "cont" + str(i): pd.Series(rs.standard_normal(NOBS))
            for i in range(request.param)
        }
    )


@pytest.fixture(scope="module", params=[True, False])
def weights(request):
    if not request.param:
        return None
    rs = np.random.RandomState(0)
    return rs.chisquare(10, NOBS) / 10.0


@pytest.fixture(scope="module", params=[0, 1, 2])
def interact(request):
    if not request.param:
        return None
    rs = np.random.RandomState(0)
    interactions = []
    for _ in range(request.param):
        cat = random_cat(4, 100, frame=True, rs=rs)
        cont = random_cont(100, rs=rs)
        interactions.append(Interaction(cat, cont))
    return interactions


def generate_data(
    k=3,
    const=True,
    nfactors=1,
    factor_density=10,
    nobs=2000,
    cont_interactions=1,
    factor_format="interaction",
    singleton_interaction=False,
    weighted=False,
    ncont=0,
):
    rs = np.random.RandomState(1234567890)
    density = [factor_density] * max(nfactors, cont_interactions)
    x = rs.standard_normal((nobs, k))
    if const:
        x = np.column_stack([np.ones(nobs), x])
    e = rs.standard_normal(nobs)
    y = x.sum(1) + e

    factors = []
    for i in range(nfactors):
        ncat = nobs // density[min(i, len(density) - 1)]
        fact = rs.randint(ncat, size=nobs)
        effects = rs.standard_normal(ncat)
        y += effects[fact]
        factors.append(pd.Series(pd.Categorical(fact)))
    conts = [pd.Series(rs.standard_normal(size=nobs)) for _ in range(ncont)]
    factors.extend(conts)

    if factors:
        factors = pd.concat(factors, axis=1)
        if factor_format == "interaction":
            if nfactors and ncont:
                factors = Interaction(
                    factors.iloc[:, :nfactors], factors.iloc[:, nfactors:]
                )
            elif nfactors:
                factors = Interaction(factors, None)
            else:
                factors = Interaction(None, factors)
    else:
        factors = None

    interactions = []
    for i in range(cont_interactions):
        ncat = nobs // density[min(i, len(density) - 1)]
        fact = rs.randint(ncat, size=nobs)
        effects = rs.standard_normal(nobs)
        y += effects
        df = pd.DataFrame(pd.Series(pd.Categorical(fact)), columns=[f"fact{i}"])
        df_eff = pd.DataFrame(effects[:, None], columns=[f"effect_{i}"])
        interactions.append(Interaction(df, df_eff))
    if factor_format == "pandas":
        for i, interact in enumerate(interactions):
            interactions[i] = pd.concat([interact.cat, interact.cont], axis=1)
    interactions = interactions if interactions else None
    if interactions and singleton_interaction:
        interactions = interactions[0]
    if weighted:
        weights = pd.DataFrame(rs.chisquare(10, size=(nobs, 1)) / 10)
    else:
        weights = None

    return AttrDict(
        y=y, x=x, absorb=factors, interactions=interactions, weights=weights
    )


# Permutations:
# k in (0,3), const in (True,False), factors=(0,1,2), interactions in (0,1)


# k=3, const=True, nfactors=1, factor_density=10, nobs=2000, cont_interactions=1,
#                   format='interaction', singleton_interaction=False
configs = product(
    [0, 3],  # k
    [False, True],  # constant
    [1, 2, 0],  # factors
    [10],  # density
    [2000],  # nobs
    [0, 1],  # cont interactions
    ["interaction", "pandas"],  # format
    [False, True],  # singleton
    [False, True],  # weighted
    [0, 1],  # ncont
)

data_configs = [c for c in configs if (c[2] or c[5] or c[9])]
id_str = (
    "k: {0}, const: {1}, nfactors: {2}, density: {3}, nobs: {4}, "
    "cont_interacts: {5}, format:{6}, singleton:{7}, weighted: {8}, ncont: {9}"
)
data_ids = [id_str.format(*config) for config in configs]


@pytest.fixture(scope="module", params=data_configs, ids=data_ids)
def data(request):
    return generate_data(*request.param)


configs_ols = product(
    [0, 3],  # k
    [False, True],  # constant
    [1, 2, 0],  # factors
    [50],  # density
    [500],  # nobs
    [0, 1],  # cont interactions
    ["interaction"],  # format
    [False],  # singleton
    [False, True],  # weighted
    [0, 1],  # ncont
)

configs_ols_data = [c for c in configs_ols if (c[0] or c[1])]
id_str = (
    "k: {0}, const: {1}, nfactors: {2}, density: {3}, nobs: {4}, "
    "cont_interacts: {5}, format:{6}, singleton:{7}, weighted: {8}, ncont: {9}"
)
ids_ols_data = [id_str.format(*config) for config in configs_ols]


@pytest.fixture(scope="module", params=configs_ols_data, ids=ids_ols_data)
def ols_data(request):
    return generate_data(*request.param)


@pytest.mark.smoke
def test_smoke(data):
    mod = AbsorbingLS(
        data.y,
        data.x,
        absorb=data.absorb,
        interactions=data.interactions,
        weights=data.weights,
    )
    res = mod.fit()
    assert isinstance(res.summary, Summary)
    assert isinstance(str(res.summary), str)


def test_absorbing_exceptions(random_gen):
    with pytest.raises(TypeError):
        absorbed = random_gen.standard_normal((NOBS, 2))
        assert isinstance(absorbed, np.ndarray)
        AbsorbingLS(
            random_gen.standard_normal(NOBS),
            random_gen.standard_normal((NOBS, 2)),
            absorb=absorbed,
        )
    with pytest.raises(ValueError):
        AbsorbingLS(
            random_gen.standard_normal(NOBS), random_gen.standard_normal((NOBS - 1, 2))
        )
    with pytest.raises(ValueError):
        AbsorbingLS(
            random_gen.standard_normal(NOBS),
            random_gen.standard_normal((NOBS, 2)),
            absorb=pd.DataFrame(random_gen.standard_normal((NOBS - 1, 1))),
        )
    with pytest.raises(ValueError):
        AbsorbingLS(
            random_gen.standard_normal(NOBS),
            random_gen.standard_normal((NOBS, 2)),
            interactions=random_cat(10, NOBS - 1, frame=True, rs=random_gen),
        )
    mod = AbsorbingLS(
        random_gen.standard_normal(NOBS),
        random_gen.standard_normal((NOBS, 2)),
        interactions=random_cat(10, NOBS, frame=True, rs=random_gen),
    )
    with pytest.raises(RuntimeError):
        assert isinstance(mod.absorbed_dependent, pd.DataFrame)
    with pytest.raises(RuntimeError):
        assert isinstance(mod.absorbed_exog, pd.DataFrame)
    with pytest.raises(TypeError):
        interactions = random_gen.randint(0, 10, size=(NOBS, 2))
        assert isinstance(interactions, np.ndarray)
        AbsorbingLS(
            random_gen.standard_normal(NOBS),
            random_gen.standard_normal((NOBS, 2)),
            interactions=interactions,
        )


def test_clear_cache():
    _VARIABLE_CACHE["key"] = {"a": np.empty(100)}
    clear_cache()
    assert len(_VARIABLE_CACHE) == 0


def test_category_product(cat):
    prod = category_product(cat)
    if cat.shape[1] == 1:
        assert_series_equal(prod, cat.iloc[:, 0], check_names=False)
    else:
        alt = cat.iloc[:, 0].astype("int64")
        for i in range(1, cat.shape[1]):
            alt += 10 ** (4 * i) * cat.iloc[:, i].astype("int64")
        alt = pd.Categorical(alt)
        alt = pd.Series(alt)
        df = pd.DataFrame([prod.cat.codes, alt.cat.codes], index=["cat_prod", "alt"]).T
        g = df.groupby("cat_prod").alt
        assert (g.nunique() == 1).all()
        g = df.groupby("alt").cat_prod
        assert (g.nunique() == 1).all()


@pytest.mark.parametrize("ncat", [5, 10])
def test_category_product_large(random_gen, ncat):
    dfc = {}
    for i in range(ncat):
        dfc[str(i)] = random_cat(10, 1000)
    cat = pd.DataFrame(dfc)
    out = category_product(cat)
    bits = 64 if np.log2(10**ncat) > 32 else 32
    max_size = 64 if np.log2(out.cat.categories.max()) > 32 else 32
    assert bits == max_size


def test_category_product_too_large(random_gen):
    dfc = {}
    for i in range(20):
        dfc[str(i)] = random_cat(10, 1000)
    cat = pd.DataFrame(dfc)
    with pytest.raises(ValueError):
        category_product(cat)


def test_category_product_not_cat(random_gen):
    cat = pd.DataFrame(
        {str(i): pd.Series(random_gen.randint(0, 10, 1000)) for i in range(3)}
    )
    with pytest.raises(TypeError):
        category_product(cat)


def test_category_interaction():
    c = pd.Series(pd.Categorical([0, 0, 0, 1, 1, 1]))
    actual = category_interaction(c, precondition=False).toarray()
    expected = np.zeros((6, 2))
    expected[:3, 0] = 1.0
    expected[3:, 1] = 1.0
    assert_allclose(actual, expected)

    actual = category_interaction(c, precondition=True).toarray()
    cond = np.sqrt((expected**2).sum(0))
    expected /= cond
    assert_allclose(actual, expected)


def test_category_continuous_interaction():
    c = pd.Series(pd.Categorical([0, 0, 0, 1, 1, 1]))
    v = pd.Series(np.arange(6.0))
    actual = category_continuous_interaction(c, v, precondition=False)
    expected = np.zeros((6, 2))
    expected[:3, 0] = v[:3]
    expected[3:, 1] = v[3:]

    assert_allclose(actual.toarray(), expected)

    actual = category_continuous_interaction(c, v, precondition=True)
    cond = np.sqrt((expected**2).sum(0))
    expected /= cond
    assert_allclose(actual.toarray(), expected)


def test_category_continuous_interaction_interwoven():
    c = pd.Series(pd.Categorical([0, 1, 0, 1, 0, 1]))
    v = pd.Series(np.arange(6.0))
    actual = category_continuous_interaction(c, v, precondition=False)
    expected = np.zeros((6, 2))
    expected[::2, 0] = v[::2]
    expected[1::2, 1] = v[1::2]
    assert_allclose(actual.toarray(), expected)


def test_interaction_cat_only(cat):
    interact = Interaction(cat=cat)
    assert interact.nobs == cat.shape[0]
    assert_frame_equal(cat, interact.cat)
    expected = category_interaction(category_product(cat), precondition=False)
    actual = interact.sparse
    assert isinstance(actual, csc_matrix)
    assert_allclose(expected.toarray(), actual.toarray())


def test_interaction_cont_only(cont):
    interact = Interaction(cont=cont)
    assert interact.nobs == cont.shape[0]
    assert_frame_equal(cont, interact.cont)
    expected = cont.to_numpy()
    actual = interact.sparse
    assert isinstance(actual, csc_matrix)
    assert_allclose(expected, actual.toarray())


def test_interaction_cat_cont(cat, cont):
    interact = Interaction(cat=cat, cont=cont)
    assert interact.nobs == cat.shape[0]
    assert_frame_equal(cat, interact.cat)
    assert_frame_equal(cont, interact.cont)
    base = category_interaction(category_product(cat), precondition=False).toarray()
    expected = []
    for i in range(cont.shape[1]):
        element = base.copy()
        element[np.where(element)] = cont.iloc[:, i].to_numpy()
        expected.append(element)
    expected = np.column_stack(expected)
    actual = interact.sparse
    assert isinstance(actual, csc_matrix)
    assert_allclose(expected, interact.sparse.toarray())


def test_interaction_from_frame(cat, cont):
    base = Interaction(cat=cat, cont=cont)
    interact = Interaction.from_frame(pd.concat([cat, cont], axis=1))
    assert_allclose(base.sparse.toarray(), interact.sparse.toarray())


def test_interaction_cat_bad_nobs():
    with pytest.raises(ValueError):
        Interaction()
    with pytest.raises(ValueError):
        Interaction(cat=np.empty((100, 0)), cont=np.empty((100, 0)))


def test_empty_interaction():
    interact = Interaction(nobs=100)
    assert isinstance(interact.sparse, csc_matrix)
    assert interact.sparse.shape == (100, 0)


def test_interaction_cat_cont_convert(cat, cont):
    base = Interaction(cat, cont)
    interact = Interaction(cat.to_numpy(), cont)
    assert_allclose(base.sparse.toarray(), interact.sparse.toarray())


def test_absorbing_regressors(cat, cont, interact, weights):
    areg = AbsorbingRegressor(
        cat=cat, cont=cont, interactions=interact, weights=weights
    )
    rank = areg.approx_rank
    expected_rank = 0

    expected = []
    for i, col in enumerate(cat):
        expected_rank += pd.Series(cat[col].cat.codes).nunique() - (i > 0)
    expected.append(dummy_matrix(cat, precondition=False)[0])
    expected_rank += cont.shape[1]
    expected.append(csc_matrix(cont))
    if interact is not None:
        for inter in interact:
            interact_mat = inter.sparse
            expected_rank += interact_mat.shape[1]
            expected.append(interact_mat)
    expected = sp.hstack(expected, format="csc")
    if weights is not None:
        expected = (sp.diags(np.sqrt(weights)).dot(expected)).asformat("csc")
    actual = areg.regressors
    assert expected.shape == actual.shape
    assert_array_equal(expected.indptr, actual.indptr)
    assert_array_equal(expected.indices, actual.indices)
    assert_allclose(expected.toarray(), actual.toarray())
    assert expected_rank == rank


def test_absorbing_regressors_hash(cat, cont, interact, weights):
    areg = AbsorbingRegressor(
        cat=cat, cont=cont, interactions=interact, weights=weights
    )
    # Build hash
    hashes = []
    for col in cat:
        hashes.append((hasher.single(cat[col].cat.codes.to_numpy().data),))
    for col in cont:
        hashes.append((hasher.single(cont[col].to_numpy().data),))
    hashes = sorted(hashes)
    if interact is not None:
        for inter in interact:
            hashes.extend(inter.hash)
    if weights is not None:
        hashes.append((hasher.single(weights.data),))
    hashes = tuple(sorted(hashes))
    assert hashes == areg.hash


def test_empty_absorbing_regressor():
    areg = AbsorbingRegressor()
    assert areg.regressors.shape == (0, 0)
    assert areg.hash == tuple()


def test_against_ols(ols_data):
    mod = AbsorbingLS(
        ols_data.y,
        ols_data.x,
        absorb=ols_data.absorb,
        interactions=ols_data.interactions,
        weights=ols_data.weights,
    )
    res = mod.fit()
    absorb = []
    has_dummy = False
    if ols_data.absorb is not None:
        absorb.append(ols_data.absorb.cont.to_numpy())
        if ols_data.absorb.cat.shape[1] > 0:
            dummies = dummy_matrix(ols_data.absorb.cat, precondition=False)[0]
            assert isinstance(dummies, sp.csc_matrix)
            absorb.append(dummies.toarray())
        has_dummy = ols_data.absorb.cat.shape[1] > 0
    if ols_data.interactions is not None:
        for interact in ols_data.interactions:
            absorb.append(interact.sparse.toarray())
    _x = ols_data.x
    if absorb:
        absorb = np.column_stack(absorb)
        if np.any(np.ptp(_x, 0) == 0) and has_dummy:
            if ols_data.weights is None:
                absorb = annihilate(absorb, np.ones((absorb.shape[0], 1)))
            else:
                root_w = np.sqrt(mod.weights.ndarray)
                wabsorb = annihilate(root_w * absorb, root_w)
                absorb = (1.0 / root_w) * wabsorb
        rank = np.linalg.matrix_rank(absorb)
        if rank < absorb.shape[1]:
            a, b = np.linalg.eig(absorb.T @ absorb)
            order = np.argsort(a)[::-1]
            a, b = a[order], b[:, order]
            z = absorb @ b
            absorb = z[:, :rank]
        _x = np.column_stack([_x, absorb])
    ols_mod = _OLS(ols_data.y, _x, weights=ols_data.weights)
    ols_res = ols_mod.fit()

    assert_results_equal(ols_res, res)


def test_cache():
    # Clear the cache to avoid side effects and order dependency
    _VARIABLE_CACHE.clear()
    gen = generate_data(
        2, True, 2, factor_format="pandas", ncont=0, cont_interactions=1
    )
    first = len(_VARIABLE_CACHE)
    mod = AbsorbingLS(
        gen.y, gen.x, absorb=gen.absorb.iloc[:, :1], interactions=gen.interactions
    )
    mod.fit()
    second = len(_VARIABLE_CACHE)
    mod = AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)
    mod.fit()
    third = len(_VARIABLE_CACHE)
    assert third - second == 1
    assert second - first == 1
    mod = AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)
    mod.fit()
    fourth = len(_VARIABLE_CACHE)
    assert fourth - third == 0
    # Clear the cache to avoid side effects and order dependency
    _VARIABLE_CACHE.clear()


def test_instrments():
    gen = generate_data(
        2, True, 2, factor_format="pandas", ncont=0, cont_interactions=1
    )
    mod = AbsorbingLS(
        gen.y, gen.x, absorb=gen.absorb.iloc[:, :1], interactions=gen.interactions
    )
    assert mod.instruments.shape[1] == 0


def assert_results_equal(
    o_res: OLSResults, a_res: AbsorbingLSResults, k: Optional[int] = None
) -> None:
    if k is None:
        k = a_res.params.shape[0]
    attrs = [v for v in dir(o_res) if not v.startswith("_")]
    callables = ["conf_int"]
    skip = [
        "summary",
        "test_linear_constraint",
        "predict",
        "model",
        "f_statistic",
        "wald_test",
        "method",
        "kappa",
    ]
    for attr in attrs:
        if attr in skip:
            continue
        left = getattr(o_res, attr)
        right = getattr(a_res, attr)
        if attr in callables:
            left = left()
            right = right()
        if isinstance(left, np.ndarray):
            raise NotImplementedError
        elif isinstance(left, pd.DataFrame):
            if attr == "conf_int":
                left = left.iloc[:k]
            elif attr == "cov":
                left = left.iloc[:k, :k]
            assert_allclose(left, right, rtol=2e-4, atol=1e-6)
        elif isinstance(left, pd.Series):
            assert_allclose(left.iloc[:k], right.iloc[:k], rtol=1e-5)
        else:
            if isinstance(left, float):
                assert_allclose(left, right, atol=1e-10)
            else:
                assert left == right
    assert isinstance(a_res.summary, Summary)
    assert isinstance(str(a_res.summary), str)
    assert isinstance(a_res.absorbed_effects, pd.DataFrame)
    tol = 1e-4 if (8 * struct.calcsize("P")) < 64 else 1e-8
    assert a_res.absorbed_rsquared <= (a_res.rsquared + tol)


def test_center_cov_arg():
    gen = generate_data(
        2, True, 2, factor_format="pandas", ncont=0, cont_interactions=1
    )
    mod = AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)
    res = mod.fit(center=True)
    assert "center" not in res.cov_config


def test_drop_missing():
    gen = generate_data(
        2, True, 2, factor_format="pandas", ncont=0, cont_interactions=1
    )
    gen.y[::53] = np.nan
    gen.x[::79] = np.nan
    with pytest.warns(MissingValueWarning):
        AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)

    gen = generate_data(
        2, True, 2, factor_format="pandas", ncont=0, cont_interactions=1
    )
    for col in gen.absorb:
        gen.absorb[col] = gen.absorb[col].astype("int64").astype("object")
        col_iloc = gen.absorb.columns.get_loc(col)
        gen.absorb.iloc[::91, col_iloc] = np.nan
        gen.absorb[col] = pd.Categorical(gen.absorb[col].to_numpy())
    with pytest.warns(MissingValueWarning):
        AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)


def test_drop_absorb(random_gen):
    absorb = random_gen.randint(0, 10, size=1000)
    x = random_gen.standard_normal((1000, 3))
    y = random_gen.standard_normal(1000)
    dfd = {f"x{i}": pd.Series(x[:, i]) for i in range(3)}
    dfd.update({"c": pd.Series(absorb, dtype="category"), "y": pd.Series(y)})
    df = pd.DataFrame(dfd)

    y = df.y
    x = df.iloc[:, :3]
    x = pd.concat([x, pd.get_dummies(df.c).iloc[:, :2]], axis=1)
    mod = AbsorbingLS(y, x, absorb=df[["c"]], drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning):
        res = mod.fit()
    assert len(res.params) == 3
    assert all(f"x{i}" in res.params for i in range(3))
    assert isinstance(str(res.summary), str)
    mod = AbsorbingLS(y, x, absorb=df[["c"]])
    with pytest.raises(AbsorbingEffectError):
        mod.fit()
    mod = AbsorbingLS(y, x.iloc[:, -2:], absorb=df[["c"]])
    with pytest.raises(AbsorbingEffectError):
        mod.fit()


def test_fully_absorb(random_gen):
    absorb = random_gen.randint(0, 10, size=1000)
    x = random_gen.standard_normal((1000, 3))
    y = random_gen.standard_normal(1000)
    dfd = {f"x{i}": pd.Series(x[:, i]) for i in range(3)}
    dfd.update({"c": pd.Series(absorb, dtype="category"), "y": pd.Series(y)})
    df = pd.DataFrame(dfd)

    y = df.y
    x = pd.get_dummies(df.c, drop_first=False)
    mod = AbsorbingLS(y, x, absorb=df[["c"]], drop_absorbed=True)
    with pytest.raises(ValueError, match="All columns in exog"):
        mod.fit()


def test_lsmr_options(random_gen):
    absorb = random_gen.randint(0, 10, size=1000)
    x = random_gen.standard_normal((1000, 3))
    y = random_gen.standard_normal(1000)
    dfd = {f"x{i}": pd.Series(x[:, i]) for i in range(3)}
    dfd.update({"c": pd.Series(absorb, dtype="category"), "y": pd.Series(y)})
    df = pd.DataFrame(dfd)

    y = df.y
    x = df.iloc[:, :3]
    mod = AbsorbingLS(y, x, absorb=df[["c"]], drop_absorbed=True)
    with pytest.warns(FutureWarning, match="lsmr_options"):
        mod.fit(lsmr_options={})
    with pytest.raises(ValueError, match="absorb_options cannot"):
        mod.fit(lsmr_options={}, absorb_options={})


def test_options(random_gen):
    absorb = random_gen.randint(0, 10, size=1000)
    x = random_gen.standard_normal((1000, 3))
    y = random_gen.standard_normal(1000)
    dfd = {f"x{i}": pd.Series(x[:, i]) for i in range(3)}
    dfd.update({"c": pd.Series(absorb, dtype="category"), "y": pd.Series(y)})
    df = pd.DataFrame(dfd)

    y = df.y
    x = df.iloc[:, :3]
    mod = AbsorbingLS(y, x, absorb=df[["c"]], drop_absorbed=True)
    mod.fit(absorb_options={"drop_singletons": False})
    mod.fit(absorb_options={"atol": 1e-7, "btol": 1e-7}, method="lsmr")

    mod = AbsorbingLS(y, x[["x0", "x1"]], absorb=df[["x2", "c"]], drop_absorbed=True)
    with pytest.raises(RuntimeError, match="HDFE has been"):
        mod.fit(absorb_options={"atol": 1e-7, "btol": 1e-7}, method="hdfe")


def test_lsmr_annihilate_empty():
    gen = np.random.default_rng(0)
    x = csc_matrix(gen.standard_normal((1000, 2)))
    y = np.empty((1000, 0))
    y_out = lsmr_annihilate(x, y)
    assert y_out.shape == y.shape
