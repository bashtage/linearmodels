from linearmodels.compat.pandas import get_codes, to_numpy
from linearmodels.compat.statsmodels import Summary

from itertools import product

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
import scipy.sparse as sp
from scipy.sparse import csc_matrix

from linearmodels.iv._utility import annihilate
from linearmodels.iv.absorbing import (_VARIABLE_CACHE, AbsorbingLS,
                                       AbsorbingRegressor, Interaction,
                                       category_continuous_interaction,
                                       category_interaction, category_product,
                                       clear_cache)
from linearmodels.iv.model import _OLS
from linearmodels.iv.results import AbsorbingLSResults, OLSResults
from linearmodels.panel.utility import dummy_matrix
from linearmodels.utility import AttrDict, MissingValueWarning

NOBS = 100
pytestmark = pytest.mark.filterwarnings('ignore:the matrix subclass:PendingDeprecationWarning')


class Hasher(object):
    @property
    def hash_func(self):
        try:
            import xxhash
            return xxhash.xxh64()
        except ImportError:
            import hashlib
            return hashlib.sha1()

    def single(self, value):
        h = self.hash_func
        h.update(np.ascontiguousarray(value))
        return h.hexdigest()


hasher = Hasher()


@pytest.fixture(scope='function')
def rs(request):
    return np.random.RandomState(12345678)


def random_cat(ncat, size, frame=False, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    series = pd.Series(pd.Categorical(rs.randint(0, ncat, size)))
    if frame:
        return pd.DataFrame(series)
    return series


def random_cont(size, frame=False, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    series = pd.Series(rs.standard_normal(size))
    if frame:
        return pd.DataFrame(series)
    return series


@pytest.fixture(scope='module', params=[1, 2, 3])
def cat(request):
    rs = np.random.RandomState(0)
    return pd.DataFrame(
        {str(i): random_cat(4, NOBS, rs=rs) for i in range(request.param)})


@pytest.fixture(scope='module', params=[1, 2])
def cont(request):
    rs = np.random.RandomState(0)
    return pd.DataFrame(
        {'cont' + str(i): pd.Series(rs.standard_normal(NOBS)) for i in range(request.param)})


@pytest.fixture(scope='module', params=[True, False])
def weights(request):
    if not request.param:
        return None
    rs = np.random.RandomState(0)
    return rs.chisquare(10, NOBS) / 10.0


@pytest.fixture(scope='module', params=[0, 1, 2])
def interact(request):
    if not request.param:
        return None
    rs = np.random.RandomState(0)
    interactions = []
    for i in range(request.param):
        cat = random_cat(4, 100, frame=True, rs=rs)
        cont = random_cont(100, rs=rs, frame=True)
        interactions.append(Interaction(cat, cont))
    return interactions


def generate_data(k=3, const=True, nfactors=1, factor_density=10, nobs=2000, cont_interactions=1,
                  format='interaction', singleton_interaction=False, weighted=False, ncont=0):
    rs = np.random.RandomState(1234567890)
    if isinstance(factor_density, int):
        density = [factor_density] * max(nfactors, cont_interactions)
    else:
        density = factor_density
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
    for i in range(ncont):
        cont = rs.standard_normal(size=nobs)
        factors.append(pd.Series(cont))

    if factors:
        factors = pd.concat(factors, 1)
        if format == 'interaction':
            if nfactors and ncont:
                factors = Interaction(factors.iloc[:, :nfactors], factors.iloc[:, nfactors:])
            elif nfactors:
                factors = Interaction(factors, None)
            else:
                factors = Interaction(None, factors)
    else:
        factors = None

    interactions = []
    for i in range(cont_interactions):
        ncat = nobs // density[min(i, len(density) - 1)]
        fact = rs.randint(ncat, size=(nobs))
        effects = rs.standard_normal(nobs)
        y += effects
        df = pd.DataFrame(pd.Series(pd.Categorical(fact)), columns=['fact{0}'.format(i)])
        df_eff = pd.DataFrame(effects[:, None], columns=['effect_{0}'.format(i)])
        interactions.append(Interaction(df, df_eff))
    if format == 'pandas':
        for i, interact in enumerate(interactions):
            interactions[i] = pd.concat([interact.cat, interact.cont], 1)
    interactions = interactions if interactions else None
    if interactions and singleton_interaction:
        interactions = interactions[0]
    if weighted:
        weights = pd.DataFrame(rs.chisquare(10, size=(nobs, 1)) / 10)
    else:
        weights = None

    return AttrDict(y=y, x=x, absorb=factors, interactions=interactions, weights=weights)


# Permutations, k in (0,3), const in (True,False), factors=(0,1,2), interactions in (0,1)


# k=3, const=True, nfactors=1, factor_density=10, nobs=2000, cont_interactions=1,
#                   format='interaction', singleton_interaction=False
configs = product([0, 3],  # k
                  [False, True],  # constant
                  [1, 2, 0],  # factors
                  [10],  # density
                  [2000],  # nobs
                  [0, 1],  # cont interactions
                  ['interaction', 'pandas'],  # format
                  [False, True],  # singleton
                  [False, True],  # weighted
                  [0, 1]  # ncont
                  )

configs = [c for c in configs if (c[2] or c[5] or c[9])]
id_str = 'k: {0}, const: {1}, nfactors: {2}, density: {3}, nobs: {4}, ' \
         'cont_interacts: {5}, format:{6}, singleton:{7}, weighted: {8}, ncont: {9}'
ids = [id_str.format(*config) for config in configs]


@pytest.fixture(scope='module', params=configs, ids=ids)
def data(request):
    return generate_data(*request.param)


configs_ols = product([0, 3],  # k
                      [False, True],  # constant
                      [1, 2, 0],  # factors
                      [50],  # density
                      [500],  # nobs
                      [0, 1],  # cont interactions
                      ['interaction'],  # format
                      [False],  # singleton
                      [False, True],  # weighted
                      [0, 1]  # ncont
                      )

configs_ols = [c for c in configs_ols if (c[0] or c[1])]
id_str = 'k: {0}, const: {1}, nfactors: {2}, density: {3}, nobs: {4}, ' \
         'cont_interacts: {5}, format:{6}, singleton:{7}, weighted: {8}, ncont: {9}'
ids_ols = [id_str.format(*config) for config in configs_ols]


@pytest.fixture(scope='module', params=configs_ols, ids=ids_ols)
def ols_data(request):
    return generate_data(*request.param)


def test_smoke(data):
    mod = AbsorbingLS(data.y, data.x, absorb=data.absorb, interactions=data.interactions,
                      weights=data.weights)
    mod.fit()


def test_absorbing_exceptions(rs):
    with pytest.raises(TypeError):
        AbsorbingLS(rs.standard_normal(NOBS), rs.standard_normal((NOBS, 2)),
                    absorb=rs.standard_normal((NOBS, 2)))
    with pytest.raises(ValueError):
        AbsorbingLS(rs.standard_normal(NOBS), rs.standard_normal((NOBS - 1, 2)))
    with pytest.raises(ValueError):
        AbsorbingLS(rs.standard_normal(NOBS), rs.standard_normal((NOBS, 2)),
                    absorb=pd.DataFrame(rs.standard_normal((NOBS - 1, 1))))
    with pytest.raises(ValueError):
        AbsorbingLS(rs.standard_normal(NOBS), rs.standard_normal((NOBS, 2)),
                    interactions=random_cat(10, NOBS - 1, frame=True, rs=rs))
    mod = AbsorbingLS(rs.standard_normal(NOBS), rs.standard_normal((NOBS, 2)),
                      interactions=random_cat(10, NOBS, frame=True, rs=rs))
    with pytest.raises(RuntimeError):
        mod.absorbed_dependent
    with pytest.raises(RuntimeError):
        mod.absorbed_exog
    with pytest.raises(TypeError):
        AbsorbingLS(rs.standard_normal(NOBS), rs.standard_normal((NOBS, 2)),
                    interactions=rs.randint(0, 10, size=(NOBS, 2)))


def test_clear_cache():
    _VARIABLE_CACHE['key'] = 'value'
    clear_cache()
    assert len(_VARIABLE_CACHE) == 0


def test_category_product(cat):
    prod = category_product(cat)
    if cat.shape[1] == 1:
        assert_series_equal(prod, cat.iloc[:, 0], check_names=False)
    else:
        alt = cat.iloc[:, 0].astype('int64')
        for i in range(1, cat.shape[1]):
            alt += 10 ** (4 * i) * cat.iloc[:, i].astype('int64')
        alt = pd.Categorical(alt)
        alt = pd.Series(alt)
        df = pd.DataFrame([prod.cat.codes, alt.cat.codes], index=['cat_prod', 'alt']).T
        g = df.groupby('cat_prod').alt
        assert (g.nunique() == 1).all()
        g = df.groupby('alt').cat_prod
        assert (g.nunique() == 1).all()


def test_category_product_too_large(rs):
    dfc = {}
    for i in range(20):
        dfc[str(i)] = random_cat(10, 1000)
    cat = pd.DataFrame(dfc)
    with pytest.raises(ValueError):
        category_product(cat)


def test_category_product_not_cat(rs):
    cat = pd.DataFrame({str(i): pd.Series(rs.randint(0, 10, 1000)) for i in range(3)})
    with pytest.raises(TypeError):
        category_product(cat)


def test_category_interaction():
    c = pd.Series(pd.Categorical([0, 0, 0, 1, 1, 1]))
    actual = category_interaction(c, precondition=False).A
    expected = np.zeros((6, 2))
    expected[:3, 0] = 1.0
    expected[3:, 1] = 1.0
    assert_allclose(actual, expected)

    actual = category_interaction(c, precondition=True).A
    cond = np.sqrt((expected ** 2).sum(0))
    expected /= cond
    assert_allclose(actual, expected)


def test_category_continuous_interaction():
    c = pd.Series(pd.Categorical([0, 0, 0, 1, 1, 1]))
    v = pd.Series(np.arange(6.))
    actual = category_continuous_interaction(c, v, precondition=False)
    expected = np.zeros((6, 2))
    expected[:3, 0] = v[:3]
    expected[3:, 1] = v[3:]

    assert_allclose(actual.A, expected)

    actual = category_continuous_interaction(c, v, precondition=True)
    cond = np.sqrt((expected ** 2).sum(0))
    expected /= cond
    assert_allclose(actual.A, expected)


def test_category_continuous_interaction_interwoven():
    c = pd.Series(pd.Categorical([0, 1, 0, 1, 0, 1]))
    v = pd.Series(np.arange(6.))
    actual = category_continuous_interaction(c, v, precondition=False)
    expected = np.zeros((6, 2))
    expected[::2, 0] = v[::2]
    expected[1::2, 1] = v[1::2]
    assert_allclose(actual.A, expected)


def test_interaction_cat_only(cat):
    interact = Interaction(cat=cat)
    assert interact.nobs == cat.shape[0]
    assert_frame_equal(cat, interact.cat)
    expected = category_interaction(category_product(cat), precondition=False)
    actual = interact.sparse
    assert isinstance(actual, csc_matrix)
    assert_allclose(expected.A, actual.A)


def test_interaction_cont_only(cont):
    interact = Interaction(cont=cont)
    assert interact.nobs == cont.shape[0]
    assert_frame_equal(cont, interact.cont)
    expected = to_numpy(cont)
    actual = interact.sparse
    assert isinstance(actual, csc_matrix)
    assert_allclose(expected, actual.A)


def test_interaction_cat_cont(cat, cont):
    interact = Interaction(cat=cat, cont=cont)
    assert interact.nobs == cat.shape[0]
    assert_frame_equal(cat, interact.cat)
    assert_frame_equal(cont, interact.cont)
    base = category_interaction(category_product(cat), precondition=False).A
    expected = []
    for i in range(cont.shape[1]):
        element = base.copy()
        element[np.where(element)] = to_numpy(cont.iloc[:, i])
        expected.append(element)
    expected = np.column_stack(expected)
    actual = interact.sparse
    assert isinstance(actual, csc_matrix)
    assert_allclose(expected, interact.sparse.A)


def test_interaction_from_frame(cat, cont):
    base = Interaction(cat=cat, cont=cont)
    interact = Interaction.from_frame(pd.concat([cat, cont], 1))
    assert_allclose(base.sparse.A, interact.sparse.A)


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
    interact = Interaction(to_numpy(cat), cont)
    assert_allclose(base.sparse.A, interact.sparse.A)


def test_absorbing_regressors(cat, cont, interact, weights):
    areg = AbsorbingRegressor(cat=cat, cont=cont, interactions=interact, weights=weights)
    rank = areg.approx_rank
    expected_rank = 0

    expected = []
    for i, col in enumerate(cat):
        expected_rank += pd.Series(get_codes(cat[col].cat)).nunique() - (i > 0)
    expected.append(dummy_matrix(cat, precondition=False)[0])
    expected_rank += cont.shape[1]
    expected.append(csc_matrix(cont))
    if interact is not None:
        for inter in interact:
            interact_mat = inter.sparse
            expected_rank += interact_mat.shape[1]
            expected.append(interact_mat)
    expected = sp.hstack(expected, format='csc')
    if weights is not None:
        expected = (sp.diags(np.sqrt(weights)).dot(expected)).asformat('csc')
    actual = areg.regressors
    assert expected.shape == actual.shape
    assert_array_equal(expected.indptr, actual.indptr)
    assert_array_equal(expected.indices, actual.indices)
    assert_allclose(expected.A, actual.A)
    assert expected_rank == rank


def test_absorbing_regressors_hash(cat, cont, interact, weights):
    areg = AbsorbingRegressor(cat=cat, cont=cont, interactions=interact, weights=weights)
    # Build hash
    hashes = []
    for col in cat:
        hashes.append((hasher.single(to_numpy(get_codes(cat[col].cat)).data),))
    for col in cont:
        hashes.append((hasher.single(to_numpy(cont[col]).data),))
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
    mod = AbsorbingLS(ols_data.y, ols_data.x, absorb=ols_data.absorb,
                      interactions=ols_data.interactions, weights=ols_data.weights)
    res = mod.fit()
    absorb = []
    has_dummy = False
    if ols_data.absorb is not None:
        absorb.append(to_numpy(ols_data.absorb.cont))
        if ols_data.absorb.cat.shape[1] > 0:
            absorb.append(dummy_matrix(ols_data.absorb.cat, precondition=False)[0].A)
        has_dummy = ols_data.absorb.cat.shape[1] > 0
    if ols_data.interactions is not None:
        for interact in ols_data.interactions:
            absorb.append(interact.sparse.A)
    _x = ols_data.x
    if absorb:
        absorb = np.column_stack(absorb)
        if np.any(np.ptp(_x, 0) == 0) and has_dummy:
            if ols_data.weights is None:
                absorb = annihilate(absorb, np.ones((absorb.shape[0], 1)))
            else:
                root_w = np.sqrt(mod.weights.ndarray)
                wabsorb = annihilate(root_w * absorb, root_w)
                absorb = (1. / root_w) * wabsorb
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
    gen = generate_data(2, True, 2, format='pandas', ncont=0, cont_interactions=1)
    first = len(_VARIABLE_CACHE)
    mod = AbsorbingLS(gen.y, gen.x, absorb=gen.absorb.iloc[:, :1], interactions=gen.interactions)
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


def test_instrments():
    gen = generate_data(2, True, 2, format='pandas', ncont=0, cont_interactions=1)
    mod = AbsorbingLS(gen.y, gen.x, absorb=gen.absorb.iloc[:, :1], interactions=gen.interactions)
    assert mod.instruments.shape[1] == 0


def assert_results_equal(o_res: OLSResults, a_res: AbsorbingLSResults, k: int = None):
    if k is None:
        k = a_res.params.shape[0]
    attrs = [v for v in dir(o_res) if not v.startswith('_')]
    callables = ['conf_int']
    skip = ['summary', 'test_linear_constraint', 'predict', 'model', 'f_statistic', 'wald_test',
            'method']
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
            if attr == 'conf_int':
                left = left.iloc[:k]
            elif attr == 'cov':
                left = left.iloc[:k, :k]
            assert_allclose(left, right, rtol=2e-4, atol=1e-6)
        elif isinstance(left, pd.Series):
            assert_allclose(left.iloc[:k], right.iloc[:k], rtol=1e-5)
        else:
            if isinstance(left, float):
                assert_allclose(left, right, atol=1e-10)
            else:
                try:
                    assert left == right
                except AssertionError:
                    print(attr)
                    assert left == right
    assert isinstance(a_res.summary, Summary)
    assert isinstance(a_res.absorbed_effects, pd.DataFrame)
    assert a_res.absorbed_rsquared <= a_res.rsquared


def test_center_cov_arg():
    gen = generate_data(2, True, 2, format='pandas', ncont=0, cont_interactions=1)
    mod = AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)
    res = mod.fit(center=True)
    assert 'center' not in res.cov_config


def test_drop_missing():
    gen = generate_data(2, True, 2, format='pandas', ncont=0, cont_interactions=1)
    gen.y[::53] = np.nan
    gen.x[::79] = np.nan
    with pytest.warns(MissingValueWarning):
        AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)

    gen = generate_data(2, True, 2, format='pandas', ncont=0, cont_interactions=1)
    for col in gen.absorb:
        gen.absorb[col] = gen.absorb[col].astype('int64').astype('object')
        gen.absorb[col].iloc[::91] = np.nan
        gen.absorb[col] = pd.Categorical(to_numpy(gen.absorb[col]))
    with pytest.warns(MissingValueWarning):
        AbsorbingLS(gen.y, gen.x, absorb=gen.absorb, interactions=gen.interactions)
