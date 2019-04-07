from itertools import product

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv.absorbing import AbsorbingLS, Interaction, category_product
from linearmodels.utility import AttrDict
from pandas.testing import assert_series_equal


@pytest.fixture(scope='module', params=[1, 2, 3])
def cats(request):
    rs = np.random.RandomState(0)
    return pd.DataFrame(
        {str(i): pd.Series(pd.Categorical(rs.randint(0, 3, 25))) for i in range(request.param)})


def generate_data(k=3, const=True, nfactors=1, factor_density=10, nobs=2000, cont_interactions=1,
                  format='interaction', singleton_interaction=False, weighted=False):
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
        fact = rs.randint(ncat, size=(nobs))
        effects = rs.standard_normal(ncat)
        y += effects[fact]
        factors.append(pd.Series(pd.Categorical(fact)))
    if factors:
        factors = pd.concat(factors, 1)
        if format == 'interaction':
            factors = Interaction(factors, None)
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
                  [False, True]  # weighted
                  )

configs = list(configs)
id_str = 'k: {0}, const: {1}, nfactors: {2}, density: {3}, nobs: {4}, ' \
         'cont_interacts: {5}, format:{6}, singleton:{7}, weighted: {8}'
ids = [id_str.format(*config) for config in configs]


@pytest.fixture(scope='module', params=configs, ids=ids)
def data(request):
    return generate_data(*request.param)


def test_smoke(data):
    mod = AbsorbingLS(data.y, data.x, absorb=data.absorb, interactions=data.interactions,
                      weights=data.weights)
    mod.fit()


def test_category_product(cats):
    prod = category_product(cats)
    if cats.shape[1] == 1:
        assert_series_equal(prod, cats.iloc[:, 0], check_names=False)
    else:
        alt = cats.iloc[:, 0].astype('int64')
        for i in range(1, cats.shape[1]):
            alt += 10 ** (4 * i) * cats.iloc[:, i].astype('int64')
        alt = pd.Categorical(alt)
        alt = pd.Series(alt)
        df = pd.DataFrame([prod.cat.codes, alt.cat.codes], index=['cat_prod', 'alt']).T
        g = df.groupby('cat_prod').alt
        assert (g.nunique() == 1).all()
        g = df.groupby('alt').cat_prod
        assert (g.nunique() == 1).all()
