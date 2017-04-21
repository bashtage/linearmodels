from itertools import product
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS
from linearmodels.tests.panel._utility import generate_data

missing = [0.0, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
perms = list(product(missing, datatypes))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype, other_effects=1)


def test_categorical_input(data):
    y = PanelData(data.y)
    nt = y.values2d.shape[0]
    effects = np.random.randint(0, 5, size=(nt, 2))
    temp = {}
    for i, e in enumerate(effects.T):
        name = 'effect.' + str(i)
        temp[name] = pd.Categorical(pd.Series(e, index=y.index, name=name))
    effects = pd.DataFrame(temp, index=y.index)
    mod = PanelOLS(data.y, data.x, other_effects=effects)
    mod.fit()

    clusters = np.random.randint(0, y.shape[2] // 2, size=(nt, 2))
    temp = {}
    for i, c in enumerate(clusters.T):
        name = 'effect.' + str(i)
        temp[name] = pd.Categorical(pd.Series(c, index=y.index, name=name))
    clusters = pd.DataFrame(temp, index=y.index)
    mod.fit(cov_type='clustered', clusters=clusters)


def test_string_input(data):
    y = PanelData(data.y)
    nt = y.values2d.shape[0]
    temp = {}
    prim = ['a', 'b', 'c', 'd', 'e']
    for i in range(2):
        name = 'effect.' + str(i)
        temp[name] = pd.Series(np.random.choice(prim, size=(nt)), index=y.index, name=name)
    effects = pd.DataFrame(temp, index=y.index)
    mod = PanelOLS(data.y, data.x, other_effects=effects)
    mod.fit()

    clusters = np.random.randint(0, y.shape[2] // 2, size=(nt, 2))
    temp = {}
    prim = list(map(lambda s: ''.join(s), list(product(ascii_lowercase, ascii_lowercase))))

    for i, c in enumerate(clusters.T):
        name = 'effect.' + str(i)
        temp[name] = pd.Series(np.random.choice(prim, size=(nt)), index=y.index, name=name)
    clusters = pd.DataFrame(temp, index=y.index)
    mod.fit(cov_type='clustered', clusters=clusters)


def test_integer_input(data):
    y = PanelData(data.y)
    nt = y.values2d.shape[0]
    effects = np.random.randint(0, 5, size=(nt, 2))
    temp = {}
    for i, e in enumerate(effects.T):
        name = 'effect.' + str(i)
        temp[name] = pd.Series(e, index=y.index, name=name)
    effects = pd.DataFrame(temp, index=y.index)
    mod = PanelOLS(data.y, data.x, other_effects=effects)
    mod.fit()

    clusters = np.random.randint(0, y.shape[2] // 2, size=(nt, 2))
    temp = {}
    for i, c in enumerate(clusters.T):
        name = 'effect.' + str(i)
        temp[name] = pd.Series(c, index=y.index, name=name)
    clusters = pd.DataFrame(temp, index=y.index)
    mod.fit(cov_type='clustered', clusters=clusters)


def test_mixed_input(data):
    y = PanelData(data.y)
    nt = y.values2d.shape[0]
    effects = np.random.randint(0, 5, size=(nt))
    prim = ['a', 'b', 'c', 'd', 'e']
    temp = {}
    temp['effect.0'] = pd.Categorical(pd.Series(effects, index=y.index))
    temp['effect.1'] = pd.Series(np.random.choice(prim, size=(nt)), index=y.index)
    effects = pd.DataFrame(temp, index=y.index)
    mod = PanelOLS(data.y, data.x, other_effects=effects)
    mod.fit()

    clusters = np.random.randint(0, y.shape[2] // 2, size=(nt, 2))
    temp = {}
    prim = list(map(lambda s: ''.join(s), list(product(ascii_lowercase, ascii_lowercase))))
    temp['var.cluster.0'] = pd.Series(np.random.choice(prim, size=(nt)), index=y.index)
    temp['var.cluster.1'] = pd.Series(clusters[:, 1], index=y.index)
    clusters = pd.DataFrame(temp, index=y.index)
    mod.fit(cov_type='clustered', clusters=clusters)


def test_nested_effects(data):
    y = PanelData(data.y)
    effects = pd.DataFrame(y.entity_ids // 2, index=y.index)
    with pytest.raises(ValueError) as exception:
        PanelOLS(data.y, data.x, entity_effects=True, other_effects=effects)
    assert 'entity effects' in str(exception.value)

    effects = pd.DataFrame(y.time_ids // 2, index=y.index)
    with pytest.raises(ValueError) as exception:
        PanelOLS(data.y, data.x, time_effects=True, other_effects=effects)
    assert 'time effects' in str(exception.value)

    effects1 = pd.Series(y.entity_ids.squeeze() // 2, index=y.index)
    effects2 = pd.Series(y.entity_ids.squeeze() // 4, index=y.index)
    effects = pd.DataFrame({'eff1': effects1, 'eff2': effects2})
    with pytest.raises(ValueError) as exception:
        PanelOLS(data.y, data.x, other_effects=effects)
    assert 'by other effects' in str(exception.value)
    assert 'time effects' not in str(exception.value)
    assert 'entity effects' not in str(exception.value)
