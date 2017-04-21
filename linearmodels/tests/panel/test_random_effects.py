from itertools import product

import pytest

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import RandomEffects
from linearmodels.tests.panel._utility import generate_data

missing = [0.0, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, ntk=(1000, 3, 5), const=const)


def test_random_effects_small_sample(data):
    y = PanelData(data.y)
    mod = RandomEffects(data.y, data.x)
    no_ss = mod.fit()
    ss = mod.fit(small_sample=True)
    if y.dataframe.shape[0] == mod.dependent.dataframe.shape[0]:
        assert (ss.variance_decomposition.Effects == no_ss.variance_decomposition.Effects)
    else:
        assert (ss.variance_decomposition.Effects != no_ss.variance_decomposition.Effects)

    mod = RandomEffects(data.y, data.x, weights=data.w)
    no_ss = mod.fit()
    ss = mod.fit(small_sample=True)
    if y.dataframe.shape[0] == mod.dependent.dataframe.shape[0]:
        assert (ss.variance_decomposition.Effects == no_ss.variance_decomposition.Effects)
    else:
        assert (ss.variance_decomposition.Effects != no_ss.variance_decomposition.Effects)


def test_results_access(data):
    mod = RandomEffects(data.y, data.x)
    res = mod.fit(debiased=False)
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()
