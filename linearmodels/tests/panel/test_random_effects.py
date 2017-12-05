from itertools import product

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import RandomEffects
from linearmodels.tests.panel._utility import generate_data, datatypes, assert_frame_similar

pytestmark = pytest.mark.filterwarnings('ignore::linearmodels.utility.MissingValueWarning')

missing = [0.0, 0.20]
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


def test_fitted_effects_residuals(data):
    mod = RandomEffects(data.y, data.x)
    res = mod.fit()

    expected = mod.exog.values2d @ res.params.values
    expected = pd.DataFrame(expected, index=mod.exog.index, columns=['fitted_values'])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    expected.iloc[:, 0] = res.resids
    expected.columns = ['idiosyncratic']
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    fitted_error = res.fitted_values + res.idiosyncratic.values
    expected.iloc[:, 0] = mod.dependent.values2d - fitted_error
    expected.columns = ['estimated_effects']
    assert_allclose(res.estimated_effects, expected)
    assert_frame_similar(res.estimated_effects, expected)
