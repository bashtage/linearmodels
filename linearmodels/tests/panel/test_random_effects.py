from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import RandomEffects
from linearmodels.tests.panel._utility import (
    access_attributes,
    assert_frame_similar,
    datatypes,
    generate_data,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)

missing = [0.0, 0.20]
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = ["-".join(str(param) for param in perms) for perm in perms]


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
        assert ss.variance_decomposition.Effects == no_ss.variance_decomposition.Effects
    else:
        assert ss.variance_decomposition.Effects != no_ss.variance_decomposition.Effects

    mod = RandomEffects(data.y, data.x, weights=data.w)
    no_ss = mod.fit()
    ss = mod.fit(small_sample=True)
    if y.dataframe.shape[0] == mod.dependent.dataframe.shape[0]:
        assert ss.variance_decomposition.Effects == no_ss.variance_decomposition.Effects
    else:
        assert ss.variance_decomposition.Effects != no_ss.variance_decomposition.Effects


def test_results_access(data):
    mod = RandomEffects(data.y, data.x)
    res = mod.fit(debiased=False)
    access_attributes(res)


def test_fitted_effects_residuals(data):
    mod = RandomEffects(data.y, data.x)
    res = mod.fit()

    expected = mod.exog.values2d @ res.params.values
    expected = pd.DataFrame(expected, index=mod.exog.index, columns=["fitted_values"])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    expected.iloc[:, 0] = res.resids
    expected.columns = ["idiosyncratic"]
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    fitted_error = res.fitted_values + res.idiosyncratic.values
    estimated_effects = mod.dependent.values2d - fitted_error
    expected.iloc[:, 0] = estimated_effects.iloc[:, 0]
    expected.columns = ["estimated_effects"]
    assert_allclose(res.estimated_effects, expected)
    assert_frame_similar(res.estimated_effects, expected)


def test_extra_df(data):
    mod = RandomEffects(data.y, data.x)
    res = mod.fit()
    res_extra = mod.fit(extra_df=10)
    assert np.all(np.diag(res_extra.cov) > np.diag(res.cov))
