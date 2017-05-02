import pytest

from linearmodels.asset_pricing.model import LinearFactorModel
from linearmodels.tests.asset_pricing._utility import generate_data


@pytest.fixture(scope='module', params=['numpy', 'pandas'])
def data(request):
    return generate_data(output=request.param)


def get_all(res):
    attrs = dir(res)
    for attr in attrs:
        if attr.startswith('_'):
            continue
        attr = getattr(res, attr)
        if callable(attr):
            attr()


def test_linear_model_cross_section_smoke(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    mod.fit(method='cs')
    res = mod.fit(method='cross-section')
    # get_all(res)


def test_linear_model_time_series_smoke(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    mod.fit(method='ts')
    res = mod.fit(method='time-series')
    get_all(res)


def test_linear_model_type_error(data):
    mod = LinearFactorModel(data.portfolios, data.factors)
    with pytest.raises(ValueError):
        mod.fit(method='unknown')
