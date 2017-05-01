import pytest

from linearmodels.asset_pricing.model import LinearFactorModel
from linearmodels.tests.asset_pricing._utility import generate_data


@pytest.fixture(scope='module', params=['numpy', 'pandas'])
def data(request):
    return generate_data(output=request.param)


def test_linear_model_cross_section_smoke(data):
    mod = LinearFactorModel(data.factors, data.portfolios)
    mod.fit(method='cs')
    mod.fit(method='cross-section')


def test_linear_model_time_series_smoke(data):
    mod = LinearFactorModel(data.factors, data.portfolios)
    mod.fit(method='ts')
    mod.fit(method='time-series')


def test_linear_model_type_error(data):
    mod = LinearFactorModel(data.factors, data.portfolios)
    with pytest.raises(ValueError):
        mod.fit(method='unknown')
