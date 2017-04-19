import pandas as pd
import pytest

from linearmodels.datasets import birthweight, card, fertility, jobtraining, \
    meps, mroz, wage, wage_panel

DATASETS = [birthweight, card, fertility, jobtraining, meps, mroz, wage, wage_panel]


@pytest.fixture(params=DATASETS, scope='module')
def module(request):
    return request.param


def test_dataset(module):
    data = module.load()
    assert isinstance(data, pd.DataFrame)
