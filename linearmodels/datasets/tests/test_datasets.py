import pandas as pd
import pytest

from linearmodels.datasets import meps, wage, mroz, card, fertility, jobtraining

DATASETS = [meps, wage, mroz, card, fertility, jobtraining]


@pytest.fixture(params=DATASETS, scope='module')
def module(request):
    return request.param


def test_dataset(module):
    data = module.load()
    assert isinstance(data, pd.DataFrame)
