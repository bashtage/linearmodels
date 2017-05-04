import pandas as pd
import pytest

from linearmodels.datasets import birthweight, card, fertility, jobtraining, \
    meps, mroz, wage, wage_panel, french

DATASETS = [birthweight, card, fertility, jobtraining, meps, mroz, wage,
            wage_panel, french]
ids = list(map(lambda x: x.__name__.split('.')[-1], DATASETS))


@pytest.fixture(params=DATASETS, ids=ids)
def dataset_module(request):
    return request.param


def test_dataset(dataset_module):
    data = dataset_module.load()
    assert isinstance(data, pd.DataFrame)
