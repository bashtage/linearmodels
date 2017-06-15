import pandas as pd
import pytest

from linearmodels.datasets import (birthweight, card, fertility, french, fringe,
                                   jobtraining, meps, mroz, munnell, wage, wage_panel)

DATASETS = [birthweight, card, fertility, french, fringe,
            jobtraining, meps, mroz, munnell, wage, wage_panel]
ids = list(map(lambda x: x.__name__.split('.')[-1], DATASETS))


@pytest.fixture(params=DATASETS, ids=ids)
def dataset_module(request):
    return request.param


def test_dataset(dataset_module):
    data = dataset_module.load()
    assert isinstance(data, pd.DataFrame)
