import pandas as pd

from linearmodels.datasets import meps, wage


def test_meps_load():
    data = meps.load()
    assert isinstance(data, pd.DataFrame)


def test_wage_load():
    data = wage.load()
    assert isinstance(data, pd.DataFrame)
