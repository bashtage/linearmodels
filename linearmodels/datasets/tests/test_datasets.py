import pandas as pd

from linearmodels.datasets import meps


def test_meps_load():
    data = meps.load()
    assert isinstance(data, pd.DataFrame)
