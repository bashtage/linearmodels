"""
BetweenOLS where the means of the regressors are collinear, and df_model

The regression of BetweenOLS uses the means of the entities. The regressors can be
collinear in the means when they are not collinear in the data: there are fewer
entities than regressors, or the dummy variables of the periods are the constant 1/T in
the means of a balanced panel and so they are collinear with the constant. The
covariance raised numpy's "Singular matrix" error when it was used (and the parameters
were not identified), and the error has to say what is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from linearmodels.datasets import wage_panel
from linearmodels.panel import BetweenOLS


def make_panel(nentity, nperiod, nvar, seed=0):
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    nobs = nentity * nperiod
    x = pd.DataFrame(
        rg.standard_normal((nobs, nvar)),
        index=index,
        columns=[f"x{i}" for i in range(nvar)],
    )
    y = pd.Series(rg.standard_normal(nobs), index=index, name="y")
    return y, x


def test_between_with_more_regressors_than_entities():
    y, x = make_panel(2, 3, 3)
    with pytest.raises(ValueError, match="collinear in the means of the entities"):
        BetweenOLS(y, x).fit()
    # The rank is not checked if the user does not want it
    res = BetweenOLS(y, x, check_rank=False).fit(debiased=False)
    assert res.params.shape == (3,)


def test_between_with_time_dummies_in_a_balanced_panel():
    data = wage_panel.load()
    year = pd.Categorical(data.year)
    data = data.set_index(["nr", "year"])
    data["year"] = year
    data["const"] = 1.0
    with pytest.raises(ValueError, match="collinear in the means of the entities"):
        BetweenOLS(data.lwage, data[["const", "expersq", "union", "year"]]).fit()
    res = BetweenOLS(data.lwage, data[["const", "expersq", "union"]]).fit()
    assert res.df_resid == 545 - 3


def test_between_df_model_is_the_number_of_regressors():
    y, x = make_panel(10, 3, 2)
    res = BetweenOLS(y, x).fit()
    assert res.df_model == 2
    assert isinstance(res.df_model, int)
