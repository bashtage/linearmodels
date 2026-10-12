"""
Entity and time effects in a panel with a single entity or a single period

The effects remove all of the variation of the regressors. The error has to say that the
effects absorbed the variables. The default algorithm that removes both effects
raised "No objects to concatenate" from pandas, since it uses the dummy variables of the
shorter dimension, and there are none.
"""

import numpy as np
import pandas as pd
import pytest

from linearmodels.panel import PanelOLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.utility import AbsorbingEffectError


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


@pytest.mark.parametrize("low_memory", [None, False, True])
@pytest.mark.parametrize("shape", [(1, 6), (6, 1)])
def test_entity_and_time_effects_absorb_a_single_entity_or_period(shape, low_memory):
    y, x = make_panel(*shape, 2)
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    with pytest.raises(AbsorbingEffectError, match="absorbed"):
        mod.fit(low_memory=low_memory)


@pytest.mark.parametrize("shape", [(1, 6), (6, 1)])
def test_demean_both_of_a_single_entity_or_period(shape):
    # There is nothing to remove besides the means of the group with the observations
    _, x = make_panel(*shape, 2)
    demeaned = PanelData(x).demean("both")
    assert demeaned.dataframe.shape == x.shape
    np.testing.assert_allclose(demeaned.values2d, 0.0, atol=1e-14)
