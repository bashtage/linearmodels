"""
Tests of the error for a keyword argument of fit that the covariance estimator does
not accept
"""

import numpy as np
import pandas as pd
import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML, AbsorbingLS
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
import linearmodels.panel.model as panel_module
from linearmodels.shared.exceptions import UnknownOptionError
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 60
UNEXPECTED = r"fit\(\) got an unexpected keyword argument 'bogus'\. "


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261014)
    z = rng.standard_normal((NOBS, 2))
    x = (z @ [1.0, 0.5] + rng.standard_normal(NOBS))[:, None]
    y = x[:, 0] + rng.standard_normal(NOBS)
    index = pd.MultiIndex.from_product([range(6), range(10)])
    factors = rng.standard_normal((NOBS, 2))
    return {
        "iv": (y, None, x, z),
        "cats": pd.DataFrame({"a": pd.Categorical(rng.integers(0, 5, NOBS))}),
        "equations": {"a": {"dependent": y, "exog": x}},
        "iv_equations": {
            "a": {"dependent": y, "exog": None, "endog": x, "instruments": z}
        },
        "panel": (pd.Series(y, index=index), pd.DataFrame(x, index=index)),
        "factors": factors,
        "portfolios": factors @ rng.standard_normal((2, 4))
        + rng.standard_normal((NOBS, 4)),
    }


MODELS = {
    "IV2SLS": lambda d: IV2SLS(*d["iv"]),
    "IVLIML": lambda d: IVLIML(*d["iv"]),
    "IVGMM": lambda d: IVGMM(*d["iv"]),
    "IVGMMCUE": lambda d: IVGMMCUE(*d["iv"]),
    "IVJIVE": lambda d: IVJIVE(*d["iv"]),
    "AbsorbingLS": lambda d: AbsorbingLS(d["iv"][0], absorb=d["cats"]),
    "SUR": lambda d: SUR(d["equations"]),
    "IV3SLS": lambda d: IV3SLS(d["iv_equations"]),
    "IVSystemGMM": lambda d: IVSystemGMM(d["iv_equations"]),
    "TradedFactorModel": lambda d: TradedFactorModel(d["portfolios"], d["factors"]),
    "LinearFactorModel": lambda d: LinearFactorModel(d["portfolios"], d["factors"]),
    "LinearFactorModelGMM": lambda d: LinearFactorModelGMM(
        d["portfolios"], d["factors"]
    ),
}


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("cov_type", ["robust", "kernel"])
def test_unexpected_keyword(data, name, cov_type):
    model = MODELS[name](data)
    with pytest.raises(TypeError, match=UNEXPECTED + r"The keyword arguments"):
        model.fit(cov_type=cov_type, bogus=1)


def test_accepted_keywords_depend_on_cov_type(data):
    # The keywords that a covariance estimator accepts depend on its type
    model = IV2SLS(*data["iv"])
    match = r"cov_type='robust' are: center, kappa\."
    with pytest.raises(TypeError, match=match):
        model.fit(cov_type="robust", clusters=np.arange(NOBS))
    model.fit(cov_type="clustered", clusters=np.arange(NOBS))


@pytest.mark.parametrize("cls", [PooledOLS, PanelOLS, BetweenOLS, RandomEffects])
def test_panel_unexpected_keyword(data, cls):
    model = cls(*data["panel"])
    with pytest.raises(ValueError, match=r"are: bogus"):
        model.fit(cov_type="robust", bogus=1)


def test_panel_cluster_options_need_clustered(data):
    model = PooledOLS(*data["panel"])
    with pytest.raises(ValueError, match=r"are: cluster_entity"):
        model.fit(cov_type="robust", cluster_entity=True)


def test_unknown_kernel_of_fit_is_reported_before_estimation(data, monkeypatch):
    def estimated(*args, **kwargs):
        raise AssertionError("The model was estimated")

    iv_model = IV2SLS(*data["iv"])
    pooled = PooledOLS(*data["panel"])
    fama_macbeth = FamaMacBeth(*data["panel"])
    monkeypatch.setattr(iv_model, "estimate_parameters", estimated)
    monkeypatch.setattr(panel_module, "_lstsq", estimated)
    with pytest.raises(UnknownOptionError, match=r"Unknown kernel: 'bogus'"):
        iv_model.fit(cov_type="kernel", kernel="bogus")
    with pytest.raises(UnknownOptionError, match=r"Unknown kernel: 'bogus'"):
        pooled.fit(cov_type="kernel", kernel="bogus")
    with pytest.raises(UnknownOptionError, match=r"Unknown kernel: 'bogus'"):
        fama_macbeth.fit(cov_type="kernel", kernel="bogus")


@pytest.mark.parametrize("cls", [IVGMM, IVGMMCUE])
def test_unknown_weight_kernel_of_iv_gmm(data, cls):
    with pytest.raises(UnknownOptionError, match=r"Unknown kernel: 'bogus'"):
        cls(*data["iv"], weight_type="kernel", kernel="bogus")


def test_unknown_weight_kernel_of_system_gmm(data):
    with pytest.raises(UnknownOptionError, match=r"Unknown kernel: 'bogus'"):
        IVSystemGMM(data["iv_equations"], weight_type="kernel", kernel="bogus")
