"""
Tests of the error for a keyword argument of fit that the covariance estimator does
not accept
"""

import re

import numpy as np
from numpy.testing import assert_allclose
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
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
import linearmodels.panel.model as panel_module
from linearmodels.shared.exceptions import UnknownOptionError
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 60
UNEXPECTED = (
    r"The keyword argument 'bogus' is not accepted with cov_type='{}'\. "
    r"The accepted keyword arguments are: "
)


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
    with pytest.raises(ValueError, match=UNEXPECTED.format(cov_type)):
        model.fit(cov_type=cov_type, bogus=1)


def test_message_of_unexpected_keywords(data):
    model = IV2SLS(*data["iv"])
    expected = (
        "The keyword argument 'bogus' is not accepted with cov_type='robust'. "
        "The accepted keyword arguments are: center, extra_df, kappa."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        model.fit(cov_type="robust", bogus=1)
    with pytest.raises(ValueError, match=r"The keyword arguments 'a', 'b' are not"):
        model.fit(cov_type="robust", a=1, b=2)
    with pytest.raises(ValueError, match=r"arguments are: none\.$"):
        LinearFactorModel(data["portfolios"], data["factors"]).fit(bogus=1)


def test_accepted_keywords_depend_on_cov_type(data):
    # The keywords that a covariance estimator accepts depend on its type
    model = IV2SLS(*data["iv"])
    match = r"cov_type='robust'\. The accepted keyword arguments are: center, extra_df"
    with pytest.raises(ValueError, match=match):
        model.fit(cov_type="robust", clusters=np.arange(NOBS))
    model.fit(cov_type="clustered", clusters=np.arange(NOBS))


@pytest.mark.parametrize("cls", [PooledOLS, PanelOLS, BetweenOLS, RandomEffects])
def test_panel_unexpected_keyword(data, cls):
    model = cls(*data["panel"])
    with pytest.raises(ValueError, match=UNEXPECTED.format("robust")):
        model.fit(cov_type="robust", bogus=1)


def test_panel_cluster_options_need_clustered(data):
    model = PooledOLS(*data["panel"])
    with pytest.raises(ValueError, match=r"argument 'cluster_entity' is not accepted"):
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


@pytest.mark.parametrize("cov_type", ["robust", "unadjusted"])
@pytest.mark.parametrize("cls", [IV2SLS, IVLIML])
def test_extra_df_is_accepted_by_the_iv_models(data, cls, cov_type):
    model = cls(*data["iv"])
    base = model.fit(cov_type=cov_type, debiased=True)
    extra = model.fit(cov_type=cov_type, debiased=True, extra_df=2)
    nobs, nvar = NOBS, 1
    scale = (nobs - nvar) / (nobs - nvar - 2)
    assert_allclose(extra.cov, base.cov * scale)


def test_cluster_options_that_the_model_does_not_support_raise(data):
    between = BetweenOLS(*data["panel"])
    for option in ("cluster_entity", "cluster_time"):
        with pytest.raises(ValueError, match=option):
            between.fit(cov_type="clustered", **{option: True})
    first_difference = FirstDifferenceOLS(*data["panel"])
    with pytest.raises(ValueError, match="cluster_time"):
        first_difference.fit(cov_type="clustered", cluster_time=True)
    first_difference.fit(cov_type="clustered", cluster_entity=True)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "clustered", "kernel"])
def test_panel_ols_sets_extra_df_from_its_effects(data, cov_type):
    # extra_df is the number of effects, so it is not a keyword argument of fit
    model = PanelOLS(*data["panel"], entity_effects=True)
    match = (
        f"The keyword argument 'extra_df' is not accepted with cov_type='{cov_type}'"
    )
    with pytest.raises(ValueError, match=match):
        model.fit(cov_type=cov_type, extra_df=1)


def test_unknown_kernel_is_shown_in_lower_case(data):
    with pytest.raises(UnknownOptionError, match=r"Unknown kernel: 'bogus'"):
        IV2SLS(*data["iv"]).fit(cov_type="kernel", kernel="Bogus")
