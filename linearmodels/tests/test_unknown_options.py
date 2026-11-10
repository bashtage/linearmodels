"""
Tests of the error for the name of a covariance estimator, a weight matrix estimator
or a kernel that is not supported

Every place that looks up one of these names raises UnknownOptionError, which is a
ValueError and a KeyError and lists the supported names. Releases before 7.1 raised a
bare KeyError, for example KeyError: 'cluster', for many of them.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML, AbsorbingLS
from linearmodels.iv.common import kleibergen_paap
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.exceptions import UnknownOptionError
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 120

# The names that each lookup accepts. They are written out so that the tests fail if a
# name is added or removed, or if the message lists a name that is not accepted
KERNELS = [
    "andrews",
    "bartlett",
    "gallant",
    "newey-west",
    "parzen",
    "qs",
    "quadratic-spectral",
]
IV_COV_TYPES = [
    "HeteroskedasticCovariance",
    "HomoskedasticCovariance",
    "KernelCovariance",
    "OneWayClusteredCovariance",
    "clustered",
    "hccm",
    "heteroskedastic",
    "homo",
    "homoskedastic",
    "kernel",
    "one-way",
    "robust",
    "unadjusted",
]
IV_WEIGHT_TYPES = [
    "clustered",
    "heteroskedastic",
    "homoskedastic",
    "kernel",
    "one-way",
    "robust",
    "unadjusted",
]
IV_GMM_COV_TYPES = [
    "clustered",
    "heteroskedastic",
    "homoskedastic",
    "kernel",
    "robust",
    "unadjusted",
]
RANK_TEST_COV_TYPES = [
    "clustered",
    "hccm",
    "heteroskedastic",
    "heteroskedasticcovariance",
    "homo",
    "homoskedastic",
    "homoskedasticcovariance",
    "kernel",
    "kernelcovariance",
    "one-way",
    "onewayclusteredcovariance",
    "robust",
    "unadjusted",
]
PANEL_COV_TYPES = [
    "ac",
    "autocorrelated",
    "clustered",
    "conventional",
    "dk",
    "driscoll-kraay",
    "heteroskedastic",
    "homoskedastic",
    "kernel",
    "robust",
    "unadjusted",
]
# BetweenOLS does not support the estimators of the time series dimension
BETWEEN_COV_TYPES = [
    "clustered",
    "conventional",
    "heteroskedastic",
    "homoskedastic",
    "robust",
    "unadjusted",
]
FAMA_MACBETH_COV_TYPES = [
    "heteroskedastic",
    "homoskedastic",
    "kernel",
    "robust",
    "unadjusted",
]
SYSTEM_COV_TYPES = [
    "clustered",
    "hac",
    "heteroskedastic",
    "homoskedastic",
    "kernel",
    "robust",
    "unadjusted",
]
# IVSystemGMM does not support clustered estimators
SYSTEM_GMM_TYPES = [
    "hac",
    "heteroskedastic",
    "homoskedastic",
    "kernel",
    "robust",
    "unadjusted",
]
ASSET_PRICING_COV_TYPES = ["heteroskedastic", "kernel", "robust"]


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(20261012)
    instruments = rng.standard_normal((NOBS, 2))
    endog = (instruments @ [1.0, 0.5] + rng.standard_normal(NOBS))[:, None]
    exog = np.column_stack([np.ones(NOBS), rng.standard_normal(NOBS)])
    y = exog @ [1.0, 0.5] + endog[:, 0] + rng.standard_normal(NOBS)
    cats = pd.DataFrame({"a": pd.Categorical(rng.integers(0, 10, NOBS))})
    index = pd.MultiIndex.from_product([range(12), range(10)], names=["i", "t"])
    panel_y = pd.Series(y, index=index, name="y")
    panel_x = pd.DataFrame(exog[:, 1:], index=index, columns=["x"])
    # Four clusters of three entities, as BetweenOLS needs clusters of entities
    panel_clusters = pd.DataFrame({"c": np.arange(NOBS) // 30}, index=index)
    equations = {}
    iv_equations = {}
    for label in ("a", "b"):
        z = rng.standard_normal((NOBS, 2))
        x = (z @ [1.0, 0.5] + rng.standard_normal(NOBS))[:, None]
        dependent = x[:, 0] + rng.standard_normal(NOBS)
        equations[label] = {"dependent": dependent, "exog": x}
        iv_equations[label] = {
            "dependent": dependent,
            "exog": None,
            "endog": x,
            "instruments": z,
        }
    factors = rng.standard_normal((NOBS, 2))
    portfolios = factors @ rng.standard_normal((2, 4)) + rng.standard_normal((NOBS, 4))
    return SimpleNamespace(
        y=y,
        exog=exog,
        endog=endog,
        instruments=instruments,
        cats=cats,
        panel_y=panel_y,
        panel_x=panel_x,
        panel_clusters=panel_clusters,
        equations=equations,
        iv_equations=iv_equations,
        factors=factors,
        portfolios=portfolios,
        clusters=np.arange(NOBS) // 8,
    )


def bandwidth_config(name):
    """An explicit bandwidth for the kernel covariance of the GMM models

    These tests are about the names, not the bandwidth. Without a bandwidth the kernel
    covariance of the GMM models uses a default that gives a DefaultBandwidthWarning.
    """
    return {"bandwidth": 5} if name == "kernel" else {}


def is_clustered(name):
    lowered = name.lower()
    return "cluster" in lowered or lowered == "one-way"


def cluster_config(inputs, name):
    """The clusters that the estimators of a clustered covariance need"""
    return {"clusters": inputs.clusters} if is_clustered(name) else {}


def panel_cluster_config(inputs, name):
    """The clusters of a panel are a frame that has the index of the data"""
    return {"clusters": inputs.panel_clusters} if is_clustered(name) else {}


class Site(NamedTuple):
    """A place that looks up a name, and how to reach it with a value"""

    call: Callable[[Any, str], Any]
    parameter: str
    supported: list[str]


def iv(cls, inputs, **kwargs):
    return cls(inputs.y, inputs.exog, inputs.endog, inputs.instruments, **kwargs)


def absorbing(inputs):
    return AbsorbingLS(inputs.y, inputs.exog[:, 1:], absorb=inputs.cats)


def panel(cls, inputs, **kwargs):
    return cls(inputs.panel_y, inputs.panel_x, **kwargs)


def factor(cls, inputs):
    return cls(inputs.portfolios, inputs.factors)


# One entry for each place in the package that looks up a name. The models that share
# the place are listed in its key. Each call has the value of the parameter as its
# second argument
SITES = {
    # iv/model.py, the covariance estimator of IV2SLS and IVLIML
    "IV2SLS.fit(cov_type)": Site(
        lambda d, v: iv(IV2SLS, d).fit(cov_type=v, **cluster_config(d, v)),
        "cov_type",
        IV_COV_TYPES,
    ),
    "IVLIML.fit(cov_type)": Site(
        lambda d, v: iv(IVLIML, d).fit(cov_type=v, **cluster_config(d, v)),
        "cov_type",
        IV_COV_TYPES,
    ),
    # iv/absorbing.py
    "AbsorbingLS.fit(cov_type)": Site(
        lambda d, v: absorbing(d).fit(cov_type=v, **cluster_config(d, v)),
        "cov_type",
        IV_COV_TYPES,
    ),
    # iv/model.py, the weight matrix estimator of the GMM estimators
    "IVGMM(weight_type)": Site(
        lambda d, v: iv(IVGMM, d, weight_type=v, **cluster_config(d, v)),
        "weight_type",
        IV_WEIGHT_TYPES,
    ),
    "IVGMMCUE(weight_type)": Site(
        lambda d, v: iv(IVGMMCUE, d, weight_type=v, **cluster_config(d, v)),
        "weight_type",
        IV_WEIGHT_TYPES,
    ),
    # iv/gmm.py, the covariance of the GMM estimators and of IVJIVE
    "IVGMM.fit(cov_type)": Site(
        lambda d, v: iv(IVGMM, d).fit(
            cov_type=v, **cluster_config(d, v), **bandwidth_config(v)
        ),
        "cov_type",
        IV_GMM_COV_TYPES,
    ),
    "IVGMMCUE.fit(cov_type)": Site(
        lambda d, v: iv(IVGMMCUE, d).fit(
            cov_type=v, **cluster_config(d, v), **bandwidth_config(v)
        ),
        "cov_type",
        IV_GMM_COV_TYPES,
    ),
    "IVJIVE.fit(cov_type)": Site(
        lambda d, v: iv(IVJIVE, d).fit(cov_type=v, **cluster_config(d, v)),
        "cov_type",
        IV_GMM_COV_TYPES,
    ),
    # iv/common.py
    "kleibergen_paap(cov_type)": Site(
        lambda d, v: kleibergen_paap(
            d.endog, d.instruments, d.exog, v, cluster_config(d, v)
        ),
        "cov_type",
        RANK_TEST_COV_TYPES,
    ),
    # iv/covariance.py, KernelCovariance
    "IV2SLS.fit(kernel)": Site(
        lambda d, v: iv(IV2SLS, d).fit(cov_type="kernel", kernel=v),
        "kernel",
        KERNELS,
    ),
    "AbsorbingLS.fit(kernel)": Site(
        lambda d, v: absorbing(d).fit(cov_type="kernel", kernel=v),
        "kernel",
        KERNELS,
    ),
    # iv/gmm.py, the kernel weights are looked up when the weight matrix is computed
    "IVGMM(weight_type='kernel', kernel)": Site(
        lambda d, v: iv(IVGMM, d, weight_type="kernel", kernel=v, bandwidth=5).fit(),
        "kernel",
        KERNELS,
    ),
    "IVGMM.fit(cov_type='kernel', kernel)": Site(
        lambda d, v: iv(IVGMM, d).fit(cov_type="kernel", kernel=v, bandwidth=5),
        "kernel",
        KERNELS,
    ),
    # iv/covariance.py, kernel_optimal_bandwidth
    "IVGMM(kernel, optimal_bw=True)": Site(
        lambda d, v: iv(
            IVGMM, d, weight_type="kernel", kernel=v, optimal_bw=True
        ).fit(),
        "kernel",
        KERNELS,
    ),
    # iv/model.py, the covariance of IVJIVE
    "IVJIVE.fit(cov_type='kernel', kernel)": Site(
        lambda d, v: iv(IVJIVE, d).fit(cov_type="kernel", kernel=v),
        "kernel",
        KERNELS,
    ),
    # panel/covariance.py, CovarianceManager. The supported names depend on the model
    "PanelOLS.fit(cov_type)": Site(
        lambda d, v: panel(PanelOLS, d, entity_effects=True)
        .fit(cov_type=v, **panel_cluster_config(d, v))
        .std_errors,
        "cov_type",
        PANEL_COV_TYPES,
    ),
    "PooledOLS.fit(cov_type)": Site(
        lambda d, v: panel(PooledOLS, d)
        .fit(cov_type=v, **panel_cluster_config(d, v))
        .std_errors,
        "cov_type",
        PANEL_COV_TYPES,
    ),
    "RandomEffects.fit(cov_type)": Site(
        lambda d, v: panel(RandomEffects, d)
        .fit(cov_type=v, **panel_cluster_config(d, v))
        .std_errors,
        "cov_type",
        PANEL_COV_TYPES,
    ),
    "FirstDifferenceOLS.fit(cov_type)": Site(
        lambda d, v: panel(FirstDifferenceOLS, d)
        .fit(cov_type=v, **panel_cluster_config(d, v))
        .std_errors,
        "cov_type",
        PANEL_COV_TYPES,
    ),
    "BetweenOLS.fit(cov_type)": Site(
        lambda d, v: panel(BetweenOLS, d)
        .fit(cov_type=v, **panel_cluster_config(d, v))
        .std_errors,
        "cov_type",
        BETWEEN_COV_TYPES,
    ),
    # panel/model.py
    "FamaMacBeth.fit(cov_type)": Site(
        lambda d, v: panel(FamaMacBeth, d).fit(cov_type=v).std_errors,
        "cov_type",
        FAMA_MACBETH_COV_TYPES,
    ),
    # panel/covariance.py, the kernel weights are looked up when the covariance is
    # created, which is in fit. The standard errors are read as well, so that the
    # names that are accepted are used to compute them
    "PanelOLS.fit(cov_type='kernel', kernel)": Site(
        lambda d, v: panel(PanelOLS, d, entity_effects=True)
        .fit(cov_type="kernel", kernel=v)
        .std_errors,
        "kernel",
        KERNELS,
    ),
    "PanelOLS.fit(cov_type='ac', kernel)": Site(
        lambda d, v: panel(PanelOLS, d, entity_effects=True)
        .fit(cov_type="ac", kernel=v)
        .std_errors,
        "kernel",
        KERNELS,
    ),
    "FamaMacBeth.fit(cov_type='kernel', kernel, bandwidth)": Site(
        lambda d, v: panel(FamaMacBeth, d)
        .fit(cov_type="kernel", kernel=v, bandwidth=2)
        .std_errors,
        "kernel",
        KERNELS,
    ),
    "FamaMacBeth.fit(cov_type='kernel', kernel)": Site(
        lambda d, v: panel(FamaMacBeth, d).fit(cov_type="kernel", kernel=v).std_errors,
        "kernel",
        KERNELS,
    ),
    # system/model.py
    "SUR.fit(cov_type)": Site(
        lambda d, v: SUR(d.equations).fit(cov_type=v, **cluster_config(d, v)),
        "cov_type",
        SYSTEM_COV_TYPES,
    ),
    "IV3SLS.fit(cov_type)": Site(
        lambda d, v: IV3SLS(d.iv_equations).fit(cov_type=v, **cluster_config(d, v)),
        "cov_type",
        SYSTEM_COV_TYPES,
    ),
    "IVSystemGMM(weight_type)": Site(
        lambda d, v: IVSystemGMM(d.iv_equations, weight_type=v),
        "weight_type",
        SYSTEM_GMM_TYPES,
    ),
    "IVSystemGMM.fit(cov_type)": Site(
        lambda d, v: IVSystemGMM(d.iv_equations).fit(cov_type=v),
        "cov_type",
        SYSTEM_GMM_TYPES,
    ),
    # asset_pricing/covariance.py, which the system models use for their kernels
    "SUR.fit(cov_type='kernel', kernel)": Site(
        lambda d, v: SUR(d.equations).fit(cov_type="kernel", kernel=v),
        "kernel",
        KERNELS,
    ),
    "IVSystemGMM(weight_type='kernel', kernel)": Site(
        lambda d, v: IVSystemGMM(d.iv_equations, weight_type="kernel", kernel=v),
        "kernel",
        KERNELS,
    ),
    "LinearFactorModel.fit(cov_type='kernel', kernel)": Site(
        lambda d, v: factor(LinearFactorModel, d).fit(cov_type="kernel", kernel=v),
        "kernel",
        KERNELS,
    ),
    # asset_pricing/model.py
    "TradedFactorModel.fit(cov_type)": Site(
        lambda d, v: factor(TradedFactorModel, d).fit(cov_type=v),
        "cov_type",
        ASSET_PRICING_COV_TYPES,
    ),
    "LinearFactorModel.fit(cov_type)": Site(
        lambda d, v: factor(LinearFactorModel, d).fit(cov_type=v),
        "cov_type",
        ASSET_PRICING_COV_TYPES,
    ),
    "LinearFactorModelGMM.fit(cov_type)": Site(
        lambda d, v: factor(LinearFactorModelGMM, d).fit(cov_type=v),
        "cov_type",
        ASSET_PRICING_COV_TYPES,
    ),
}


def expected_message(parameter, value, supported, suggestion=None, problem=None):
    """The message, written out in full and not made by the function that makes it"""
    names = ", ".join(repr(name) for name in sorted(supported))
    start = problem or f"Unknown {parameter}: {value!r}."
    hint = f" Did you mean {suggestion!r}?" if suggestion else ""
    return f"{start}{hint} The supported values of {parameter} are: {names}."


def check_error(call, parameter, value, supported, suggestion=None, problem=None):
    with pytest.raises(UnknownOptionError) as info:
        call()
    error = info.value
    # Code that catches either of the errors that were raised before 7.1 still works
    assert isinstance(error, ValueError)
    assert isinstance(error, KeyError)
    message = str(error)
    assert message == expected_message(parameter, value, supported, suggestion, problem)
    # It is not quoted, as the message of a KeyError is
    assert not message.startswith(("'", '"'))


@pytest.mark.parametrize("site", SITES.values(), ids=list(SITES))
def test_unknown_name(inputs, site):
    def call():
        site.call(inputs, "foo")

    check_error(call, site.parameter, "foo", site.supported)


@pytest.mark.parametrize("site", SITES.values(), ids=list(SITES))
def test_cluster_is_not_a_name_but_clustered_is(inputs, site):
    # The name that doc/source/iv/introduction.rst used before 7.1, and that users
    # of the issue tracker typed. It suggests "clustered" where that is supported
    def call():
        site.call(inputs, "cluster")

    suggestion = "clustered" if "clustered" in site.supported else None
    check_error(call, site.parameter, "cluster", site.supported, suggestion)


ACCEPTED = [
    pytest.param(site, name, id=f"{key}-{name}")
    for key, site in SITES.items()
    for name in site.supported
]


@pytest.mark.parametrize(("site", "name"), ACCEPTED)
def test_every_listed_name_is_accepted(inputs, site, name):
    # The message lists what the lookup accepts. A name that is listed and rejected, or
    # that is accepted and not listed, would make the message wrong
    site.call(inputs, name)


def test_iv_system_gmm_with_clustered_weights(inputs):
    # The IVSystemGMM does not support clustered weights, which was reported as a
    # KeyError for both the weights and the covariance
    clusters = inputs.clusters
    check_error(
        lambda: IVSystemGMM(inputs.iv_equations, weight_type="clustered"),
        "weight_type",
        "clustered",
        SYSTEM_GMM_TYPES,
    )
    model = IVSystemGMM(inputs.iv_equations)
    check_error(
        lambda: model.fit(cov_type="clustered", clusters=clusters),
        "cov_type",
        "clustered",
        SYSTEM_GMM_TYPES,
    )


def test_case_of_the_names_of_the_system_models(inputs):
    # The names are lower cased before they are looked up, and the message shows the
    # name in lower case if it is a cov_type of SUR and IV3SLS
    SUR(inputs.equations).fit(cov_type="Robust")
    IVSystemGMM(inputs.iv_equations, weight_type="Robust")
    check_error(
        lambda: SUR(inputs.equations).fit(cov_type="FOO"),
        "cov_type",
        "foo",
        SYSTEM_COV_TYPES,
    )


def test_case_of_kernel_names_of_the_system_and_asset_pricing_models(inputs):
    # They are lower cased before they are looked up, and the message shows the name
    # that was passed
    SUR(inputs.equations).fit(cov_type="kernel", kernel="Parzen")
    check_error(
        lambda: SUR(inputs.equations).fit(cov_type="kernel", kernel="Foo"),
        "kernel",
        "foo",
        KERNELS,
    )


def test_a_name_that_the_model_does_not_support(inputs):
    # The name exists, as an estimator of other panel models, but BetweenOLS does not
    # support it. The message lists the names that BetweenOLS supports
    problem = "Requested covariance estimator is not supported for the BetweenOLS."
    check_error(
        lambda: panel(BetweenOLS, inputs).fit(cov_type="driscoll-kraay"),
        "cov_type",
        "driscoll-kraay",
        BETWEEN_COV_TYPES,
        problem=problem,
    )
    check_error(
        lambda: panel(BetweenOLS, inputs).fit(cov_type="dk"),
        "cov_type",
        "dk",
        BETWEEN_COV_TYPES,
        problem=problem,
    )


def test_case_of_kernel_names_of_the_panel_models(inputs):
    # The names are not case sensitive. The message shows the name in lower case and
    # suggests a name that is close
    panel(FamaMacBeth, inputs).fit(cov_type="kernel", kernel="Parzen")
    check_error(
        lambda: panel(FamaMacBeth, inputs)
        .fit(cov_type="kernel", kernel="Parzan")
        .std_errors,
        "kernel",
        "parzan",
        KERNELS,
        suggestion="parzen",
    )
