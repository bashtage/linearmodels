"""
Every model with weights rejects weights that are not finite and strictly positive

Weights of inf were not rejected by the IV models and AbsorbingLS, which only checked
that the weights are positive: the weights were divided by their mean, which is infinite,
so that one weight was NaN and the others were 0, and the models failed when they were
estimated with ``LinAlgError("Singular matrix")``. The models share one check, with one
message that names the weights. A NaN is a missing value, and is dropped. The weights
are scaled to have a mean of 1, and the mean of very large weights overflows: weights of
1e308 became 0, and the scaled weights have to be finite for any finite weights.
"""

from contextlib import nullcontext

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML, AbsorbingLS
from linearmodels.iv.model import _OLS
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

NOBS = 120
MESSAGE = r"^weights must be strictly positive and finite\.$"
SYSTEM_MESSAGE = r"^weights in equation 'a' must be strictly positive and finite\.$"


class IVSample:
    """Data for the IV models and the system models"""

    def __init__(self, keep=None):
        rg = np.random.default_rng(20261012)
        nobs = NOBS
        self.exog = np.c_[np.ones(nobs), rg.standard_normal((nobs, 2))]
        self.instr = rg.standard_normal((nobs, 3))
        noise = rg.standard_normal(nobs)
        self.endog = (self.instr @ [0.6, 0.4, 0.2] + noise)[:, None]
        error = 0.5 * noise + rg.standard_normal(nobs)
        self.dep = (self.exog @ [1.0, 0.5, -0.3] + 0.4 * self.endog[:, 0] + error)[
            :, None
        ]
        self.dep2 = (self.exog @ [0.5, 0.2, 0.1] + rg.standard_normal(nobs))[:, None]
        self.groups = pd.DataFrame({"g": pd.Categorical(rg.integers(0, 6, nobs))})
        if keep is not None:
            self.exog, self.instr, self.endog = (
                self.exog[keep],
                self.instr[keep],
                self.endog[keep],
            )
            self.dep, self.dep2 = self.dep[keep], self.dep2[keep]
            self.groups = self.groups[keep].reset_index(drop=True)

    def take(self, keep):
        return type(self)(keep)

    def weights(self, values):
        return None if values is None else np.asarray(values, dtype=float)


class PanelSample:
    """Data for the panel models: 30 entities and 4 periods"""

    def __init__(self, keep=None):
        rg = np.random.default_rng(20261012)
        index = pd.MultiIndex.from_product(
            [np.arange(30), np.arange(4)], names=["entity", "time"]
        )
        nobs = len(index)
        assert nobs == NOBS
        self.x = pd.DataFrame(
            rg.standard_normal((nobs, 2)), index=index, columns=["a", "b"]
        )
        self.y = pd.Series(
            self.x @ [0.5, -0.3] + rg.standard_normal(nobs), index=index, name="y"
        )
        self.index = index
        if keep is not None:
            self.x, self.y = self.x[keep], self.y[keep]
            self.index = self.index[keep]

    def take(self, keep):
        return type(self)(keep)

    def weights(self, values):
        if values is None:
            return None
        return pd.Series(np.asarray(values, dtype=float), index=self.index)


def system_equations(d, w):
    first = {
        "dependent": d.dep,
        "exog": d.exog,
        "endog": d.endog,
        "instruments": d.instr,
    }
    second = {"dependent": d.dep2, "exog": d.exog}
    if w is not None:
        first["weights"] = w
    return {"a": first, "b": second}


MODELS = {
    "IV2SLS": (
        IVSample,
        lambda d, w: IV2SLS(d.dep, d.exog, d.endog, d.instr, weights=w),
    ),
    "IVLIML": (
        IVSample,
        lambda d, w: IVLIML(d.dep, d.exog, d.endog, d.instr, weights=w),
    ),
    "IVGMM": (IVSample, lambda d, w: IVGMM(d.dep, d.exog, d.endog, d.instr, weights=w)),
    "IVGMMCUE": (
        IVSample,
        lambda d, w: IVGMMCUE(d.dep, d.exog, d.endog, d.instr, weights=w),
    ),
    "IVJIVE": (
        IVSample,
        lambda d, w: IVJIVE(d.dep, d.exog, d.endog, d.instr, weights=w),
    ),
    "OLS": (IVSample, lambda d, w: _OLS(d.dep, d.exog, weights=w)),
    "AbsorbingLS": (
        IVSample,
        lambda d, w: AbsorbingLS(d.dep, d.exog[:, 1:], absorb=d.groups, weights=w),
    ),
    "SUR": (
        IVSample,
        lambda d, w: SUR(
            {
                key: {k: v for k, v in eq.items() if k not in ("endog", "instruments")}
                for key, eq in system_equations(d, w).items()
            }
        ),
    ),
    "IV3SLS": (IVSample, lambda d, w: IV3SLS(system_equations(d, w))),
    "IVSystemGMM": (IVSample, lambda d, w: IVSystemGMM(system_equations(d, w))),
    "PooledOLS": (PanelSample, lambda d, w: PooledOLS(d.y, d.x, weights=w)),
    "PanelOLS": (
        PanelSample,
        lambda d, w: PanelOLS(d.y, d.x, weights=w, entity_effects=True),
    ),
    "BetweenOLS": (PanelSample, lambda d, w: BetweenOLS(d.y, d.x, weights=w)),
    "FirstDifferenceOLS": (
        PanelSample,
        lambda d, w: FirstDifferenceOLS(d.y, d.x, weights=w),
    ),
    "RandomEffects": (PanelSample, lambda d, w: RandomEffects(d.y, d.x, weights=w)),
    "FamaMacBeth": (PanelSample, lambda d, w: FamaMacBeth(d.y, d.x, weights=w)),
}
SYSTEM = ("SUR", "IV3SLS", "IVSystemGMM")
# The models that are estimated without searching, which is quick
FIT = [name for name in MODELS if name != "IVGMMCUE"]
FIT_OPTIONS = {"AbsorbingLS": {"method": "lsmr"}}


def build(name, weights=None, keep=None):
    sample_class, builder = MODELS[name]
    sample = sample_class(keep)
    return builder(sample, sample.weights(weights))


def message(name):
    return SYSTEM_MESSAGE if name in SYSTEM else MESSAGE


def get_weights(mod):
    """All of the weights of a model as a vector, from a model or a system model"""
    if hasattr(mod, "_w") and isinstance(mod._w, list):
        return np.concatenate([np.ravel(w) for w in mod._w])
    weights = mod.weights
    if hasattr(weights, "values2d"):
        return np.ravel(weights.values2d)
    return np.ravel(weights.ndarray)


def extreme(kind, nobs=NOBS):
    if kind == "1e308":
        return np.full(nobs, 1e308)
    if kind == "1e-308":
        return np.full(nobs, 1e-308)
    if kind == "mixed 1e308 and 1e-308":
        return np.where(np.arange(nobs) % 2 == 0, 1e308, 1e-308)
    if kind == "mixed 1e300, 1 and 1e-300":
        return np.array([1e300, 1.0, 1e-300])[np.arange(nobs) % 3]
    if kind == "many scales":
        return 10.0 ** np.random.default_rng(3).uniform(-300, 300, nobs)
    raise NotImplementedError(kind)


EXTREME = [
    "1e308",
    "1e-308",
    "mixed 1e308 and 1e-308",
    "mixed 1e300, 1 and 1e-300",
    "many scales",
]


@pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["inf", "-inf"])
@pytest.mark.parametrize("name", MODELS)
def test_infinite_weights_are_rejected(name, bad):
    weights = np.ones(NOBS)
    weights[7] = bad
    with pytest.raises(ValueError, match=message(name)):
        build(name, weights)


@pytest.mark.parametrize("bad", [0.0, -1.0], ids=["zero", "negative"])
@pytest.mark.parametrize("name", MODELS)
def test_weights_that_are_not_positive_are_rejected(name, bad):
    weights = np.ones(NOBS)
    weights[7] = bad
    with pytest.raises(ValueError, match=message(name)):
        build(name, weights)


def test_the_equation_of_the_weights_is_named():
    # The weights of the second equation are the problem
    equations = system_equations(IVSample(), np.ones(NOBS))
    bad = np.ones(NOBS)
    bad[3] = np.inf
    equations["b"]["weights"] = bad
    with pytest.raises(ValueError, match=r"^weights in equation 'b' must be strictly"):
        IV3SLS(equations)


@pytest.mark.parametrize("name", FIT)
def test_missing_weights_are_dropped_with_their_observations(name):
    rg = np.random.default_rng(1)
    weights = 0.5 + rg.chisquare(3, NOBS)
    missing = [3, 17, 18, 100]
    weights_missing = weights.copy()
    weights_missing[missing] = np.nan
    keep = np.ones(NOBS, dtype=bool)
    keep[missing] = False
    options = FIT_OPTIONS.get(name, {})
    with pytest.warns(MissingValueWarning):
        res = build(name, weights_missing).fit(**options)
    expected = build(name, weights[keep], keep=keep).fit(**options)
    assert_allclose(res.params, expected.params, rtol=1e-8)
    assert_allclose(res.std_errors, expected.std_errors, rtol=1e-8)
    assert res.nobs == expected.nobs


@pytest.mark.parametrize("kind", EXTREME)
@pytest.mark.parametrize("name", MODELS)
def test_extreme_weights_are_finite_when_they_are_scaled(name, kind):
    # Half of the weights of the mixed 1e308 and 1e-308 are 0 once they are scaled,
    # which leaves cross-sections of Fama-MacBeth with too few observations
    if name == "FamaMacBeth" and kind == "mixed 1e308 and 1e-308":
        warns = pytest.warns(MissingValueWarning, match="cross-sectional regressions")
    else:
        warns = nullcontext()
    with warns:
        mod = build(name, extreme(kind))
    weights = get_weights(mod)
    all_finite = np.isfinite(weights).all()
    assert all_finite
    # The system models have 2 equations, and the weights are those of the first
    own = weights[:NOBS] if name in SYSTEM else weights
    assert_allclose(own.mean(), 1.0, rtol=1e-12)
    # The large weights are scaled and are not 0
    largest = own.max()
    assert largest > 0


@pytest.mark.parametrize("value", [1e308, 1e-308, 1e300, 5e-324])
@pytest.mark.parametrize("name", FIT)
def test_equal_extreme_weights_give_the_unweighted_model(name, value):
    # Scaling all of the weights by a number does not change the estimates
    options = FIT_OPTIONS.get(name, {})
    res = build(name, np.full(NOBS, value)).fit(**options)
    expected = build(name, None).fit(**options)
    finite = np.isfinite(res.params).all() and np.isfinite(res.std_errors).all()
    assert finite
    assert_allclose(res.params, expected.params, rtol=1e-8)
    assert_allclose(res.std_errors, expected.std_errors, rtol=1e-8)
    assert_allclose(res.rsquared, expected.rsquared, rtol=1e-8)


@pytest.mark.parametrize("name", FIT)
def test_scaling_the_weights_does_not_change_the_estimates(name):
    rg = np.random.default_rng(5)
    weights = 0.5 + rg.chisquare(3, NOBS)
    options = FIT_OPTIONS.get(name, {})
    expected = build(name, weights).fit(**options)
    for scale in (1e-250, 1e250):
        res = build(name, weights * scale).fit(**options)
        assert_allclose(res.params, expected.params, rtol=1e-7)
        assert_allclose(res.std_errors, expected.std_errors, rtol=1e-7)
