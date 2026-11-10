"""
Tests of how the models treat missing values, and of how to turn off the warning

A model drops the observations that have a missing value in any of its inputs and
issues a MissingValueWarning. The warning is controlled by
linearmodels.WARN_ON_MISSING, by the environment variable
LINEARMODELS_WARN_ON_MISSING, which sets it when linearmodels is imported, and by
warnings filters.
"""

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import linearmodels
from linearmodels import (
    IV2SLS,
    SUR,
    AbsorbingLS,
    FamaMacBeth,
    LinearFactorModel,
    PanelOLS,
)
from linearmodels.shared.exceptions import MissingValueWarning

NOBS = 120
# Each model has a missing value in three different rows
MISSING_ROWS = 3
MESSAGE = "Dropping rows with missing observations"


def iv_model(rng, missing):
    instruments = rng.standard_normal((NOBS, 2))
    endog = instruments @ [1.0, 0.5] + rng.standard_normal(NOBS)
    exog = np.column_stack([np.ones(NOBS), rng.standard_normal(NOBS)])
    y = exog @ [1.0, 0.5] + endog + rng.standard_normal(NOBS)
    if missing:
        y[3] = np.nan
        endog[40] = np.nan
        instruments[77, 0] = np.nan
    return IV2SLS(y, exog, endog, instruments)


def absorbing_model(rng, missing):
    cats = pd.DataFrame({"a": pd.Categorical(rng.integers(0, 6, NOBS))})
    x = rng.standard_normal((NOBS, 2))
    weights = rng.uniform(0.5, 1.5, NOBS)
    y = x @ [1.0, 0.5] + rng.standard_normal(NOBS)
    if missing:
        y[3] = np.nan
        x[40, 1] = np.nan
        weights[77] = np.nan
    return AbsorbingLS(y, x, absorb=cats, weights=weights)


def panel_data(rng, missing):
    index = pd.MultiIndex.from_product([range(12), range(10)], names=["i", "t"])
    x = pd.DataFrame(rng.standard_normal((NOBS, 2)), index=index, columns=["a", "b"])
    y = pd.Series(x.to_numpy() @ [1.0, 0.5] + rng.standard_normal(NOBS), index=index)
    if missing:
        y.iloc[3] = np.nan
        x.iloc[40, 0] = np.nan
        x.iloc[77, 1] = np.nan
    return y, x


def panel_model(rng, missing):
    return PanelOLS(*panel_data(rng, missing), entity_effects=True)


def fama_macbeth_model(rng, missing):
    return FamaMacBeth(*panel_data(rng, missing))


def system_model(rng, missing):
    y = rng.standard_normal((2, NOBS))
    x = rng.standard_normal((2, NOBS, 2))
    if missing:
        y[0, 3] = np.nan
        x[1, 40, 0] = np.nan
        y[1, 77] = np.nan
    equations = {f"eq{i}": {"dependent": y[i], "exog": x[i]} for i in range(2)}
    return SUR(equations)


def factor_model(rng, missing):
    factors = rng.standard_normal((NOBS, 2))
    portfolios = factors @ rng.standard_normal((2, 4)) + rng.standard_normal((NOBS, 4))
    if missing:
        portfolios[3, 1] = np.nan
        factors[40, 0] = np.nan
        portfolios[77, 3] = np.nan
    return LinearFactorModel(portfolios, factors)


MODELS = {
    "IV2SLS": iv_model,
    "AbsorbingLS": absorbing_model,
    "PanelOLS": panel_model,
    "FamaMacBeth": fama_macbeth_model,
    "SUR": system_model,
    "LinearFactorModel": factor_model,
}


def build(name, missing=True):
    rng = np.random.default_rng(20261012)
    return MODELS[name](rng, missing)


def nobs_after_dropping(missing=True):
    """The observations that are left, which are in each equation of a system"""
    return NOBS - MISSING_ROWS if missing else NOBS


@pytest.fixture(autouse=True)
def warn_on_missing(monkeypatch):
    # The tests do not depend on the environment variable, and the setting is
    # restored after each test
    monkeypatch.setattr(linearmodels, "WARN_ON_MISSING", True)


@pytest.mark.parametrize("name", MODELS)
def test_missing_values_are_dropped_with_a_warning(name):
    with pytest.warns(MissingValueWarning, match=MESSAGE):
        model = build(name)
    assert model.fit().nobs == nobs_after_dropping()


@pytest.mark.parametrize("name", MODELS)
def test_no_warning_if_nothing_is_missing(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingValueWarning)
        model = build(name, missing=False)
    assert model.fit().nobs == nobs_after_dropping(missing=False)


@pytest.mark.parametrize("name", MODELS)
def test_warn_on_missing_false_turns_the_warning_off(name, monkeypatch):
    monkeypatch.setattr(linearmodels, "WARN_ON_MISSING", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingValueWarning)
        model = build(name)
    # The observations are dropped as before
    assert model.fit().nobs == nobs_after_dropping()


@pytest.mark.parametrize("name", MODELS)
def test_warnings_filter_ignore(name):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", MissingValueWarning)
        model = build(name)
    assert not [w for w in record if issubclass(w.category, MissingValueWarning)]
    assert model.fit().nobs == nobs_after_dropping()


@pytest.mark.parametrize("name", MODELS)
def test_warnings_filter_error(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingValueWarning)
        with pytest.raises(MissingValueWarning, match=MESSAGE):
            build(name)


# Imports linearmodels with each of the values in argv[1] of the environment variable
# in turn, and prints WARN_ON_MISSING for each as JSON. A null value means that the
# variable is not set.
ENVIRONMENT_SCRIPT = """
import importlib
import json
import os
import sys

import linearmodels

result = {}
for value in json.loads(sys.argv[1]):
    if value is None:
        os.environ.pop("LINEARMODELS_WARN_ON_MISSING", None)
    else:
        os.environ["LINEARMODELS_WARN_ON_MISSING"] = value
    importlib.reload(linearmodels)
    result[json.dumps(value)] = linearmodels.WARN_ON_MISSING
print(json.dumps(result))
"""

# The warning is off for the values "", "0", "false" and "False", and on for any other
# value and if the variable is not set
WARN_ON_MISSING_FROM_ENVIRONMENT = {
    None: True,
    "": False,
    "0": False,
    "false": False,
    "False": False,
    "1": True,
    "true": True,
    "True": True,
    "FALSE": True,
    "no": True,
    "off": True,
}


def test_warn_on_missing_environment_variable():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LINEARMODELS_")}
    values = json.dumps(list(WARN_ON_MISSING_FROM_ENVIRONMENT))
    cmd = [sys.executable, "-c", ENVIRONMENT_SCRIPT, values]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = {json.dumps(k): v for k, v in WARN_ON_MISSING_FROM_ENVIRONMENT.items()}
    assert result == expected


def test_environment_variable_is_read_when_linearmodels_is_imported(monkeypatch):
    # Setting it later has no effect
    monkeypatch.setenv("LINEARMODELS_WARN_ON_MISSING", "0")
    with pytest.warns(MissingValueWarning, match=MESSAGE):
        build("IV2SLS")
