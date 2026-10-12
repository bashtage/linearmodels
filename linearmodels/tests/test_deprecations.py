"""
Tests of the features that are deprecated and will be removed in linearmodels 8.0

Remove the tests of a feature when the feature is removed.
"""

import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import linearmodels
from linearmodels.iv import IV2SLS
from linearmodels.iv.absorbing import AbsorbingLS
from linearmodels.shared.exceptions import MissingValueWarning

LSMR_OPTIONS_MESSAGE = (
    "lsmr_options is deprecated and will be removed in linearmodels 8.0"
)
ORDERING = "linearmodels.__future__.ordering"
ORDERING_MESSAGE = (
    "linearmodels.__future__.ordering has no effect and will be removed in "
    "linearmodels 8.0"
)
DROP_MISSING_MESSAGE = (
    "linearmodels.DROP_MISSING has no effect and will be removed in linearmodels 8.0"
)
DROP_MISSING_ENV_MESSAGE = (
    "The environment variable LINEARMODELS_DROP_MISSING has no effect and will be "
    "removed in linearmodels 8.0"
)


@pytest.fixture
def absorbing_data():
    # Two crossed factors, so that LSMR needs more than one iteration
    rng = np.random.default_rng(20261011)
    nobs = 300
    cat = pd.DataFrame(
        {
            "a": pd.Series(rng.integers(0, 12, nobs), dtype="category"),
            "b": pd.Series(rng.integers(0, 9, nobs), dtype="category"),
        }
    )
    x = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["x0", "x1"])
    effects = cat["a"].cat.codes.to_numpy() + 0.5 * cat["b"].cat.codes.to_numpy()
    y = pd.Series(x.sum(axis=1).to_numpy() + effects + rng.standard_normal(nobs))
    return y, x, cat


def test_lsmr_options_warns_of_removal(absorbing_data):
    y, x, cat = absorbing_data
    mod = AbsorbingLS(y, x, absorb=cat)
    with pytest.warns(FutureWarning, match=re.escape(LSMR_OPTIONS_MESSAGE)) as record:
        mod.fit(method="lsmr", lsmr_options={})
    matches = [w for w in record if LSMR_OPTIONS_MESSAGE in str(w.message)]
    assert len(matches) == 1
    # The warning points at the call to fit
    assert Path(matches[0].filename).resolve() == Path(__file__).resolve()


def test_lsmr_options_docstring_says_when_it_is_removed():
    doc = AbsorbingLS.fit.__doc__
    if doc is None:
        pytest.skip("Docstrings are not available")
    # The text of the parameter, with whitespace normalized to ignore line breaks
    text = doc.partition("lsmr_options :")[2].partition("cov_config")[0]
    block = " ".join(text.split())
    assert ".. deprecated:: 4.17" in block
    assert "will be removed in linearmodels 8.0" in block


def test_lsmr_options_are_still_used(absorbing_data):
    y, x, cat = absorbing_data
    # One iteration is far from the solution that the default options find
    options = {"maxiter": 1}
    default = AbsorbingLS(y, x, absorb=cat).fit(method="lsmr", use_cache=False)
    expected = AbsorbingLS(y, x, absorb=cat).fit(
        method="lsmr", absorb_options=options, use_cache=False
    )
    assert np.max(np.abs(default.params - expected.params)) > 1e-3

    with pytest.warns(FutureWarning, match="lsmr_options"):
        res = AbsorbingLS(y, x, absorb=cat).fit(
            method="lsmr", lsmr_options=options, use_cache=False
        )
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.std_errors, expected.std_errors, rtol=1e-10)


@pytest.fixture
def forget_ordering():
    """Make the next import of the ordering module run the module again"""
    package = importlib.import_module("linearmodels.__future__")
    module = sys.modules.pop(ORDERING, None)
    attribute = package.__dict__.pop("ordering", None)
    yield
    sys.modules.pop(ORDERING, None)
    package.__dict__.pop("ordering", None)
    if module is not None:
        sys.modules[ORDERING] = module
    if attribute is not None:
        package.ordering = attribute


@pytest.mark.parametrize(
    "statement",
    [
        "import linearmodels.__future__.ordering",
        "import linearmodels.__future__.ordering as ordering",
        "from linearmodels.__future__ import ordering",
    ],
)
def test_ordering_warns_of_removal(forget_ordering, statement):
    # The module only runs on its first import, so run the statement as if it was
    # in another file, on line 2, to check where the warning points
    code = compile(f"\n{statement}\n", "importing_module.py", "exec")
    with pytest.warns(FutureWarning, match=re.escape(ORDERING_MESSAGE)) as record:
        exec(code, {})  # noqa: S102
    matches = [w for w in record if ORDERING_MESSAGE in str(w.message)]
    assert len(matches) == 1
    assert matches[0].filename == "importing_module.py"
    assert matches[0].lineno == 2


def test_ordering_warns_in_a_new_interpreter():
    # The module only runs on its first import in a process, so use a new interpreter
    # to check what a user sees. -W makes this independent of PYTHONWARNINGS
    code = "\nfrom linearmodels.__future__ import ordering\n"
    cmd = [sys.executable, "-W", "default::FutureWarning", "-c", code]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    assert f"<string>:2: FutureWarning: {ORDERING_MESSAGE}" in proc.stderr


@pytest.fixture
def restore_drop_missing():
    """Leave DROP_MISSING as it was: undefined, so that reading it warns"""
    namespace = vars(linearmodels)
    defined = "DROP_MISSING" in namespace
    original = namespace.get("DROP_MISSING")
    yield
    if defined:
        namespace["DROP_MISSING"] = original
    else:
        namespace.pop("DROP_MISSING", None)


def test_drop_missing_warns_when_it_is_read():
    with pytest.warns(FutureWarning, match=re.escape(DROP_MISSING_MESSAGE)) as record:
        value = linearmodels.DROP_MISSING
    # What 7.0 gave: the setting of the environment variable, True if it is not set
    legacy = os.environ.get("LINEARMODELS_DROP_MISSING", "1")
    assert value is (legacy not in ("", "0", "false", "False"))
    matches = [w for w in record if DROP_MISSING_MESSAGE in str(w.message)]
    assert len(matches) == 1
    # The warning points at the line that reads it
    assert Path(matches[0].filename).resolve() == Path(__file__).resolve()
    assert "WARN_ON_MISSING" in str(matches[0].message)


def test_drop_missing_warns_when_it_is_imported():
    # Run the statement as if it was in another file, on line 2, to check where the
    # warning points
    code = compile("\nfrom linearmodels import DROP_MISSING\n", "importing.py", "exec")
    with pytest.warns(FutureWarning, match=re.escape(DROP_MISSING_MESSAGE)) as record:
        exec(code, {})  # noqa: S102
    matches = [w for w in record if DROP_MISSING_MESSAGE in str(w.message)]
    assert matches
    assert all(w.filename == "importing.py" and w.lineno == 2 for w in matches)


def test_drop_missing_is_not_exported():
    # A name in __all__ would be read, and so warn, by import *
    assert "DROP_MISSING" not in linearmodels.__all__
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        exec("from linearmodels import *", namespace)  # noqa: S102
    assert "DROP_MISSING" not in namespace
    assert namespace["IV2SLS"] is IV2SLS


def test_assigning_drop_missing_does_not_warn(restore_drop_missing):
    vars(linearmodels).pop("DROP_MISSING", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        linearmodels.DROP_MISSING = False
        assert linearmodels.DROP_MISSING is False
    # Deleting what was assigned makes reading it warn again
    del linearmodels.DROP_MISSING
    with pytest.warns(FutureWarning, match=re.escape(DROP_MISSING_MESSAGE)):
        value = linearmodels.DROP_MISSING
    assert isinstance(value, bool)


@pytest.mark.parametrize("name", ["DROP_MISSINGS", "drop_missing", "not_an_attribute"])
def test_other_unknown_attributes_still_raise_attribute_error(name):
    message = f"module 'linearmodels' has no attribute '{name}'"
    with pytest.raises(AttributeError, match=re.escape(message)):
        getattr(linearmodels, name)
    assert not hasattr(linearmodels, name)
    assert getattr(linearmodels, name, 7) == 7


def two_stage_least_squares(y, exog, endog, instruments):
    x = np.column_stack([exog, endog])
    z = np.column_stack([exog, instruments])
    fitted = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    return np.linalg.solve(fitted.T @ x, fitted.T @ y)


@pytest.mark.parametrize("drop_missing", [True, False])
def test_drop_missing_has_no_effect(restore_drop_missing, drop_missing):
    rng = np.random.default_rng(531)
    nobs = 200
    instruments = rng.standard_normal((nobs, 2))
    endog = instruments @ [1.0, 0.5] + rng.standard_normal(nobs)
    exog = np.column_stack([np.ones(nobs), rng.standard_normal(nobs)])
    y = exog @ [1.0, 0.5] + endog + rng.standard_normal(nobs)
    complete = (y, exog, endog, instruments)
    y, endog, instruments = y.copy(), endog.copy(), instruments.copy()
    y[3] = np.nan
    endog[40] = np.nan
    instruments[77, 0] = np.nan
    keep = np.ones(nobs, dtype=bool)
    keep[[3, 40, 77]] = False
    expected = two_stage_least_squares(*(c[keep] for c in complete))

    linearmodels.DROP_MISSING = drop_missing
    with pytest.warns(MissingValueWarning):
        res = IV2SLS(y, exog, endog, instruments).fit()
    # The rows with a missing value are dropped, whatever DROP_MISSING is
    assert res.nobs == nobs - 3
    assert_allclose(res.params, expected, rtol=1e-10)


def run_python(*args, env_vars=None):
    """Run a new interpreter without the settings of linearmodels in its environment"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LINEARMODELS_")}
    env.update(env_vars or {})
    cmd = [sys.executable, *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def test_drop_missing_environment_variable_warns_in_a_new_interpreter():
    # The warning happens when linearmodels is first imported in a process, so use a
    # new interpreter to check what a user sees. The code is on line 2. -W makes this
    # independent of PYTHONWARNINGS
    code = "\nimport linearmodels\n"
    proc = run_python(
        "-W",
        "default::FutureWarning",
        "-c",
        code,
        env_vars={"LINEARMODELS_DROP_MISSING": "0"},
    )
    assert f"<string>:2: FutureWarning: {DROP_MISSING_ENV_MESSAGE}" in proc.stderr


# Imports linearmodels with the environment variable unset, then imports it again
# with each of the values in argv[1]. Prints the warnings about DROP_MISSING, and
# what reading DROP_MISSING returns, as JSON.
ENVIRONMENT_SCRIPT = """
import importlib
import json
import os
import sys
import warnings


def drop_missing_warnings(record):
    return [str(w.message) for w in record if "DROP_MISSING" in str(w.message)]


result = {}
with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always")
    import linearmodels
    # from-imports of submodules rely on __getattr__ raising AttributeError
    from linearmodels import datasets
    namespace = {}
    exec("from linearmodels import *", namespace)
result["plain import"] = drop_missing_warnings(record)
result["import * defines DROP_MISSING"] = "DROP_MISSING" in namespace

for value in json.loads(sys.argv[1]):
    os.environ["LINEARMODELS_DROP_MISSING"] = value
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        importlib.reload(linearmodels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drop_missing = linearmodels.DROP_MISSING
    result[value] = [drop_missing_warnings(record), drop_missing]
print(json.dumps(result))
"""

# What DROP_MISSING was in 7.0 for each value of LINEARMODELS_DROP_MISSING
LEGACY_DROP_MISSING = {
    "": False,
    "0": False,
    "false": False,
    "False": False,
    "1": True,
    "true": True,
    "no": True,
}


def test_drop_missing_environment_variable_in_a_new_interpreter():
    proc = run_python("-c", ENVIRONMENT_SCRIPT, json.dumps(list(LEGACY_DROP_MISSING)))
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    # No warning without the environment variable, also for import *
    assert result.pop("plain import") == []
    assert result.pop("import * defines DROP_MISSING") is False
    # Any value, including the ones that turned it off, warns once and keeps what it
    # gave in 7.0
    for value, expected in LEGACY_DROP_MISSING.items():
        messages, drop_missing = result.pop(value)
        assert len(messages) == 1
        assert DROP_MISSING_ENV_MESSAGE in messages[0]
        assert drop_missing is expected
    assert not result
