"""
Tests of the features that are deprecated and will be removed in linearmodels 8.0

Remove the tests of a feature when the feature is removed.
"""

import importlib
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv.absorbing import AbsorbingLS

LSMR_OPTIONS_MESSAGE = (
    "lsmr_options is deprecated and will be removed in linearmodels 8.0"
)
ORDERING = "linearmodels.__future__.ordering"
ORDERING_MESSAGE = (
    "linearmodels.__future__.ordering has no effect and will be removed in "
    "linearmodels 8.0"
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
