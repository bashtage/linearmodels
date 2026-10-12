"""
predict with data evaluates the formula in the frame that calls it.

The default depth of IV model.predict was 4, which is the depth of results.predict, so
calling model.predict from the top level of a script raised "call stack is not deep
enough", and calling it from a nested function used the variables of another frame.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML

MODELS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]
FORMULA = "y ~ 1 + I(x * mult) + [e ~ z]"
MULT = 3.0


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((200, 4)), columns=["y", "x", "e", "z"])
    df["e"] = df["e"] + df["z"]
    return df


def expected_prediction(params, new):
    """The prediction from the parameters, using the formula by hand"""
    return (params.iloc[0] + params.iloc[1] * MULT * new["x"].to_numpy()) + params.iloc[
        2
    ] * new["e"].to_numpy()


def predict_in_this_frame(cls, df, new):
    """Fit a model whose formula uses a variable of this function, and predict"""
    mult = MULT  # noqa: F841  Used by the formula
    mod = cls.from_formula(FORMULA, df)
    res = mod.fit()
    return res.params, mod.predict(res.params, data=new), res.predict(data=new)


@pytest.mark.parametrize("cls", MODELS)
def test_predict_in_a_module_level_function(frame, cls):
    new = frame.iloc[:7]
    params, by_model, by_results = predict_in_this_frame(cls, frame, new)
    expected = expected_prediction(params, new)
    assert_allclose(by_model.to_numpy().squeeze(), expected, rtol=1e-12)
    assert_allclose(by_results.to_numpy().squeeze(), expected, rtol=1e-12)


@pytest.mark.parametrize("cls", MODELS)
def test_predict_in_nested_functions(frame, cls):
    new = frame.iloc[:7]

    def outer():
        def inner():
            return predict_in_this_frame(cls, frame, new)

        return inner()

    params, by_model, by_results = outer()
    expected = expected_prediction(params, new)
    assert_allclose(by_model.to_numpy().squeeze(), expected, rtol=1e-12)
    assert_allclose(by_results.to_numpy().squeeze(), expected, rtol=1e-12)


def test_predict_in_the_test_function(frame):
    # The variables of this function are used, and not those of a frame above it
    mult = MULT  # noqa: F841  Used by the formula
    new = frame.iloc[:7]
    mod = IV2SLS.from_formula(FORMULA, frame)
    res = mod.fit()
    expected = expected_prediction(res.params, new)
    assert_allclose(
        mod.predict(res.params, data=new).to_numpy().squeeze(), expected, rtol=1e-12
    )
    assert_allclose(res.predict(data=new).to_numpy().squeeze(), expected, rtol=1e-12)


def test_explicit_depth_counts_the_frames_of_the_parser(frame):
    # eval_env is counted from the formula parser, as it always was: 2 is the frame
    # that calls predict, and each wrapper function needs one more
    mult = MULT  # noqa: F841  Used by the formula
    new = frame.iloc[:7]
    mod = IV2SLS.from_formula(FORMULA, frame)
    res = mod.fit()
    expected = expected_prediction(res.params, new)

    def wrapper():
        return mod.predict(res.params, data=new, eval_env=3)

    assert_allclose(wrapper().to_numpy().squeeze(), expected, rtol=1e-12)
    same = mod.predict(res.params, data=new, eval_env=2)
    assert_allclose(same.to_numpy().squeeze(), expected, rtol=1e-12)


SCRIPT = """
import json

import numpy as np
import pandas as pd

from linearmodels.iv import IV2SLS

rng = np.random.default_rng(0)
df = pd.DataFrame(rng.standard_normal((200, 4)), columns=["y", "x", "e", "z"])
df["e"] = df["e"] + df["z"]
new = df.iloc[:7]
mult = 3.0
mod = IV2SLS.from_formula("y ~ 1 + I(x * mult) + [e ~ z]", df)
res = mod.fit()
out = {"params": res.params.tolist()}
out["model"] = mod.predict(res.params, data=new).to_numpy().squeeze().tolist()
out["results"] = res.predict(data=new).to_numpy().squeeze().tolist()


def nested():
    def deeper():
        mult = 3.0
        m = IV2SLS.from_formula("y ~ 1 + I(x * mult) + [e ~ z]", df)
        r = m.fit()
        return m.predict(r.params, data=new), r.predict(data=new)

    return deeper()


by_model, by_results = nested()
out["nested_model"] = by_model.to_numpy().squeeze().tolist()
out["nested_results"] = by_results.to_numpy().squeeze().tolist()
print(json.dumps(out))
"""


def test_predict_at_the_top_level_of_a_script(tmp_path):
    script = tmp_path / "predict_script.py"
    script.write_text(textwrap.dedent(SCRIPT))
    done = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((200, 4)), columns=["y", "x", "e", "z"])
    df["e"] = df["e"] + df["z"]
    expected = expected_prediction(pd.Series(out["params"]), df.iloc[:7])
    for key in ("model", "results", "nested_model", "nested_results"):
        assert_allclose(out[key], expected, rtol=1e-12, err_msg=key)
