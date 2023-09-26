from itertools import product
import pickle

import numpy as np
from pandas import Series, concat
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from linearmodels import IV3SLS, SUR, IVSystemGMM
from linearmodels.formula import iv_3sls, iv_system_gmm, sur
from linearmodels.shared.utility import AttrDict
from linearmodels.system.model import SystemFormulaParser
from linearmodels.tests.system._utility import generate_3sls_data_v2

data = generate_3sls_data_v2(k=2, const=False)
parts = []
for i, key in enumerate(data):
    eq = data[key]
    parts.append(Series(eq.dependent[:, 0], name=f"y{i + 1}"))
    for j, col in enumerate(eq.exog.T):
        parts.append(Series(col, name=f"x{i + 1}{j + 1}"))
    k = len(eq.exog.T)
    for j, col in enumerate(eq.endog.T):
        parts.append(Series(col, name=f"x{i + 1}{j + k + 1}"))
    for j, col in enumerate(eq.instruments.T):
        parts.append(Series(col, name=f"z{i + 1}{j + 1}"))
joined = concat(parts, axis=1)

fmlas = [
    {"eq1": "y1 ~ x11 + x12", "eq2": "y2 ~ x21 + x22"},
    {"eq1": "y1 ~ 1 + x11 + x12", "eq2": "y2 ~ 1 + x21 + x22"},
    {"eq1": "y1 ~ 1 + x11 + np.exp(x12)", "eq2": "y2 ~ 1 + x21 + sigmoid(x22)"},
    {
        "eq1": "y1 ~ 1 + x11 + [x14 + x15 ~ z11 + z12 + z13]",
        "eq2": "y2 ~ 1 + x21 + x22",
    },
    {
        "eq1": "y1 ~ [x14 + x15 ~ 1 + x11 + x12 + x13 + z11 + z12 + z13]",
        "eq2": "y2 ~ x21 + [x24 ~ 1 + z21 + z22 + z23]",
    },
]

models = ((SUR, sur), (IVSystemGMM, iv_system_gmm), (IV3SLS, iv_3sls))

params = list(product(fmlas, models))

ids = []
for f, m in params:
    key = "--".join([value for value in f.values()])
    key += " : " + str(m[0].__name__)
    ids.append(key)


def sigmoid(v):
    return np.exp(v) / (1 + np.exp(v))


@pytest.fixture(scope="module", params=params, ids=ids)
def config(request):
    fmla, model_interace = request.param
    model, interface = model_interace
    return fmla, model, interface


def test_formula(config):
    fmla, model, interface = config
    for key in fmla:
        if "[" in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    mod = model.from_formula(fmla, joined)
    pmod = pickle.loads(pickle.dumps(mod))
    mod_fmla = interface(fmla, joined)
    res = mod.fit()
    ppres = pmod.fit()
    pres = pickle.loads(pickle.dumps(res))
    res_fmla = mod_fmla.fit()
    assert_series_equal(res.params, res_fmla.params)
    assert_series_equal(res.params, pres.params)
    assert_series_equal(res.params, ppres.params)


def test_predict(config):
    fmla, model, _ = config
    for key in fmla:
        if "[" in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    mod = model.from_formula(fmla, joined)
    res = mod.fit()
    pred = res.predict(data=joined)
    assert isinstance(pred, dict)
    pred2 = res.predict(data=joined, dataframe=True)
    pred3 = res.predict(equations=mod._equations, dataframe=True)
    assert_frame_equal(pred2, pred3)


def test_predict_partial(config):
    fmla, model, interface = config
    for key in fmla:
        if "[" in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    mod = model.from_formula(fmla, joined)
    res = mod.fit()
    eqns = AttrDict()
    for key in list(mod._equations.keys())[1:]:
        eqns[key] = mod._equations[key]
    pred = res.predict(equations=eqns, dataframe=True)
    for key in mod._equations:
        if key in eqns:
            assert key in pred
        else:
            assert key not in pred
    pred2 = res.predict(data=joined, dataframe=True)
    assert_frame_equal(pred2[pred.columns], pred)

    eqns = AttrDict()
    for key in list(mod._equations.keys())[1:]:
        eqns[key] = mod._equations[key]
    final = list(mod._equations.keys())[0]
    eqns[final] = {"exog": None, "endog": None}
    pred3 = res.predict(equations=eqns, dataframe=True)
    assert_frame_equal(pred2[pred3.columns], pred3)

    eqns = AttrDict()
    for key in mod._equations:
        eqns[key] = {k: v for k, v in mod._equations[key].items() if v is not None}
    pred4 = res.predict(equations=eqns, dataframe=True)
    assert_frame_equal(pred2, pred4)


def test_invalid_predict(config):
    fmla, model, interface = config
    for key in fmla:
        if "[" in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    mod = model.from_formula(fmla, joined)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.predict(data=joined, equations=mod._equations)


def test_parser(config):
    fmla, model, interface = config
    parser = SystemFormulaParser(fmla, joined, eval_env=1)
    orig_data = parser.data
    assert isinstance(orig_data, dict)
    assert parser.eval_env == 1

    alt_parser = SystemFormulaParser(fmla, joined, eval_env=0)
    alt_parser.eval_env = 1
    assert alt_parser.eval_env == 1
    exog = alt_parser.exog
    dep = alt_parser.dependent
    endog = alt_parser.endog
    instr = alt_parser.instruments
    for key in orig_data:
        eq = orig_data[key]
        if exog[key] is None:
            assert eq["exog"] is None
        else:
            assert_frame_equal(exog[key], eq["exog"])
        assert_frame_equal(dep[key], eq["dependent"])
        if endog[key] is None:
            assert eq["endog"] is None
        else:
            assert_frame_equal(endog[key], eq["endog"])
        if instr[key] is None:
            assert eq["instruments"] is None
        else:
            assert_frame_equal(instr[key], eq["instruments"])

    labels = parser.equation_labels
    for label in labels:
        assert label in orig_data
    new_parser = SystemFormulaParser(parser.formula, joined, eval_env=1)

    new_data = new_parser.data
    for key in orig_data:
        eq1 = orig_data[key]
        eq2 = new_data[key]
        for key in eq1:
            if eq1[key] is not None:
                assert_frame_equal(eq1[key], eq2[key])


def test_formula_escaped(config):
    fmla, model, interface = config
    for key in fmla:
        if "[" in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    replacements = {}
    cols = []
    for val in joined.columns:
        replacements[val] = f"`{val[0]} {val[1:]}`"
        cols.append(f"{val[0]} {val[1:]}")

    def fix_formula(fmla):
        mod_fmla = {}
        for key, eq in fmla.items():
            for orig in replacements:
                eq = eq.replace(orig, replacements[orig])
            mod_fmla[key] = eq

        return mod_fmla

    escaped_fmla = fix_formula(fmla)
    data = joined.copy()
    data.columns = cols

    mod = model.from_formula(escaped_fmla, data)
    pmod = pickle.loads(pickle.dumps(mod))
    mod_fmla = interface(escaped_fmla, data)
    res = mod.fit()
    ppres = pmod.fit()
    pres = pickle.loads(pickle.dumps(res))
    res_fmla = mod_fmla.fit()
    assert_series_equal(res.params, res_fmla.params)
    assert_series_equal(res.params, pres.params)
    assert_series_equal(res.params, ppres.params)
