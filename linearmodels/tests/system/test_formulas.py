from itertools import product

import numpy as np
import pytest
from pandas import Series, concat

from linearmodels import SUR, IVSystemGMM, IV3SLS
from linearmodels.compat.pandas import assert_series_equal, assert_frame_equal
from linearmodels.formula import sur, iv_system_gmm, iv_3sls
from linearmodels.tests.system._utility import generate_3sls_data_v2
from linearmodels.utility import AttrDict

data = generate_3sls_data_v2(k=2, const=False)
joined = []
for i, key in enumerate(data):
    eq = data[key]
    joined.append(Series(eq.dependent[:, 0], name='y{0}'.format(i + 1)))
    for j, col in enumerate(eq.exog.T):
        joined.append(Series(col, name='x{0}{1}'.format(i + 1, j + 1)))
    k = len(eq.exog.T)
    for j, col in enumerate(eq.endog.T):
        joined.append(Series(col, name='x{0}{1}'.format(i + 1, j + k + 1)))
    for j, col in enumerate(eq.instruments.T):
        joined.append(Series(col, name='z{0}{1}'.format(i + 1, j + 1)))
joined = concat(joined, 1)

fmlas = [
    {'eq1': 'y1 ~ x11 + x12', 'eq2': 'y2 ~ x21 + x22'},
    {'eq1': 'y1 ~ 1 + x11 + x12', 'eq2': 'y2 ~ 1 + x21 + x22'},
    {'eq1': 'y1 ~ 1 + x11 + np.exp(x12)', 'eq2': 'y2 ~ 1 + x21 + sigmoid(x22)'},
    {'eq1': 'y1 ~ 1 + x11 + [x14 + x15 ~ z11 + z12 + z13]', 'eq2': 'y2 ~ 1 + x21 + x22'},
    {'eq1': 'y1 ~ [x14 + x15 ~ 1 + x11 + x12 + x13 + z11 + z12 + z13]',
     'eq2': 'y2 ~ x21 + [x24 ~ 1 + z21 + z22 + z23]'}
]

models = ((SUR, sur), (IVSystemGMM, iv_system_gmm), (IV3SLS, iv_3sls))

params = list(product(fmlas, models))

ids = []
for f, m in params:
    key = '--'.join([value for value in f.values()])
    key += ' : ' + str(m[0].__name__)
    ids.append(key)


def sigmoid(v):
    return np.exp(v) / (1 + np.exp(v))


@pytest.fixture(scope='module', params=params, ids=ids)
def config(request):
    fmla, model_interace = request.param
    model, interface = model_interace
    return fmla, model, interface


def test_fromula(config):
    fmla, model, interface = config
    for key in fmla:
        if '[' in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    mod = model.from_formula(fmla, joined)
    mod_fmla = interface(fmla, joined)
    res = mod.fit()
    res_fmla = mod_fmla.fit()
    assert_series_equal(res.params, res_fmla.params)


def test_predict(config):
    fmla, model, interface = config
    for key in fmla:
        if '[' in fmla[key] and model not in (IVSystemGMM, IV3SLS):
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
        if '[' in fmla[key] and model not in (IVSystemGMM, IV3SLS):
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


def test_invalid_predict(config):
    fmla, model, interface = config
    for key in fmla:
        if '[' in fmla[key] and model not in (IVSystemGMM, IV3SLS):
            return
    mod = model.from_formula(fmla, joined)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.predict(data=joined, equations=mod._equations)
