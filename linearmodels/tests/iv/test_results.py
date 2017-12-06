import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.compat.pandas import assert_series_equal
from linearmodels.iv.data import IVData
from linearmodels.iv.model import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.tests.iv._utility import generate_data
from linearmodels.tests.panel._utility import assert_frame_similar


@pytest.fixture(scope='module')
def data():
    return generate_data()


@pytest.fixture(scope='module', params=[IV2SLS, IVLIML, IVGMM, IVGMMCUE])
def model(request):
    return request.param


def result_checker(res):
    for attr in dir(res):
        if attr.startswith('_') or attr in ('test_linear_constraint',):
            continue
        if attr == 'first_stage':
            result_checker(getattr(res, attr))
        attr = getattr(res, attr)
        if callable(attr):
            attr()
        else:
            attr
            str(attr)


def test_results(data, model):
    mod = model(data.dep, data.exog, data.endog, data.instr)
    result_checker(mod.fit(cov_type='unadjusted'))
    result_checker(mod.fit(cov_type='robust'))
    result_checker(mod.fit(cov_type='kernel'))
    result_checker(mod.fit(cov_type='clustered', clusters=data.clusters))

    result_checker(model(data.dep, data.exog, None, None).fit())


def test_results_single(data, model):
    mod = model(data.dep, data.exog[:, 0], data.endog[:, 0], data.instr[:, 0])
    result_checker(mod.fit(cov_type='unadjusted'))
    result_checker(mod.fit(cov_type='robust'))
    result_checker(mod.fit(cov_type='kernel'))
    result_checker(mod.fit(cov_type='clustered', clusters=data.clusters))


def test_results_no_exog(data, model):
    mod = model(data.dep, None, data.endog[:, 0], data.instr[:, 0])
    result_checker(mod.fit(cov_type='unadjusted'))
    result_checker(mod.fit(cov_type='robust'))
    result_checker(mod.fit(cov_type='kernel'))
    result_checker(mod.fit(cov_type='clustered', clusters=data.clusters))


def test_fitted_predict(data, model):
    mod = model(data.dep, None, data.endog, data.instr)
    res = mod.fit()
    assert_series_equal(res.idiosyncratic, res.resids)
    y = mod.dependent.pandas
    expected = y.values - res.resids.values[:, None]
    expected = pd.DataFrame(expected, y.index, ['fitted_values'])
    assert_frame_similar(expected, res.fitted_values)
    assert_allclose(expected, res.fitted_values)
    pred = res.predict()
    nobs = res.resids.shape[0]
    assert isinstance(pred, pd.DataFrame)
    assert pred.shape == (nobs, 1)
    pred = res.predict(idiosyncratic=True, missing=True)
    nobs = IVData(data.dep).pandas.shape[0]
    assert pred.shape == (nobs, 2)
    assert list(pred.columns) == ['fitted_values', 'residual']


def test_predict_no_selection(data, model):
    mod = model(data.dep, None, data.endog, data.instr)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.predict(fitted=False, idiosyncratic=False, missing=True)
