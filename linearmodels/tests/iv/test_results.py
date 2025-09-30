from numpy import asarray
from numpy.testing import assert_allclose
from pandas import DataFrame
from pandas.testing import assert_series_equal
import pytest

from linearmodels.iv.data import IVData
from linearmodels.iv.model import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.tests.iv._utility import generate_data
from linearmodels.tests.panel._utility import assert_frame_similar


@pytest.fixture(scope="module")
def data():
    return generate_data()


@pytest.fixture(scope="module", params=[IV2SLS, IVLIML, IVGMM, IVGMMCUE])
def model(request):
    return request.param


def result_checker(res):
    for attr in dir(res):
        if attr.startswith("_") or attr in ("test_linear_constraint", "wald_test"):
            continue
        print(attr)
        if attr == "summary":
            print(attr)
        if attr == "first_stage":
            result_checker(getattr(res, attr))
        _attr = getattr(res, attr)
        if callable(_attr):
            _attr()
        else:
            assert isinstance(_attr, object)
            str(_attr)


def test_results(data, model):
    mod = model(data.dep, data.exog, data.endog, data.instr)
    # OLS-like results
    result_checker(model(data.dep, data.exog, None, None).fit())

    result_checker(mod.fit(cov_type="unadjusted"))
    result_checker(mod.fit(cov_type="robust"))
    result_checker(mod.fit(cov_type="kernel"))
    result_checker(mod.fit(cov_type="clustered", clusters=data.clusters))


def test_results_single(data, model):
    mod = model(data.dep, data.exog[:, 0], data.endog[:, 0], data.instr[:, 0])
    result_checker(mod.fit(cov_type="unadjusted"))
    result_checker(mod.fit(cov_type="robust"))
    result_checker(mod.fit(cov_type="kernel"))
    result_checker(mod.fit(cov_type="clustered", clusters=data.clusters))


def test_results_no_exog(data, model):
    mod = model(data.dep, None, data.endog[:, 0], data.instr[:, 0])
    result_checker(mod.fit(cov_type="unadjusted"))
    result_checker(mod.fit(cov_type="robust"))
    result_checker(mod.fit(cov_type="kernel"))
    result_checker(mod.fit(cov_type="clustered", clusters=data.clusters))


def test_fitted_predict(data, model):
    mod = model(data.dep, None, data.endog, data.instr)
    res = mod.fit()
    assert_series_equal(res.idiosyncratic, res.resids)
    y = mod.dependent.pandas
    expected = asarray(y) - asarray(res.resids)[:, None]
    expected = DataFrame(expected, y.index, ["fitted_values"])
    assert_frame_similar(expected, res.fitted_values)
    assert_allclose(expected, res.fitted_values)
    pred = res.predict()
    nobs = res.resids.shape[0]
    assert isinstance(pred, DataFrame)
    assert pred.shape == (nobs, 1)
    pred = res.predict(idiosyncratic=True, missing=True)
    nobs = IVData(data.dep).pandas.shape[0]
    assert pred.shape == (nobs, 2)
    assert list(pred.columns) == ["fitted_values", "residual"]


def test_fitted_predict_exception(data, model):
    mod = model(data.dep, None, data.endog, data.instr)
    res = mod.fit()
    df = DataFrame([[1]])
    with pytest.raises(ValueError, match=r"Unable to use data when the model was "):
        res.predict(data=df)


def test_predict_no_selection(data, model):
    mod = model(data.dep, None, data.endog, data.instr)
    res = mod.fit()
    with pytest.raises(ValueError, match=r"At least one output must be selected"):
        res.predict(fitted=False, idiosyncratic=False, missing=True)
