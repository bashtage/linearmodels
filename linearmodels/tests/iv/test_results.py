import pytest

from linearmodels.iv.model import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.tests.iv._utility import generate_data


@pytest.fixture(scope='module')
def data():
    return generate_data()


@pytest.fixture(scope='module', params=[IV2SLS, IVLIML, IVGMM, IVGMMCUE])
def model(request):
    return request.param


def result_checker(res):
    for attr in dir(res):
        if attr.startswith('_') or attr in ('test_linear_constraint'):
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
