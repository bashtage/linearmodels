import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.system import IV3SLS, SUR
from linearmodels.tests.system._utility import generate_simultaneous_data
from linearmodels.tests.system.results.parse_stata_3sls_results import results


@pytest.fixture(scope='module', params=list(results.keys()))
def fit(request):
    method = request.param
    data = generate_simultaneous_data()
    if 'ols' in method or 'sur' in method:
        mod = SUR
        for key in data:
            temp = data[key]
            temp['exog'] = pd.concat([temp['exog'], temp['endog']], 1)
            del temp['endog']
            del temp['instruments']
    else:
        mod = IV3SLS
    if 'ols' in method or '2sls' in method:
        fit_method = 'ols'
    else:
        fit_method = 'gls'
    mod = mod(data)
    iterate = 'ireg3' in method
    stata = results[method]
    debiased = method in ('ols', '2sls')
    kwargs = {}
    decimal = 2 if 'ireg3' in method else 5
    rtol = 10 ** -decimal
    res = mod.fit(cov_type='unadjusted', method=fit_method,
                  debiased=debiased, iterate=iterate, **kwargs)
    return stata, res, rtol


def test_params(fit):
    stata, result, rtol = fit
    for idx in result.params.index:
        val = result.params[idx]

        dep = '_'.join(idx.split('_')[:2])
        variable = '_'.join(idx.split('_')[2:])
        variable = '_cons' if variable == 'const' else variable
        stata_val = stata.params[dep].loc[variable, 'param']

        assert_allclose(stata_val, val, rtol=rtol)


def test_tstats(fit):
    stata, result, rtol = fit
    for idx in result.tstats.index:
        val = result.tstats[idx]

        dep = '_'.join(idx.split('_')[:2])
        variable = '_'.join(idx.split('_')[2:])
        variable = '_cons' if variable == 'const' else variable
        stata_val = stata.params[dep].loc[variable, 'tstat']
        assert_allclose(stata_val, val, rtol=rtol)


def test_pval(fit):
    stata, result, rtol = fit
    for idx in result.pvalues.index:
        val = result.pvalues[idx]

        dep = '_'.join(idx.split('_')[:2])
        variable = '_'.join(idx.split('_')[2:])
        variable = '_cons' if variable == 'const' else variable
        stata_val = stata.params[dep].loc[variable, 'pval']
        assert_allclose(1 + stata_val, 1 + val, rtol=rtol)


def test_sigma(fit):
    stata, result, rtol = fit
    assert_allclose(stata.sigma.values, result.sigma, rtol=rtol)
