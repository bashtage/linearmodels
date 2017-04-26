import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.api import add_constant

from linearmodels.iv import IV2SLS, IVGMM
from linearmodels.utility import AttrDict

CWD = os.path.split(os.path.abspath(__file__))[0]

HOUSING_DATA = pd.read_csv(os.path.join(CWD, 'results', 'housing.csv'), index_col=0)
HOUSING_DATA.region = HOUSING_DATA.region.astype('category')
HOUSING_DATA.state = HOUSING_DATA.state.astype('category')
HOUSING_DATA.division = HOUSING_DATA.division.astype('category')

SIMULATED_DATA = pd.read_stata(os.path.join(CWD, 'results', 'simulated-data.dta'))


@pytest.fixture(scope='module')
def data():
    return AttrDict(dep=SIMULATED_DATA.y_robust,
                    exog=add_constant(SIMULATED_DATA[['x3', 'x4', 'x5']]),
                    endog=SIMULATED_DATA[['x1', 'x2']],
                    instr=SIMULATED_DATA[['z1', 'z2']])


def test_sargan(data):
    # Stata code:
    # ivregress 2sls y_robust x3 x4 x5 (x1=z1 z2)
    # estat overid
    res = IV2SLS(data.dep, data.exog, data.endog[['x1']], data.instr).fit(cov_type='unadjusted')
    assert_allclose(res.sargan.stat, .176535, rtol=1e-4)
    assert_allclose(res.sargan.pval, 0.6744, rtol=1e-4)


def test_basmann(data):
    # Stata code:
    # ivregress 2sls y_robust x3 x4 x5 (x1=z1 z2)
    # estat overid
    res = IV2SLS(data.dep, data.exog, data.endog[['x1']], data.instr).fit(cov_type='unadjusted')
    assert_allclose(res.basmann.stat, .174822, rtol=1e-4)
    assert_allclose(res.basmann.pval, 0.6759, rtol=1e-3)


def test_durbin(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit(cov_type='unadjusted')
    assert_allclose(res.durbin().stat, 35.1258, rtol=1e-4)
    assert_allclose(res.durbin().pval, 0.0000, atol=1e-6)

    assert_allclose(res.durbin('x1').stat, .156341, rtol=1e-4)
    assert_allclose(res.durbin('x1').pval, 0.6925, rtol=1e-3)


def test_wu_hausman(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit(cov_type='unadjusted')
    assert_allclose(res.wu_hausman().stat, 18.4063, rtol=1e-4)
    assert_allclose(res.wu_hausman().pval, 0.0000, atol=1e-6)

    assert_allclose(res.wu_hausman('x1').stat, .154557, rtol=1e-4)
    assert_allclose(res.wu_hausman('x1').pval, 0.6944, rtol=1e-3)


def test_wooldridge_score(data):
    res = IV2SLS(data.dep, data.exog, data.endog[['x1', 'x2']], data.instr).fit(cov_type='robust')
    assert_allclose(res.wooldridge_score.stat, 22.684, rtol=1e-4)
    assert_allclose(res.wooldridge_score.pval, 0.0000, atol=1e-4)


def test_wooldridge_regression(data):
    mod = IV2SLS(data.dep, data.exog, data.endog[['x1', 'x2']], data.instr)
    res = mod.fit(cov_type='robust', debiased=True)
    # Scale to correct for F vs Wald treatment
    assert_allclose(res.wooldridge_regression.stat, 2 * 13.3461, rtol=1e-4)
    assert_allclose(res.wooldridge_regression.pval, 0.0000, atol=1e-4)


def test_wooldridge_overid(data):
    res = IV2SLS(data.dep, data.exog, data.endog[['x1']], data.instr).fit(cov_type='robust')
    assert_allclose(res.wooldridge_overid.stat, 0.221648, rtol=1e-4)
    assert_allclose(res.wooldridge_overid.pval, 0.6378, rtol=1e-3)


def test_anderson_rubin(data):
    res = IV2SLS(data.dep, data.exog, data.endog[['x1']], data.instr).fit(cov_type='unadjusted')
    assert_allclose(res.nobs * (res._liml_kappa - 1), .176587, rtol=1e-4)


def test_basmann_f(data):
    res = IV2SLS(data.dep, data.exog, data.endog[['x1']], data.instr).fit(cov_type='unadjusted')
    assert_allclose(res.basmann_f.stat, .174821, rtol=1e-4)
    assert_allclose(res.basmann_f.pval, 0.6760, rtol=1e-3)


def test_c_stat_smoke(data):
    res = IVGMM(data.dep, data.exog, data.endog, data.instr).fit(cov_type='robust')
    c_stat = res.c_stat()
    assert_allclose(c_stat.stat, 22.684, rtol=1e-4)
    assert_allclose(c_stat.pval, 0.00, atol=1e-3)
    c_stat = res.c_stat(['x1'])
    assert_allclose(c_stat.stat, .158525, rtol=1e-3)
    assert_allclose(c_stat.pval, 0.6905, rtol=1e-3)
    # Final test
    c_stat2 = res.c_stat('x1')
    assert_allclose(c_stat.stat, c_stat2.stat)


def test_linear_restriction(data):
    res = IV2SLS(data.dep, data.exog, data.endog, data.instr).fit(cov_type='robust')
    nvar = len((res.params))
    q = np.eye(nvar)
    ts = res.test_linear_constraint(q, np.zeros(nvar))
    p = res.params.values[:, None]
    c = res.cov.values
    stat = float(p.T @ np.linalg.inv(c) @ p)
    assert_allclose(stat, ts.stat)
    assert ts.df == nvar
