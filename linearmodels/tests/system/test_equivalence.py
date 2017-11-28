from linearmodels import IV3SLS, IVSystemGMM, IV2SLS, IVGMM
from linearmodels.tests.system._utility import generate_3sls_data_v2
from numpy.testing import assert_allclose


def test_gmm_3sls_equiv():
    eqns = generate_3sls_data_v2(k=3)
    gmm = IVSystemGMM(eqns).fit(iter_limit=1)
    tsls = IV3SLS(eqns).fit(method='ols')
    assert_allclose(gmm.params, tsls.params)


def test_3sls_2sls_equiv():
    eqns = generate_3sls_data_v2(k=1)
    tsls_mod = IV3SLS(eqns)
    tsls = tsls_mod.fit(method='ols', cov_type='unadjusted', debiased=False)
    eqn = eqns[list(eqns.keys())[0]]
    ivmod = IV2SLS(eqn.dependent, eqn.exog, eqn.endog, eqn.instruments)
    iv = ivmod.fit(cov_type='unadjusted', debiased=False)
    assert_allclose(iv.params, tsls.params)
    assert_allclose(iv.tstats, tsls.tstats)
    assert_allclose(iv.rsquared, tsls.rsquared)

    tsls = tsls_mod.fit(method='ols', cov_type='unadjusted', debiased=True)
    iv = ivmod.fit(cov_type='unadjusted', debiased=True)
    assert_allclose(iv.tstats, tsls.tstats)

    tsls = tsls_mod.fit(method='ols', cov_type='robust', debiased=False)
    iv = ivmod.fit(cov_type='robust', debiased=False)
    assert_allclose(iv.tstats, tsls.tstats)


def test_gmm_equiv():
    eqns = generate_3sls_data_v2(k=1)
    sys_mod = IVSystemGMM(eqns)
    eqn = eqns[list(eqns.keys())[0]]
    gmm_mod = IVGMM(eqn.dependent, eqn.exog, eqn.endog, eqn.instruments)
    sys_res = sys_mod.fit()
    gmm_res = gmm_mod.fit()

    assert_allclose(sys_res.params, gmm_res.params)
    assert_allclose(sys_res.rsquared, gmm_res.rsquared)
    assert_allclose(sys_res.tstats, gmm_res.tstats)
