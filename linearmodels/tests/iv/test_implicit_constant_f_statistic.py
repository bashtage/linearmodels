"""The IV model F test excludes an intercept spanned by dummy columns."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from linearmodels.iv import IV2SLS, IVGMM, IVLIML


@pytest.mark.parametrize("model", [IV2SLS, IVLIML, IVGMM])
@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (-2.0, 5.0)])
def test_model_f_statistic_with_implicit_constant(
    model, debiased, cov_type, weighted, scales
):
    rng = np.random.default_rng(0)
    nobs = 200
    group = np.arange(nobs) % 2
    x = rng.standard_normal(nobs)
    z = rng.standard_normal((nobs, 2))
    endog = z @ np.array([0.8, 0.4]) + rng.standard_normal(nobs)
    y = 1.0 + 0.3 * group + 0.2 * x + 0.4 * endog + rng.standard_normal(nobs)
    exog = np.column_stack([scales[0] * (group == 0), scales[1] * (group == 1), x])
    weights = rng.uniform(0.5, 2.0, nobs) if weighted else None
    res = model(y, exog, endog, z, weights=weights).fit(
        cov_type=cov_type, debiased=debiased
    )
    # A constant fitted value requires equal group effects and zero slopes.
    restriction = np.array([[scales[0], -scales[1], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    expected = res.wald_test(restriction).stat / (3 if debiased else 1)
    assert res.f_statistic.df == 3
    assert_allclose(res.f_statistic.stat, expected, rtol=1e-10)
    explicit = model(
        y, np.column_stack([np.ones(nobs), group, x]), endog, z, weights=weights
    ).fit(cov_type=cov_type, debiased=debiased)
    assert_allclose(res.f_statistic.stat, explicit.f_statistic.stat, rtol=1e-10)
    assert_allclose(res.f_statistic.pval, explicit.f_statistic.pval, rtol=1e-10)
