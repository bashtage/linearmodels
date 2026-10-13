from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from pandas import DataFrame, Series, concat
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from linearmodels.iv.covariance import kernel_weight_parzen
from linearmodels.shared.exceptions import DefaultBandwidthWarning
from linearmodels.shared.utility import AttrDict
from linearmodels.system import IV3SLS, IVSystemGMM
from linearmodels.system.gmm import (
    HeteroskedasticWeightMatrix,
    HomoskedasticWeightMatrix,
    KernelWeightMatrix,
)
from linearmodels.tests._utility import warns_if_default_bandwidth
from linearmodels.tests.system._utility import (
    generate_3sls_data,
    generate_3sls_data_v2,
    simple_gmm,
    weighted_fit_measures,
)

params = list(product([1, 2], [True, False]))


def gen_id(r):
    _id = f"steps:{r[0]}"
    if r[1]:
        _id += ",robust"
    else:
        _id += ",unadjusted"
    return _id


ids = [gen_id(param) for param in params]


@pytest.fixture(scope="module", params=params, ids=ids)
def data(request):
    steps, robust = request.param
    weight_type = "robust" if robust else "unadjusted"
    eqns = generate_3sls_data_v2(k=3)
    y = [eqns[key].dependent for key in eqns]
    x = [np.concatenate([eqns[key].exog, eqns[key].endog], 1) for key in eqns]
    z = [np.concatenate([eqns[key].exog, eqns[key].instruments], 1) for key in eqns]

    return AttrDict(
        eqns=eqns, x=x, y=y, z=z, steps=steps, robust=robust, weight_type=weight_type
    )


@pytest.fixture(scope="module")
def weight_data():
    eqns = generate_3sls_data_v2(k=2)
    mod = IV3SLS(eqns)
    x = mod._x
    z = mod._z
    res = mod.fit(cov_type="unadjusted")
    eps = np.asarray(res.resids)
    sigma = np.asarray(res.sigma)
    return x, z, eps, sigma


@pytest.fixture(params=[True, False])
def center(request):
    return request.param


@pytest.fixture(params=[True, False])
def debias(request):
    return request.param


def test_params(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    if data.steps == 1:
        beta = simple.beta0
    else:
        beta = simple.beta1
    assert_allclose(res.params.values, beta, rtol=1e-4)


def test_weights(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    w = simple.w0 if data.steps == 1 else simple.w1
    assert_allclose(res.w, w, rtol=1e-4)


def test_cov(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    assert_allclose(np.asarray(res.cov), simple.cov)


def test_formula_equivalence(data):
    mod = IVSystemGMM(data.eqns, weight_type="unadjusted")
    formula = []
    df = []
    for i, key in enumerate(data.eqns):
        eqn = data.eqns[key]
        dep = eqn.dependent
        ex = eqn.exog
        en = eqn.endog
        instr = eqn.instruments
        dep = DataFrame(dep, columns=[f"dep_{i}"])
        has_const = False
        if np.any(np.all(ex == 1, 0)):
            ex = ex[:, 1:]
            has_const = True
        ex = DataFrame(ex, columns=[f"ex_{i}_{j}" for j in range(ex.shape[1])])
        en = DataFrame(en, columns=[f"en_{i}_{j}" for j in range(en.shape[1])])
        instr = DataFrame(instr, columns=[f"instr_{i}_{j}" for j in range(ex.shape[1])])
        fmla = "".join(dep.columns) + " ~  "
        if has_const:
            fmla += " 1 + "
        fmla += " + ".join(ex.columns) + " + ["
        fmla += " + ".join(en.columns) + " ~ "
        fmla += " + ".join(instr.columns) + " ] "
        formula.append(fmla)
        df.extend([dep, ex, en, instr])

    formulas = {}
    for i, f in enumerate(formula):
        formulas[f"eq{i}"] = f
    df = concat(df, axis=1, sort=False)
    formula_mod = IVSystemGMM.from_formula(formulas, df, weight_type="unadjusted")
    res = mod.fit(cov_type="unadjusted")
    formula_res = formula_mod.fit(cov_type="unadjusted")
    assert_allclose(res.params, formula_res.params)


def test_formula_equivalence_weights(data):
    weights = {}
    eqn_copy = AttrDict()
    for key in data.eqns:
        eqn = dict(data.eqns[key].items())
        nobs = eqn["dependent"].shape[0]
        w = np.random.chisquare(2, (nobs, 1)) / 2
        weights[key] = w
        eqn["weights"] = w
        eqn_copy[key] = eqn

    mod = IVSystemGMM(eqn_copy, weight_type="unadjusted")
    df = []
    formulas = {}
    for i, key in enumerate(data.eqns):
        eqn = data.eqns[key]
        dep = eqn.dependent
        ex = eqn.exog
        en = eqn.endog
        instr = eqn.instruments
        dep = DataFrame(dep, columns=[f"dep_{i}"])
        has_const = False
        if np.any(np.all(ex == 1, 0)):
            ex = ex[:, 1:]
            has_const = True
        ex = DataFrame(ex, columns=[f"ex_{i}_{j}" for j in range(ex.shape[1])])
        en = DataFrame(en, columns=[f"en_{i}_{j}" for j in range(en.shape[1])])
        instr = DataFrame(instr, columns=[f"instr_{i}_{j}" for j in range(ex.shape[1])])
        fmla = "".join(dep.columns) + " ~  "
        if has_const:
            fmla += " 1 + "
        fmla += " + ".join(ex.columns) + " + ["
        fmla += " + ".join(en.columns) + " ~ "
        fmla += " + ".join(instr.columns) + " ] "
        formulas[key] = fmla
        df.extend([dep, ex, en, instr])
    df = concat(df, axis=1, sort=False)
    formula_mod = IVSystemGMM.from_formula(
        formulas, df, weights=weights, weight_type="unadjusted"
    )
    res = mod.fit(cov_type="unadjusted")
    formula_res = formula_mod.fit(cov_type="unadjusted")
    assert_allclose(res.params, formula_res.params)


def test_weight_options(data):
    mod = IVSystemGMM(data.eqns, weight_type="unadjusted", debiased=True, center=True)
    res = mod.fit(cov_type="unadjusted")
    assert res.weight_config == {"debiased": True, "center": True}
    assert res.weight_type == "unadjusted"
    assert "Debiased: True" in str(res.summary)
    assert str(hex(id(res._weight_estimtor))) in res._weight_estimtor.__repr__()
    assert res._weight_estimtor.config == {"debiased": True, "center": True}
    base_res = IVSystemGMM(data.eqns, weight_type="unadjusted").fit(
        cov_type="unadjusted"
    )
    assert np.all(np.diag(res.w) >= np.diag(base_res.w))

    mod = IVSystemGMM(data.eqns, weight_type="robust", debiased=True)
    mod.fit(cov_type="robust")


@pytest.mark.smoke
def test_no_constant_smoke():
    eqns = generate_3sls_data_v2(k=3, const=False)
    mod = IVSystemGMM(eqns)
    mod.fit()


def test_unknown_weight_type(data):
    with pytest.raises(ValueError, match=r"Unknown weight_type"):
        IVSystemGMM(data.eqns, weight_type="unknown")


def test_unknown_cov_type(data):
    mod = IVSystemGMM(data.eqns)
    with pytest.raises(ValueError, match=r"Unknown cov_type"):
        mod.fit(cov_type="unknown")
    with pytest.raises(ValueError, match=r"Unknown cov_type"):
        mod.fit(cov_type=3)


def test_initial_weight_matrix(data):
    mod = IVSystemGMM(data.eqns)
    z = [
        np.concatenate([data.eqns[key].exog, data.eqns[key].instruments], 1)
        for key in data.eqns
    ]
    z = np.concatenate(z, 1)
    ze = z + np.random.standard_normal(size=z.shape)
    w0 = ze.T @ ze / ze.shape[0]
    res0 = mod.fit(initial_weight=w0, iter_limit=1)
    res = mod.fit(iter_limit=1)
    assert np.any(res0.params != res.params)


def test_summary(data):
    mod = IVSystemGMM(data.eqns)
    res = mod.fit()
    assert "Instruments" in res.summary.as_text()
    assert "Weight Estimator" in res.summary.as_text()
    for eq in res.equations:
        assert "Weight Estimator" in res.equations[eq].summary.as_text()
        assert "Instruments" in res.equations[eq].summary.as_text()

    res = mod.fit(iter_limit=10)
    if res.iterations > 2:
        assert "Iterative System GMM" in res.summary.as_text()


def test_summary_homoskedastic(data):
    mod = IVSystemGMM(data.eqns, weight_type="unadjusted", debiased=True)
    res = mod.fit(cov_type="homoskedastic", debiased=True)
    assert "Homoskedastic (Unadjusted) Weighting" in res.summary.as_text()


def test_fixed_sigma(data):
    mod = IVSystemGMM(data.eqns, weight_type="unadjusted")
    res = mod.fit(cov_type="unadjusted")
    k = len(data.eqns)
    b = np.random.standard_normal((k, 1))
    sigma = b @ b.T + np.diag(np.ones(k))
    mod_sigma = IVSystemGMM(data.eqns, weight_type="unadjusted", sigma=sigma)
    res_sigma = mod_sigma.fit()
    assert np.any(res.params != res_sigma.params)
    assert np.any(np.asarray(res.sigma != res_sigma.sigma))


def test_incorrect_sigma_shape(data):
    k = len(data.eqns)
    b = np.random.standard_normal((k + 2, 1))
    sigma = b @ b.T + np.diag(np.ones(k + 2))
    with pytest.raises(ValueError, match=r"sigma must be a square matrix"):
        IVSystemGMM(data.eqns, weight_type="unadjusted", sigma=sigma)


def test_invalid_sigma_usage(data):
    k = len(data.eqns)
    b = np.random.standard_normal((k, 1))
    sigma = b @ b.T + np.diag(np.ones(k))
    with pytest.warns(UserWarning, match=r"sigma has been provided"):
        IVSystemGMM(data.eqns, weight_type="robust", sigma=sigma)


def test_j_statistic_direct(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    res = mod.fit(cov_type=data.weight_type, iter_limit=data.steps)
    simple = simple_gmm(data.y, data.x, data.z, data.robust, steps=data.steps)
    assert_allclose(res.j_stat.stat, simple.j_stat, rtol=1e-4)


def test_linear_constraint(data):
    mod = IVSystemGMM(data.eqns, weight_type=data.weight_type)
    p = mod.param_names
    r = DataFrame(np.zeros((1, len(p))), index=[0], columns=p)
    r.iloc[0, 1::6] = 1
    q = Series([6])
    mod.add_constraints(r, q)

    res = mod.fit()
    assert_allclose(res.params.iloc[1::6].sum(), 6)


def test_kernel_equiv(data):
    mod = IVSystemGMM(data.eqns, weight_type="kernel", bandwidth=0)
    res = mod.fit(cov_type="kernel", debiased=True, bandwidth=0)
    assert "Kernel (HAC) Weighting" in res.summary.as_text()
    rob_mod = IVSystemGMM(data.eqns, weight_type="robust")
    rob_res = rob_mod.fit(cov_type="robust", debiased=True)
    assert_allclose(res.tstats, rob_res.tstats)


def test_kernel_optimal_bandwidth(data):
    # nobs - 2 was the default until 7.1, and is restored by optimal_bw=False
    mod = IVSystemGMM(data.eqns, weight_type="kernel", optimal_bw=False)
    res = mod.fit(cov_type="kernel", debiased=True)
    nobs = data.eqns[next(iter(data.eqns.keys()))].dependent.shape[0]
    assert res.weight_config["bandwidth"] == (nobs - 2)

    mod = IVSystemGMM(data.eqns, weight_type="kernel", optimal_bw=True)
    optimal = mod.fit(cov_type="kernel", debiased=True)
    assert 0 <= optimal.weight_config["bandwidth"] < nobs - 2

    # The default estimates the bandwidth from the data, and warns
    mod = IVSystemGMM(data.eqns, weight_type="kernel")
    with pytest.warns(DefaultBandwidthWarning, match=r"7\.1"):
        default = mod.fit(cov_type="kernel", debiased=True)
    assert default.weight_config["bandwidth"] == optimal.weight_config["bandwidth"]
    assert_allclose(default.params, optimal.params)


@pytest.mark.parametrize(
    "options",
    [{}, {"bandwidth": 5}, {"optimal_bw": False}, {"optimal_bw": True}],
    ids=["default", "bandwidth", "not_optimal_bw", "optimal_bw"],
)
def test_kernel_weights_with_one_step(options):
    # The first step does not use the weight matrix, so its bandwidth is not
    # known. It was an AssertionError if the bandwidth was not given
    eqns = generate_3sls_data_v2(k=2, n=200)
    with warns_if_default_bandwidth(**options):
        res = IVSystemGMM(eqns, weight_type="kernel", **options).fit(iter_limit=1)
    assert res.weight_config["bandwidth"] == options.get("bandwidth")
    assert res.weight_config["center"] is False
    robust = IVSystemGMM(eqns, weight_type="robust").fit(iter_limit=1)
    assert_allclose(res.params, robust.params, rtol=1e-12)
    assert_allclose(res.w, robust.w, rtol=1e-12)
    assert_allclose(res.j_stat.stat, robust.j_stat.stat, rtol=1e-10)


def test_homoskedastic_weight_direct(weight_data, center, debias):
    wm = HomoskedasticWeightMatrix(center, debias)
    x, z, eps, sigma = weight_data
    weights = wm.weight_matrix(x, z, eps, sigma=sigma)
    nobs = x[0].shape[0]
    big_z = []
    k = len(z)
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(z[j])
            else:
                row.append(np.zeros_like(z[j]))
        big_z.append(np.concatenate(row, 1))
    big_z = np.concatenate(big_z, 0)
    direct = big_z.T @ np.kron(sigma, np.eye(nobs)) @ big_z / nobs
    assert_allclose(weights, direct)


def test_heteroskedastic_weight_direct(weight_data, center, debias):
    wm = HeteroskedasticWeightMatrix(center, debias)
    x, z, eps, sigma = weight_data
    weights = wm.weight_matrix(x, z, eps, sigma=sigma)
    k = len(z)
    ze = [z[i] * eps[:, i : i + 1] for i in range(k)]
    ze = np.concatenate(ze, 1)
    if center:
        ze = ze - ze.mean(0)
    nobs = ze.shape[0]
    direct = ze.T @ ze / nobs
    if debias:
        df = [vx.shape[1] * np.ones(vz.shape[1]) for vx, vz in zip(x, z, strict=False)]
        df = np.concatenate(df)[:, None]
        df = np.sqrt(df)
        adj = nobs / (nobs - df @ df.T)
        direct *= adj
    assert_allclose(weights, direct)


def test_kernel_weight_direct(weight_data, center, debias):
    bandwidth = 12
    wm = KernelWeightMatrix(center, debias, kernel="parzen", bandwidth=bandwidth)
    x, z, eps, sigma = weight_data
    weights = wm.weight_matrix(x, z, eps, sigma=sigma)
    k = len(z)
    ze = [z[i] * eps[:, i : i + 1] for i in range(k)]
    ze = np.concatenate(ze, 1)
    if center:
        ze = ze - ze.mean(0)
    nobs = ze.shape[0]
    direct = ze.T @ ze / nobs
    w = kernel_weight_parzen(bandwidth)
    for i in range(1, bandwidth + 1):
        op = ze[:-i].T @ ze[i:] / nobs
        direct += w[i] * (op + op.T)
    if debias:
        df = [vx.shape[1] * np.ones(vz.shape[1]) for vx, vz in zip(x, z, strict=False)]
        df = np.concatenate(df)[:, None]
        df = np.sqrt(df)
        adj = nobs / (nobs - df @ df.T)
        direct *= adj
    assert_allclose(weights, direct)


def test_fitted(data):
    mod = IVSystemGMM(data.eqns)
    res = mod.fit()
    expected = []
    for i, key in enumerate(res.equations):
        eq = res.equations[key]
        fv = res.fitted_values[key].copy()
        fv.name = "fitted_values"
        assert_series_equal(eq.fitted_values, fv)
        b = eq.params.values
        direct = mod._x[i] @ b
        expected.append(direct[:, None])
        assert_allclose(eq.fitted_values, direct, atol=1e-8)
    expected = np.concatenate(expected, 1)
    expected = DataFrame(
        expected,
        index=mod._dependent[i].pandas.index,
        columns=list(res.equations),
    )
    assert_frame_equal(expected, res.fitted_values)


@pytest.mark.parametrize("weight_type", ["robust", "unadjusted"])
def test_weighted_residuals_and_r2(weight_type):
    eqns = generate_3sls_data(
        n=250,
        k=3,
        p=3,
        en=2,
        instr=3,
        const=True,
        rho=0.8,
        common_exog=False,
        included_weights=True,
        output_dict=True,
    )
    res = IVSystemGMM(eqns, weight_type=weight_type).fit()
    rss = tss = 0.0
    for key, eqn in eqns.items():
        eq_res = res.equations[key]
        direct = weighted_fit_measures(eqn, eq_res.params)
        # The residuals are those of the model and do not use the weights
        assert_allclose(eq_res.resids, direct.resid.squeeze())
        assert_allclose(eq_res.wresids, direct.wresid.squeeze())
        assert_allclose(
            eq_res.fitted_values + eq_res.resids, eqn["dependent"].squeeze()
        )
        assert_allclose(eq_res.resid_ss, direct.rss)
        assert_allclose(eq_res.rsquared, 1 - direct.rss / direct.tss)
        rss += direct.rss
        tss += direct.tss
    assert_allclose(res.rsquared, 1 - rss / tss)
