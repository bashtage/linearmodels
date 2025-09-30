from collections.abc import Mapping
from itertools import product
import warnings

import numpy as np
from numpy.testing import assert_allclose
from pandas import DataFrame, Series, concat
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import inv as spinv
import scipy.stats

from linearmodels.iv.model import _OLS as OLS
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.shared.utility import AttrDict
from linearmodels.system._utility import (
    blocked_column_product,
    blocked_diag_product,
    inv_matrix_sqrt,
)
from linearmodels.system.model import SUR
from linearmodels.tests.system._utility import generate_data, simple_sur

p = [3, [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]]
const = [True, False]
rho = [0.8, 0.0]
common_exog = [True, False]
included_weights = [True, False]
output_dict = [True, False]
params = list(product(p, const, rho, common_exog, included_weights, output_dict))


def gen_id(param):
    idstr = "homo" if isinstance(param[0], list) else "hetero"
    idstr += "-const" if param[1] else ""
    idstr += "-correl" if param[2] != 0 else ""
    idstr += "-common" if param[3] else ""
    idstr += "-weights" if param[4] else ""
    idstr += "-dist" if param[4] else "-tuple"
    return idstr


ids = [gen_id(param) for param in params]


def check_results(res1, res2):
    assert_allclose(res1.params, res2.params)
    assert_allclose(res1.rsquared, res2.rsquared)
    assert_allclose(res1.cov, res2.cov)
    assert_allclose(res1.pvalues, res2.pvalues, atol=1e-6)
    assert_allclose(res1.resids, res2.resids)
    assert_allclose(res1.wresids, res2.wresids)
    assert_allclose(res1.tstats, res2.tstats)
    assert_allclose(res1.std_errors, res2.std_errors)
    if hasattr(res1, "rsquared_adj"):
        assert_allclose(res1.rsquared_adj, res2.rsquared_adj)
    if hasattr(res1, "f_statistic"):
        assert_allclose(res1.f_statistic.stat, res2.f_statistic.stat)
        if res2.f_statistic.df_denom is None:
            # Do not test case of F dist due to DOF differences
            assert_allclose(res1.f_statistic.pval, res2.f_statistic.pval)


def get_res(res):
    d = filter(lambda s: not s.startswith("_"), dir(res))
    for attr in d:
        value = getattr(res, attr)
        if isinstance(value, Mapping):
            for key in value:
                get_res(value[key])


@pytest.fixture(params=params, ids=ids)
def data(request):
    p, const, rho, common_exog, included_weights, output_dict = request.param
    if common_exog and isinstance(p, list):
        p = 3
    return generate_data(
        p=p,
        const=const,
        rho=rho,
        common_exog=common_exog,
        included_weights=included_weights,
        output_dict=output_dict,
    )


@pytest.fixture(scope="module", params=[0, 0.1])
def missing_data(request):
    eqns = generate_data()
    np.random.seed(12345)
    missing = np.random.random_sample(500)
    missing = missing < request.param
    for key in eqns:
        eqns[key]["dependent"][missing] = np.nan
    return eqns


mvreg_params = list(product(const, rho, included_weights))


def mvreg_gen_id(param):
    idstr = "const" if param[0] else ""
    idstr += "-correl" if param[1] != 0 else ""
    idstr += "-weights" if param[2] else ""
    return idstr


mvreg_ids = [mvreg_gen_id(param) for param in mvreg_params]


@pytest.fixture(scope="module", params=mvreg_params, ids=mvreg_ids)
def mvreg_data(request):
    const, rho, included_weights = request.param
    values = generate_data(
        const=const, rho=rho, common_exog=True, included_weights=included_weights
    )
    dep = []
    for key in values:
        exog = values[key]["exog"]
        dep.append(values[key]["dependent"])
    return np.hstack(dep), exog


kernels = ["bartlett", "newey-west", "parzen", "gallant", "qs", "andrews"]
bandwidths = [None, 0, 10]
debiased = [True, False]
kernel_params = list(product(kernels, bandwidths, debiased))
kernel_ids = [
    p[0] + ", BW: " + str(p[1]) + ", Debiased: " + str(p[2]) for p in kernel_params
]


@pytest.fixture(params=kernel_params, ids=kernel_ids)
def kernel_options(request):
    return {
        "kernel": request.param[0],
        "bandwidth": request.param[1],
        "debiased": request.param[2],
    }


@pytest.mark.smoke
def test_smoke(data):
    mod = SUR(data)
    mod.fit()
    mod.fit(cov_type="unadjusted")
    mod.fit(cov_type="unadjusted", method="ols")
    res = mod.fit(full_cov=False)

    get_res(res)


def test_errors():
    with pytest.raises(TypeError, match=r"equations must be a dictionary-like"):
        SUR([])
    with pytest.raises(TypeError, match=r"Contents of each equation must be either"):
        SUR({"a": "absde", "b": 12345})

    moddata = {
        "a": {
            "dependent": np.random.standard_normal((100, 1)),
            "exog": np.random.standard_normal((100, 5)),
        }
    }
    mod = SUR(moddata)
    with pytest.raises(ValueError, match=r"Unknown cov_type"):
        mod.fit(cov_type="unknown")

    moddata = {
        "a": {
            "dependent": np.random.standard_normal((100, 1)),
            "exog": np.random.standard_normal((101, 5)),
        }
    }
    with pytest.raises(ValueError, match=r"Array required to have"):
        SUR(moddata)

    moddata = {
        "a": {
            "dependent": np.random.standard_normal((10, 1)),
            "exog": np.random.standard_normal((10, 20)),
        }
    }
    with pytest.raises(ValueError, match=r"Fewer observations than variables"):
        SUR(moddata)

    x = np.random.standard_normal((100, 2))
    x = np.c_[x, x]
    moddata = {"a": {"dependent": np.random.standard_normal((100, 1)), "exog": x}}
    with pytest.raises(ValueError, match=r"Equation `a` regressor array"):
        SUR(moddata)


@pytest.mark.smoke
def test_mv_reg_smoke(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    mod.fit()
    mod.fit(cov_type="unadjusted")
    res = mod.fit(cov_type="unadjusted", method="ols")
    assert res.method == "OLS"
    res = mod.fit(full_cov=False)

    get_res(res)


def test_formula():
    data = DataFrame(
        np.random.standard_normal((500, 4)), columns=["y1", "y2", "x1", "x2"]
    )
    formula = {"eq1": "y1 ~ 1 + x1", "eq2": "y2 ~ 1 + x2"}
    mod = SUR.from_formula(formula, data)
    mod.fit()

    formula = "{y1 ~ 1 + x1} {y2 ~ 1 + x2}"
    mod = SUR.from_formula(formula, data)
    mod.fit(cov_type="heteroskedastic")

    formula = """
    {y1 ~ 1 + x1}
    {y2 ~ 1 + x2}
    """
    mod = SUR.from_formula(formula, data)
    mod.fit(cov_type="heteroskedastic")

    formula = """
    {eq.a:y1 ~ 1 + x1}
    {second: y2 ~ 1 + x2}
    """
    mod = SUR.from_formula(formula, data)
    res = mod.fit(cov_type="heteroskedastic")
    assert "eq.a" in res.equation_labels
    assert "second" in res.equation_labels


# TODO: Implement weights
# TODO: 1. MV OLS and OLS (weighted) homo and hetero
# TODO: Implement observation dropping and check


def test_mv_ols_equivalence(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type="unadjusted")
    keys = res.equation_labels
    assert res.method == "OLS"

    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type="unadjusted", debiased=False)
        mv_res = res.equations[keys[i]]
        assert mv_res.method == "OLS"
        check_results(mv_res, ols_res)


def test_mv_ols_equivalence_robust(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type="robust")
    keys = res.equation_labels

    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type="robust", debiased=False)
        mv_res = res.equations[keys[i]]
        check_results(mv_res, ols_res)


def test_mv_ols_equivalence_debiased(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type="unadjusted", debiased=True)
    keys = res.equation_labels

    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type="unadjusted", debiased=True)
        mv_res = res.equations[keys[i]]
        check_results(mv_res, ols_res)


def test_mv_ols_equivalence_hetero_debiased(mvreg_data):
    dependent, exog = mvreg_data
    mod = SUR.multivariate_ls(dependent, exog)
    res = mod.fit(cov_type="robust", debiased=True)
    keys = res.equation_labels

    for i in range(dependent.shape[1]):
        ols_mod = OLS(dependent[:, i], exog)
        ols_res = ols_mod.fit(cov_type="robust", debiased=True)
        mv_res = res.equations[keys[i]]
        check_results(mv_res, ols_res)


def test_gls_eye_mv_ols_equiv(mvreg_data):
    dependent, exog = mvreg_data
    mv_mod = SUR.multivariate_ls(dependent, exog)
    mv_res = mv_mod.fit()
    keys = mv_res.equation_labels

    ad = AttrDict()
    for i in range(dependent.shape[1]):
        key = f"dependent.{i}"
        df = DataFrame(dependent[:, [i]], columns=[key])
        ad[key] = {"dependent": df, "exog": exog.copy()}
    gls_mod = SUR(ad, sigma=np.eye(len(ad)))
    gls_res = gls_mod.fit(method="gls")
    check_results(mv_res, gls_res)

    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)

    mv_res = mv_mod.fit(cov_type="robust")
    gls_res = gls_mod.fit(cov_type="robust", method="gls")
    check_results(mv_res, gls_res)

    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)

    mv_res = mv_mod.fit(cov_type="robust", debiased=True)
    gls_res = gls_mod.fit(cov_type="robust", method="gls", debiased=True)
    check_results(mv_res, gls_res)

    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)


def test_gls_without_mv_ols_equiv(mvreg_data):
    dependent, exog = mvreg_data
    mv_mod = SUR.multivariate_ls(dependent, exog)
    mv_res = mv_mod.fit()
    keys = mv_res.equation_labels

    ad = AttrDict()
    for i in range(dependent.shape[1]):
        key = f"dependent.{i}"
        df = DataFrame(dependent[:, [i]], columns=[key])
        ad[key] = {"dependent": df, "exog": exog.copy()}
    gls_mod = SUR(ad)
    gls_res = gls_mod.fit(method="ols")
    check_results(mv_res, gls_res)

    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)

    mv_res = mv_mod.fit(cov_type="robust")
    gls_res = gls_mod.fit(cov_type="robust", method="ols")
    check_results(mv_res, gls_res)

    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)

    mv_res = mv_mod.fit(cov_type="robust", debiased=True)
    gls_res = gls_mod.fit(cov_type="robust", method="ols", debiased=True)
    check_results(mv_res, gls_res)

    for i in range(dependent.shape[1]):
        mv_res_eq = mv_res.equations[keys[i]]
        gls_res_eq = gls_res.equations[keys[i]]
        check_results(mv_res_eq, gls_res_eq)


def test_ols_against_gls(data):
    mod = SUR(data)
    res = mod.fit(method="gls")
    if isinstance(data[next(iter(data.keys()))], dict):
        predictions = mod.predict(res.params, equations=data)
        predictions2 = mod.predict(np.asarray(res.params)[:, None], equations=data)
        assert_allclose(predictions, predictions2)
    sigma = res.sigma
    sigma_m12 = inv_matrix_sqrt(np.asarray(sigma))
    key = next(iter(data.keys()))

    if isinstance(data[key], Mapping):
        y = [data[key]["dependent"] for key in data]
        x = [data[key]["exog"] for key in data]
        try:
            w = [data[key]["weights"] for key in data]
        except KeyError:
            w = [np.ones_like(data[key]["dependent"]) for key in data]
    else:
        y = [data[key][0] for key in data]
        x = [data[key][1] for key in data]
        try:
            w = [data[key][2] for key in data]
        except IndexError:
            w = [np.ones_like(data[key][0]) for key in data]

    wy = [_y * np.sqrt(_w / _w.mean()) for _y, _w in zip(y, w, strict=False)]
    wx = [_x * np.sqrt(_w / _w.mean()) for _x, _w in zip(x, w, strict=False)]

    wy = blocked_column_product(wy, sigma_m12)
    wx = blocked_diag_product(wx, sigma_m12)

    ols_res = OLS(wy, wx).fit(debiased=False)
    assert_allclose(res.params, ols_res.params)


def test_constraint_setting(data):
    mod = SUR(data)
    pn = mod.param_names
    c = Series(np.zeros(len(pn)), index=pn)
    c1 = c.copy()
    c1.iloc[::7] = 1
    c2 = c.copy()
    c2.iloc[::11] = 1
    r = concat([c1, c2], axis=1).T
    q = Series([0, 1], index=r.index)

    mod.add_constraints(r)
    mod.fit(method="ols")
    res = mod.fit(method="ols", cov_type="unadjusted")
    assert_allclose(r.values @ res.params.values[:, None], np.zeros((2, 1)), atol=1e-8)
    mod.fit(method="gls")
    res = mod.fit(method="gls", cov_type="unadjusted")
    assert_allclose(r.values @ res.params.values[:, None], np.zeros((2, 1)), atol=1e-8)

    mod.add_constraints(r, q)
    res = mod.fit(method="ols")
    assert_allclose(r.values @ res.params.values[:, None], q.values[:, None], atol=1e-8)
    res = mod.fit(method="gls")
    assert_allclose(r.values @ res.params.values[:, None], q.values[:, None], atol=1e-8)


def test_invalid_constraints(data):
    # 1. Wrong types
    mod = SUR(data)
    pn = mod.param_names
    c = Series(np.zeros(len(pn)), index=pn)
    c1 = c.copy()
    c1.iloc[::7] = 1
    c2 = c.copy()
    c2.iloc[::11] = 1
    r = concat([c1, c2], axis=1).T
    q = Series([0, 1], index=r.index)
    with pytest.raises(TypeError, match=r"r must be a DataFram"):
        mod.add_constraints(r.values)
    with pytest.raises(TypeError, match=r"q must be a Series"):
        mod.add_constraints(r, q.values)

    # 2. Wrong shape
    with pytest.raises(ValueError, match=r"r is incompatible with the"):
        mod.add_constraints(r.iloc[:, :-2])
    with pytest.raises(ValueError, match=r"Constraint inputs are not shape"):
        mod.add_constraints(r, q.iloc[:-1])

    # 3. Redundant constraint
    r = concat([c1, c1], axis=1).T
    with pytest.raises(ValueError, match=r"Constraints must be non-redundant"):
        mod.add_constraints(r)

    # 4. Infeasible constraint
    with pytest.raises(ValueError, match=r"One or more constraints are"):
        mod.add_constraints(r, q)


def test_contrains_reset(data):
    mod = SUR(data)
    pn = mod.param_names
    c = Series(np.zeros(len(pn)), index=pn)
    c1 = c.copy()
    c1.iloc[::7] = 1
    c2 = c.copy()
    c2.iloc[::11] = 1
    r = concat([c1, c2], axis=1).T
    q = Series([0, 1], index=r.index)
    mod.add_constraints(r, q)
    cons = mod.constraints
    assert_allclose(np.asarray(cons.r), np.asarray(r))
    assert_allclose(np.asarray(cons.q), np.asarray(q))
    mod.reset_constraints()
    cons = mod.constraints
    assert cons is None


def test_missing(data):
    primes = [11, 13, 17, 19, 23]
    for i, key in enumerate(data):
        if isinstance(data[key], Mapping):
            data[key]["dependent"][:: primes[i % 5]] = np.nan
        else:
            data[key][0][:: primes[i % 5]] = np.nan

    with warnings.catch_warnings(record=True) as w:
        SUR(data)
        assert len(w) == 1
        assert "missing" in w[0].message.args[0]


def test_formula_errors():
    data = DataFrame(
        np.random.standard_normal((500, 4)), columns=["y1", "y2", "x1", "x2"]
    )
    with pytest.raises(TypeError, match=r"formula must be a string"):
        SUR.from_formula(np.ones(10), data)


def test_formula_repeated_key():
    data = DataFrame(
        np.random.standard_normal((500, 4)), columns=["y1", "y2", "x1", "x2"]
    )

    formula = """
    {first:y1 ~ 1 + x1}
    {first: y2 ~ 1 + x2}
    """
    mod = SUR.from_formula(formula, data)
    res = mod.fit()
    assert "first" in res.equation_labels
    assert "first.0" in res.equation_labels


def test_formula_weights():
    data = DataFrame(
        np.random.standard_normal((500, 4)), columns=["y1", "y2", "x1", "x2"]
    )
    weights = DataFrame(np.random.chisquare(5, (500, 2)), columns=["eq1", "eq2"])
    formula = {"eq1": "y1 ~ 1 + x1", "eq2": "y2 ~ 1 + x1"}
    mod = SUR.from_formula(formula, data, weights=weights)
    mod.fit()
    expected = weights.values[:, [0]]
    assert_allclose(mod._w[0], expected / expected.mean())
    expected = weights.values[:, [1]]
    assert_allclose(mod._w[1], expected / expected.mean())

    formula = "{y1 ~ 1 + x1} {y2 ~ 1 + x2}"
    weights = DataFrame(np.random.chisquare(5, (500, 2)), columns=["y1", "y2"])
    mod = SUR.from_formula(formula, data, weights=weights)
    mod.fit()
    expected = weights.values[:, [0]]
    assert_allclose(mod._w[0], expected / expected.mean())
    expected = weights.values[:, [1]]
    assert_allclose(mod._w[1], expected / expected.mean())


def test_formula_partial_weights():
    data = DataFrame(
        np.random.standard_normal((500, 4)), columns=["y1", "y2", "x1", "x2"]
    )
    weights = DataFrame(np.random.chisquare(5, (500, 1)), columns=["eq2"])
    formula = {"eq1": "y1 ~ 1 + x1", "eq2": "y2 ~ 1 + x1"}
    with warnings.catch_warnings(record=True) as w:
        mod = SUR.from_formula(formula, data, weights=weights)
        assert len(w) == 1
        assert "Weights" in w[0].message.args[0]
        assert "eq1" in w[0].message.args[0]
        assert "eq2" not in w[0].message.args[0]
    mod.fit()
    expected = np.ones((500, 1))
    assert_allclose(mod._w[0], expected / expected.mean())
    expected = weights.values[:, [0]]
    assert_allclose(mod._w[1], expected / expected.mean())

    formula = "{y1 ~ 1 + x1} {y2 ~ 1 + x2}"
    weights = DataFrame(np.random.chisquare(5, (500, 1)), columns=["y2"])
    with warnings.catch_warnings(record=True) as w:
        mod = SUR.from_formula(formula, data, weights=weights)
        assert len(w) == 1
        assert "y1" in w[0].message.args[0]
        assert "y2" not in w[0].message.args[0]

    expected = np.ones((500, 1))
    assert_allclose(mod._w[0], expected / expected.mean())
    expected = weights.values[:, [0]]
    assert_allclose(mod._w[1], expected / expected.mean())


def test_invalid_equation_labels(data):
    data = {i: data[key] for i, key in enumerate(data)}
    with pytest.raises(ValueError, match=r"Equation labels \(keys\)"):
        SUR(data)


def test_against_direct_model(data):
    keys = list(data.keys())
    if not isinstance(data[keys[0]], Mapping):
        return
    if "weights" in data[keys[0]]:
        return
    y = []
    x = []
    data_copy = {}
    for i in range(min(3, len(data))):
        data_copy[keys[i]] = data[keys[i]]
        y.append(data[keys[i]]["dependent"])
        x.append(data[keys[i]]["exog"])

    direct = simple_sur(y, x)
    mod = SUR(data_copy)
    res = mod.fit(method="ols")
    assert_allclose(res.params.values[:, None], direct.beta0)

    res = mod.fit(method="gls")
    assert_allclose(res.params.values[:, None], direct.beta1)


def test_restricted_f_statistic():
    data = generate_data(k=2, p=2)
    mod = SUR(data)
    r = DataFrame(np.zeros((1, 6)), columns=mod.param_names)
    r.iloc[0, 1] = 1.0
    mod.add_constraints(r)
    res = mod.fit()
    eqn = res.equations[res.equation_labels[0]]
    assert isinstance(eqn.f_statistic, InvalidTestStatistic)


def test_model_repr(data):
    mod = SUR(data)
    mod_repr = mod.__repr__()
    assert str(len(data)) in mod_repr
    assert str(hex(id(mod))) in mod_repr
    assert "Seemingly Unrelated Regression (SUR)" in mod_repr


@pytest.mark.smoke
def test_mv_ols_hac_smoke(kernel_options):
    data = generate_data(
        p=3,
        const=True,
        rho=0.8,
        common_exog=False,
        included_weights=False,
        output_dict=True,
    )
    mod = SUR(data)
    res = mod.fit(cov_type="kernel", **kernel_options)
    assert "Kernel (HAC) " in str(res)
    assert "Kernel: {}".format(kernel_options["kernel"]) in str(res)
    if kernel_options["bandwidth"] == 0:
        res_base = mod.fit(cov_type="robust", debiased=kernel_options["debiased"])
        assert_allclose(res.tstats, res_base.tstats)


def test_invalid_kernel_options(kernel_options):
    data = generate_data(
        p=3,
        const=True,
        rho=0.8,
        common_exog=False,
        included_weights=False,
        output_dict=True,
    )
    mod = SUR(data)
    ko = dict(kernel_options.items())
    ko["bandwidth"] = "None"
    with pytest.raises(TypeError, match=r"bandwidth must be either None"):
        mod.fit(cov_type="kernel", **ko)
    ko = dict(kernel_options.items())
    ko["kernel"] = 1
    with pytest.raises(TypeError, match=r"kernel must be the name of a kernel"):
        mod.fit(cov_type="kernel", **ko)


def test_fitted(data):
    mod = SUR(data)
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


@pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)
def test_predict(missing_data):
    mod = SUR(missing_data)
    res = mod.fit()
    pred_no_missing = res.predict(equations=missing_data, missing=False, dataframe=True)
    pred_missing = res.predict(equations=missing_data, missing=True, dataframe=True)
    assert pred_missing.shape[0] <= pred_no_missing.shape[0]
    pred = res.predict()
    for key in pred:
        assert_series_equal(
            pred[key].iloc[:, 0], res.equations[key].fitted_values, check_names=False
        )
    pred = res.predict(fitted=False, idiosyncratic=True)
    for key in pred:
        assert_series_equal(
            pred[key].iloc[:, 0], res.equations[key].resids, check_names=False
        )
    pred = res.predict(fitted=True, idiosyncratic=True)
    assert isinstance(pred, dict)
    for key in res.equations:
        assert key in pred

    pred = res.predict(dataframe=True)
    assert isinstance(pred, DataFrame)
    assert_frame_equal(pred, res.fitted_values)
    pred = res.predict(fitted=False, idiosyncratic=True, dataframe=True)
    assert isinstance(pred, DataFrame)
    assert_frame_equal(pred, res.resids)
    pred = res.predict(fitted=True, idiosyncratic=True, dataframe=True)
    assert isinstance(pred, dict)
    assert "fitted_values" in pred
    assert_frame_equal(pred["fitted_values"], res.fitted_values)
    assert "idiosyncratic" in pred
    assert_frame_equal(pred["idiosyncratic"], res.resids)

    nobs = missing_data[next(iter(missing_data.keys()))]["dependent"].shape[0]
    pred = res.predict(fitted=True, idiosyncratic=False, dataframe=True, missing=True)
    assert pred.shape[0] == nobs

    pred = res.predict(fitted=True, idiosyncratic=True, missing=True)
    for key in pred:
        assert pred[key].shape[0] == nobs


@pytest.mark.filterwarnings(
    "ignore::linearmodels.shared.exceptions.MissingValueWarning"
)
def test_predict_error(missing_data):
    mod = SUR(missing_data)
    res = mod.fit()
    with pytest.raises(ValueError, match=r"At least one output must be selected"):
        res.predict(fitted=False, idiosyncratic=False)


def reference_mcelroy(u, y, sigma):
    u = np.asarray(u)
    nobs = u.shape[0]
    sigma = np.asarray(sigma)
    y = np.asarray(y)
    u = u.T.ravel()
    y = y.T.ravel()
    sigma_inv = np.linalg.inv(sigma)
    omega_inv = np.kron(sigma_inv, np.eye(nobs))
    num = u @ omega_inv @ u
    iota = np.ones((nobs, 1))
    core = np.kron(sigma_inv, np.eye(nobs) - iota @ iota.T / nobs)
    denom = y @ core @ y

    return 1 - num / denom


def reference_berndt(u, y):
    u = np.asarray(u)
    nobs = u.shape[0]
    num = np.linalg.det(u.T @ u / nobs)
    y = np.asarray(y)
    mu = y.mean(0)
    y = y - mu
    denom = np.linalg.det(y.T @ y / nobs)
    return 1 - num / denom


def test_system_r2_direct():
    eqns = generate_data(k=3)
    mod = SUR(eqns)
    res = mod.fit(method="ols", cov_type="unadjusted")
    y = np.hstack([eqns[eq]["dependent"] for eq in eqns])
    ref = reference_mcelroy(res.resids, y, res.sigma)
    assert_allclose(ref, res.system_rsquared.mcelroy)
    ref = reference_berndt(res.resids, y)
    assert_allclose(ref, res.system_rsquared.berndt)

    res = mod.fit(method="gls", cov_type="unadjusted", iter_limit=100)
    y = np.hstack([eqns[eq]["dependent"] for eq in eqns])
    ref = reference_mcelroy(res.resids, y, res.sigma)
    assert_allclose(ref, res.system_rsquared.mcelroy)
    ref = reference_berndt(res.resids, y)
    assert_allclose(ref, res.system_rsquared.berndt, atol=1e-3, rtol=1e-3)


def direct_gls(eqns, scale):
    y = []
    x = []
    for key in eqns:
        y.append(eqns[key]["dependent"])
        x.append(eqns[key]["exog"])
    y = scale * np.vstack(y)

    n, k = x[0].shape
    _x = lil_matrix((len(x) * n, len(x) * k))
    for i, val in enumerate(x):
        _x[i * n : (i + 1) * n, i * k : (i + 1) * k] = val

    b = spinv(csc_matrix(_x.T @ _x)) @ (_x.T @ y)
    e = y - _x @ b
    e = e.reshape((-1, n)).T
    sigma = e.T @ e / n
    omega_inv = np.kron(np.linalg.inv(sigma), np.eye(n))
    b_gls = np.linalg.inv(_x.T @ omega_inv @ _x) @ (_x.T @ omega_inv @ y)
    xpx = _x.T @ omega_inv @ _x / n
    xpxi = np.linalg.inv(xpx)
    e = y - _x @ b_gls
    xe = (_x.T @ omega_inv).T * e
    _xe = np.zeros((n, len(x) * k))
    for i in range(len(x)):
        _xe += xe[i * n : (i + 1) * n]
    xeex = _xe.T @ _xe / n
    cov = xpxi @ xeex @ xpxi / n
    return b_gls, cov, xpx, xeex, _xe


@pytest.mark.parametrize("method", ["ols", "gls"])
@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "hac", "clustered"])
def test_tvalues_homogeneity(method, cov_type):
    eqns = generate_data(k=3)
    mod = SUR(eqns)
    kwargs = {}

    base = direct_gls(eqns, 1)
    base_tstat = np.squeeze(base[0]) / np.sqrt(np.diag(base[1]))
    base_100 = direct_gls(eqns, 1 / 100)
    base_100_tstat = np.squeeze(base_100[0]) / np.sqrt(np.diag(base_100[1]))
    assert_allclose(base_tstat, base_100_tstat)

    if cov_type == "hac":
        kwargs["bandwidth"] = 1
    elif cov_type == "clustered":
        key0 = next(iter(eqns.keys()))
        nobs = eqns[key0]["dependent"].shape[0]
        rs = np.random.RandomState(231823)
        kwargs["clusters"] = rs.randint(0, nobs // 5, size=(nobs, 1))
    res0 = mod.fit(method=method, cov_type=cov_type, **kwargs)
    for key in eqns:
        eqns[key]["dependent"] = eqns[key]["dependent"] / 100.0

    mod = SUR(eqns)
    res1 = mod.fit(method=method, cov_type=cov_type, **kwargs)
    assert_allclose(res0.tstats, res1.tstats)
    if cov_type == "robust" and method == "gls":
        assert_allclose(res0.tstats, base_tstat)
        assert_allclose(res1.tstats, base_100_tstat)


@pytest.mark.parametrize("k", [1, 3])
def test_brequsch_pagan(k):
    eqns = generate_data(k=k)
    mod = SUR(eqns)
    res = mod.fit()
    stat = res.breusch_pagan()
    if k == 1:
        assert isinstance(stat, InvalidTestStatistic)
        assert "Breusch-Pagan" in str(stat)
        assert np.isnan(stat.stat)
        return
    rho = np.asarray(res.resids.corr())
    nobs = res.resids.shape[0]
    direct = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            direct += rho[i, j] ** 2
    direct *= nobs
    assert isinstance(stat, WaldTestStatistic)
    assert_allclose(stat.stat, direct)
    assert stat.df == 3
    assert_allclose(stat.pval, 1.0 - scipy.stats.chi2(3).cdf(direct))
    assert "Residuals are uncorrelated" in stat.null
    assert "Breusch-Pagan" in str(stat)


@pytest.mark.parametrize("k", [1, 3])
def test_likelihood_ratio(k):
    eqns = generate_data(k=k)
    mod = SUR(eqns)
    res = mod.fit()
    stat = res.likelihood_ratio()
    if k == 1:
        assert isinstance(stat, InvalidTestStatistic)
        assert "Likelihood Ratio Test" in str(stat)
        assert np.isnan(stat.stat)
        return
    eps = np.asarray(res.resids)
    sigma = eps.T @ eps / eps.shape[0]
    nobs = res.resids.shape[0]
    direct = np.linalg.slogdet(sigma * np.eye(k))[1]
    direct -= np.linalg.slogdet(sigma)[1]
    direct *= nobs
    assert isinstance(stat, WaldTestStatistic)
    assert_allclose(stat.stat, direct)
    assert stat.df == 3
    assert_allclose(stat.pval, 1.0 - scipy.stats.chi2(3).cdf(direct))
    assert "Covariance is diagonal" in stat.null
    assert "Likelihood Ratio Test" in str(stat)


def test_unknown_method():
    mod = SUR(generate_data(k=3))
    with pytest.raises(ValueError, match=r"method must be 'ols' or 'gls'"):
        mod.fit(method="other")


def test_sur_contraint_with_value():
    n = 100
    rg = np.random.RandomState(np.random.MT19937(12345))
    x1 = rg.normal(size=n)
    x2 = rg.normal(size=n)
    x3 = rg.normal(size=n)

    y1 = 3 + 1.5 * x1 - 2.0 * x2 + np.random.normal(size=n)
    y2 = -1 + 0.5 * x2 + 1.2 * x3 + np.random.normal(size=n)

    data = DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})

    equations = {"eq1": "y1 ~ x1 + x2", "eq2": "y2 ~ x2 + x3"}

    model = SUR.from_formula(equations, data)

    # coefficients of eq1_x1 and eq2_x2 are equal
    r = DataFrame(
        [[0] * 4], columns=model.param_names, index=["rest"], dtype=np.float64
    )
    r.iloc[0, 0] = -1
    r.iloc[0, 2] = 1

    q = Series([0])
    model.add_constraints(r, q)
    result = model.fit()

    # No error without q
    model = SUR.from_formula(equations, data)
    model.add_constraints(r)
    result_without_q = model.fit()
    assert_allclose(result.params, result_without_q.params)
