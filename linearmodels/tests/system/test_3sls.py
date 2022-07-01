from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from linearmodels.system.model import IV3SLS
from linearmodels.tests.system._utility import (
    generate_3sls_data,
    generate_3sls_data_v2,
    simple_3sls,
)

nexog = [3, [1, 2, 3, 4, 5]]
nendog = [2, [1, 2, 1, 2, 1]]
ninstr = [3, 2, [2, 3, 2, 3, 2]]

const = [True, False]
rho = [0.8, 0.0]
common_exog = [True, False]
included_weights = [True, False]
output_dict = [True, False]
params = list(
    product(
        nexog, nendog, ninstr, const, rho, common_exog, included_weights, output_dict
    )
)

nexog = [[0, 1, 2]]
nendog = [[1, 0, 1]]
ninstr = [[2, 0, 1]]

# Explicitly test variables that have no columns
add_params = list(
    product(
        nexog, nendog, ninstr, const, rho, common_exog, included_weights, output_dict
    )
)

params += add_params


def gen_id(param):
    idstr = "homo" if isinstance(param[0], list) else "hetero"
    idstr += "-homo_endog" if isinstance(param[1], list) else "-hetero_endog"
    idstr += "-homo_instr" if isinstance(param[2], list) else "-hetero_instr"
    idstr += "-const" if param[3] else ""
    idstr += "-correl" if param[4] != 0 else ""
    idstr += "-common" if param[5] else ""
    idstr += "-weights" if param[6] else ""
    idstr += "-dict" if param[7] else "-tuple"
    return idstr


ids = [gen_id(param) for param in params]


@pytest.fixture(params=params, ids=ids)
def data(request):
    p, en, instr, const, rho, common_exog, included_weights, output_dict = request.param
    list_like = isinstance(p, list) or isinstance(en, list) or isinstance(instr, list)
    k = 4
    if common_exog and list_like:
        p = 3
        en = 2
        instr = 3
    elif list_like:

        def safe_len(a):
            a = np.array(a)
            if a.ndim == 0:
                return 0
            return len(a)

        k = max(map(safe_len, [p, en, instr]))

    return generate_3sls_data(
        n=250,
        k=k,
        p=p,
        en=en,
        instr=instr,
        const=const,
        rho=rho,
        common_exog=common_exog,
        included_weights=included_weights,
        output_dict=output_dict,
    )


def test_direct_simple(data):
    mod = IV3SLS(data)
    res = mod.fit(cov_type="unadjusted")

    y = []
    x = []
    z = []
    for key in data:
        val = data[key]
        if isinstance(val, tuple):
            y.append(val[0])
            nobs = val[0].shape[0]
            v1 = val[1] if val[1] is not None else np.empty((nobs, 0))
            v2 = val[2] if val[2] is not None else np.empty((nobs, 0))
            v3 = val[3] if val[3] is not None else np.empty((nobs, 0))
            x.append(np.concatenate([v1, v2], 1))
            z.append(np.concatenate([v1, v3], 1))
            if len(val) == 5:
                return  # weighted
        else:
            y.append(val["dependent"])
            nobs = val["dependent"].shape[0]
            vexog = val["exog"] if val["exog"] is not None else np.empty((nobs, 0))
            vendog = val["endog"] if val["endog"] is not None else np.empty((nobs, 0))
            vinstr = (
                val["instruments"]
                if val["instruments"] is not None
                else np.empty((nobs, 0))
            )
            x.append(np.concatenate([vexog, vendog], 1))
            z.append(np.concatenate([vexog, vinstr], 1))
            if "weights" in val:
                return  # weighted
    out = simple_3sls(y, x, z)
    assert_allclose(res.params.values, out.beta1.squeeze())
    assert_allclose(res.sigma, out.sigma)
    assert_allclose(np.asarray(res.resids), out.eps, atol=1e-4)
    assert_allclose(np.diag(res.cov), np.diag(out.cov))


def test_single_equation(data):
    key = list(data.keys())[0]
    data = {key: data[key]}

    mod = IV3SLS(data)
    res = mod.fit(cov_type="unadjusted")

    y = []
    x = []
    z = []
    for key in data:
        val = data[key]
        if isinstance(val, tuple):
            y.append(val[0])
            x.append(np.concatenate([val[1], val[2]], 1))
            z.append(np.concatenate([val[1], val[3]], 1))
            if len(val) == 5:
                return  # weighted
        else:
            y.append(val["dependent"])
            x.append(np.concatenate([val["exog"], val["endog"]], 1))
            z.append(np.concatenate([val["exog"], val["instruments"]], 1))
            if "weights" in val:
                return  # weighted
    out = simple_3sls(y, x, z)
    assert_allclose(res.params.values, out.beta1.squeeze())
    assert_allclose(res.sigma, out.sigma)
    assert_allclose(np.asarray(res.resids), out.eps)
    assert_allclose(np.diag(res.cov), np.diag(out.cov))


def test_too_few_instruments():
    n = 200
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 1))
    eqns = {}
    for i in range(2):
        eqns[f"eqn.{i}"] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_redundant_instruments():
    n = 200
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 1))
    instr = np.concatenate([exog, instr], 1)
    eqns = {}
    for i in range(2):
        eqns[f"eqn.{i}"] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_too_many_instruments():
    n = 50
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, n + 1))
    eqns = {}
    for i in range(2):
        eqns[f"eqn.{i}"] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_wrong_input_type():
    n = 200
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    endog = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 1))
    instr = np.concatenate([exog, instr], 1)
    eqns = []
    for i in range(2):
        eqns.append((dep[:, i], exog, endog, instr))
    with pytest.raises(TypeError):
        IV3SLS(eqns)

    eqns = {}
    for i in range(2):
        eqns[i] = (dep[:, i], exog, endog, instr)
    with pytest.raises(ValueError):
        IV3SLS(eqns)


def test_multivariate_iv():
    n = 250
    dep = np.random.standard_normal((n, 2))
    exog = np.random.standard_normal((n, 3))
    exog = DataFrame(exog, columns=[f"exog.{i}" for i in range(3)])
    endog = np.random.standard_normal((n, 2))
    endog = DataFrame(endog, columns=[f"endog.{i}" for i in range(2)])
    instr = np.random.standard_normal((n, 3))
    instr = DataFrame(instr, columns=[f"instr.{i}" for i in range(3)])
    eqns = {}
    for i in range(2):
        eqns[f"dependent.{i}"] = (dep[:, i], exog, endog, instr)
    mod = IV3SLS(eqns)
    res = mod.fit()

    common_mod = IV3SLS.multivariate_iv(dep, exog, endog, instr)
    common_res = common_mod.fit()

    assert_series_equal(res.params, common_res.params)


def test_multivariate_iv_bad_data():
    n = 250
    dep = np.random.standard_normal((n, 2))
    instr = np.random.standard_normal((n, 3))
    instr = DataFrame(instr, columns=[f"instr.{i}" for i in range(3)])

    with pytest.raises(ValueError):
        IV3SLS.multivariate_iv(dep, None, None, instr)


def test_fitted(data):
    mod = IV3SLS(data)
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
        columns=[key for key in res.equations],
    )
    assert_frame_equal(expected, res.fitted_values)


def test_no_exog():
    data = generate_3sls_data_v2(nexog=0, const=False)
    mod = IV3SLS(data)
    res = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, omitted="drop")
    mod = IV3SLS(data)
    res2 = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, omitted="empty")
    mod = IV3SLS(data)
    res3 = mod.fit()

    data = generate_3sls_data_v2(nexog=0, const=False, output_dict=False)
    mod = IV3SLS(data)
    res4 = mod.fit()

    data = generate_3sls_data_v2(
        nexog=0, const=False, output_dict=False, omitted="empty"
    )
    mod = IV3SLS(data)
    res5 = mod.fit()
    assert_series_equal(res.params, res2.params)
    assert_series_equal(res.params, res3.params)
    assert_series_equal(res.params, res4.params)
    assert_series_equal(res.params, res5.params)


def test_no_endog():
    data = generate_3sls_data_v2(nendog=0, ninstr=0)
    mod = IV3SLS(data)
    res = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, omitted="drop")
    mod = IV3SLS(data)
    res2 = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, omitted="empty")
    mod = IV3SLS(data)
    res3 = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, output_dict=False)
    mod = IV3SLS(data)
    res4 = mod.fit()

    data = generate_3sls_data_v2(nendog=0, ninstr=0, output_dict=False, omitted="empty")
    mod = IV3SLS(data)
    res5 = mod.fit()
    assert_series_equal(res.params, res2.params)
    assert_series_equal(res.params, res3.params)
    assert_series_equal(res.params, res4.params)
    assert_series_equal(res.params, res5.params)


def test_uneven_shapes():
    data = generate_3sls_data_v2()
    eq = data[list(data.keys())[0]]
    eq["weights"] = np.ones(eq.dependent.shape[0] // 2)
    with pytest.raises(ValueError):
        IV3SLS(data)


def test_predict_simultaneous_equations():
    rs = np.random.RandomState(1234)
    e = rs.standard_normal((40000, 2))
    x = rs.standard_normal((40000, 2))
    y2 = x[:, 1] / 2 - x[:, 0] / 2 + e[:, 1] / 2 - e[:, 0] / 2
    y1 = x[:, 1] / 2 + x[:, 0] / 2 + e[:, 1] / 2 + e[:, 0] / 2
    df = pd.DataFrame(np.column_stack([y1, y2, x]), columns=["y1", "y2", "x1", "x2"])
    data = {
        "y1": {
            "dependent": df.y1,
            "exog": df.x1,
            "endog": df.y2,
            "instruments": df.x2,
        },
        "y2": {
            "dependent": df.y2,
            "exog": df.x2,
            "endog": df.y1,
            "instruments": df.x1,
        },
    }
    res = IV3SLS(data).fit()
    base_pred = res.predict(dataframe=True)
    pred_data = {
        "y1": {
            "exog": df.x1,
            "endog": df.y2,
        },
        "y2": {
            "exog": df.x2,
            "endog": df.y1,
        },
    }
    exog_pred_data = res.predict(pred_data, dataframe=True)
    assert_frame_equal(base_pred, exog_pred_data)

    rs = np.random.RandomState(12345)
    e = rs.standard_normal((1000, 2))
    x = rs.standard_normal((1000, 2))
    y2 = x[:, 1] / 2 - x[:, 0] / 2 + e[:, 1] / 2 - e[:, 0] / 2
    y1 = x[:, 1] / 2 + x[:, 0] / 2 + e[:, 1] / 2 + e[:, 0] / 2
    df = pd.DataFrame(np.column_stack([y1, y2, x]), columns=["y1", "y2", "x1", "x2"])
    pred_data = {
        "y1": {
            "exog": df.x1,
            "endog": df.y2,
        },
        "y2": {
            "exog": df.x2,
            "endog": df.y1,
        },
    }
    exog_pred_data = res.predict(pred_data, dataframe=True)
    assert exog_pred_data.shape == (1000, 2)
    assert_allclose(
        res.params.y1_x1 * df.x1 + res.params.y1_y2 * df.y2, exog_pred_data.iloc[:, 0]
    )
    assert_allclose(
        res.params.y2_x2 * df.x2 + res.params.y2_y1 * df.y1, exog_pred_data.iloc[:, 1]
    )
