from __future__ import annotations

from linearmodels.compat.pandas import ANNUAL_FREQ

from typing import Literal

import numpy as np
from numpy.linalg import lstsq
from numpy.random import RandomState, standard_normal
from numpy.testing import assert_allclose
import pandas
from pandas import Categorical, DataFrame, Series, date_range, get_dummies
from pandas.testing import assert_frame_equal, assert_series_equal

from linearmodels.panel.data import PanelData
from linearmodels.shared.utility import AttrDict, panel_to_frame

try:
    import xarray  # noqa: F401

    MISSING_XARRAY = False
except ImportError:
    MISSING_XARRAY = True

datatypes = ["numpy", "pandas"]
if not MISSING_XARRAY:
    datatypes += ["xarray"]


def lsdv(
    y: pandas.DataFrame,
    x: pandas.DataFrame,
    has_const=False,
    entity=False,
    time=False,
    general=None,
):
    nvar = x.shape[1]
    temp = x.reset_index()
    cat_index = temp.index
    if entity:
        cat = Series(Categorical(temp.iloc[:, 0]))
        cat.index = cat_index
        dummies = get_dummies(cat, drop_first=has_const)
        x = DataFrame(
            np.c_[x.values, dummies.values.astype(np.float64)],
            index=x.index,
            columns=list(x.columns) + list(dummies.columns),
        )
    if time:
        cat = Series(Categorical(temp.iloc[:, 1]))
        cat.index = cat_index
        dummies = get_dummies(cat, drop_first=(has_const or entity))
        x = DataFrame(
            np.c_[x.values, dummies.values.astype(np.float64)],
            index=x.index,
            columns=list(x.columns) + list(dummies.columns),
        )
    if general is not None:
        cat = Series(Categorical(general))
        cat.index = cat_index
        dummies = get_dummies(cat, drop_first=(has_const or entity or time))
        x = DataFrame(
            np.c_[x.values, dummies.values.astype(np.float64)],
            index=x.index,
            columns=list(x.columns) + list(dummies.columns),
        )
    w = np.ones_like(y)

    wy = w * y.values
    wx = w * x.values
    params = lstsq(wx, wy, rcond=None)[0]
    params = params.squeeze()

    return params[:nvar]


def generate_data(
    missing: bool,
    datatype: Literal["pandas", "xarray", "numpy"],
    const: bool = False,
    ntk: tuple[int, int, int] = (971, 7, 5),
    other_effects: int = 0,
    rng: RandomState | None = None,
    num_cats: int | list[int] = 4,
):
    if rng is None:
        np.random.seed(12345)
    else:
        np.random.set_state(rng.get_state())
    import linearmodels.typing.data

    n, t, k = ntk
    k += const
    x = standard_normal((k, t, n))
    beta = np.arange(1, k + 1)[:, None, None] / k
    y: linearmodels.typing.data.Float64Array = np.empty((t, n), dtype=np.float64)
    y[:, :] = (x * beta).sum(0) + standard_normal((t, n)) + 2 * standard_normal((1, n))
    w = np.random.chisquare(5, (t, n)) / 5
    c = np.empty((y.size, 0), dtype=int)
    if other_effects == 1:
        cats = ["Industries"]
    else:
        cats = ["cat." + str(i) for i in range(other_effects)]
    if other_effects:
        if isinstance(num_cats, int):
            num_cats = [num_cats] * other_effects
        oe = []
        for i in range(other_effects):
            nc = num_cats[i]
            oe.append(np.random.randint(0, nc, (1, t, n)))
        c = np.concatenate(oe, 0)

    vcats = ["varcat." + str(i) for i in range(2)]
    vc2 = np.ones((2, t, 1)) @ np.random.randint(0, n // 2, (2, 1, n))
    vc1 = vc2[[0]]

    if const:
        x[0] = 1.0

    if missing > 0:
        locs = np.random.choice(n * t, int(n * t * missing))
        y.flat[locs] = float(np.nan)
        locs = np.random.choice(n * t * k, int(n * t * k * missing))
        x.flat[locs] = float(np.nan)
    if rng is not None:
        rng.set_state(np.random.get_state())
    if datatype == "numpy":
        return AttrDict(y=y, x=x, w=w, c=c, vc1=vc1, vc2=vc2)

    entities = ["firm" + str(i) for i in range(n)]
    time = date_range("1-1-1900", periods=t, freq=ANNUAL_FREQ)
    var_names = ["x" + str(i) for i in range(k)]
    # y = DataFrame(y, index=time, columns=entities)
    y_df = panel_to_frame(
        y[None], items=["y"], major_axis=time, minor_axis=entities, swap=True
    )
    w_df = panel_to_frame(
        w[None], items=["w"], major_axis=time, minor_axis=entities, swap=True
    )
    w_df = w_df.reindex(y_df.index)
    x_df = panel_to_frame(
        x, items=var_names, major_axis=time, minor_axis=entities, swap=True
    )
    x_df = x_df.reindex(y_df.index)
    if c.shape[1]:
        c_df = panel_to_frame(
            c, items=cats, major_axis=time, minor_axis=entities, swap=True
        )
    else:
        c_df = DataFrame(index=y_df.index)
    c_df = c_df.reindex(y_df.index)
    vc1_df = panel_to_frame(
        vc1, items=vcats[:1], major_axis=time, minor_axis=entities, swap=True
    )
    vc1_df = vc1_df.reindex(y_df.index)
    vc2_df = panel_to_frame(
        vc2, items=vcats, major_axis=time, minor_axis=entities, swap=True
    )
    vc2_df = vc2_df.reindex(y_df.index)
    if datatype == "pandas":
        return AttrDict(y=y_df, x=x_df, w=w_df, c=c_df, vc1=vc1_df, vc2=vc2_df)

    assert datatype == "xarray"
    import xarray as xr
    from xarray.core.dtypes import NA

    x_xr = xr.DataArray(
        PanelData(x_df).values3d,
        coords={"entities": entities, "time": time, "vars": var_names},
        dims=["vars", "time", "entities"],
    )
    y_xr = xr.DataArray(
        PanelData(y_df).values3d,
        coords={"entities": entities, "time": time, "vars": ["y"]},
        dims=["vars", "time", "entities"],
    )
    w_xr = xr.DataArray(
        PanelData(w_df).values3d,
        coords={"entities": entities, "time": time, "vars": ["w"]},
        dims=["vars", "time", "entities"],
    )
    c_vals = PanelData(c_df).values3d if c.shape[1] else NA
    c_xr = xr.DataArray(
        c_vals,
        coords={"entities": entities, "time": time, "vars": c_df.columns},
        dims=["vars", "time", "entities"],
    )
    vc1_xr = xr.DataArray(
        PanelData(vc1_df).values3d,
        coords={"entities": entities, "time": time, "vars": vc1_df.columns},
        dims=["vars", "time", "entities"],
    )
    vc2_xr = xr.DataArray(
        PanelData(vc2_df).values3d,
        coords={"entities": entities, "time": time, "vars": vc2_df.columns},
        dims=["vars", "time", "entities"],
    )
    return AttrDict(y=y_xr, x=x_xr, w=w_xr, c=c_xr, vc1=vc1_xr, vc2=vc2_xr)


def assert_results_equal(res1, res2, test_fit=True, test_df=True, strict=True):
    n = min(res1.params.shape[0], res2.params.shape[0])

    assert_series_equal(res1.params.iloc[:n], res2.params.iloc[:n])
    assert_series_equal(res1.pvalues.iloc[:n], res2.pvalues.iloc[:n])
    assert_series_equal(res1.tstats.iloc[:n], res2.tstats.iloc[:n])
    assert_frame_equal(res1.cov.iloc[:n, :n], res2.cov.iloc[:n, :n])
    assert_frame_equal(res1.conf_int().iloc[:n], res2.conf_int().iloc[:n])

    assert_allclose(res1.s2, res2.s2)

    rtol = 1e-7 if strict else 1e-4
    delta = 1 + (res1.resids.values - res2.resids.values) / max(
        res1.resids.std(), res2.resids.std()
    )
    assert_allclose(delta, np.ones_like(delta), rtol=rtol)
    delta = 1 + (res1.wresids.values - res2.wresids.values) / max(
        res1.wresids.std(), res2.wresids.std()
    )
    assert_allclose(delta, np.ones_like(delta), rtol=rtol)

    if test_df:
        assert_allclose(res1.df_model, res2.df_model)
        assert_allclose(res1.df_resid, res2.df_resid)

    if test_fit:
        assert_allclose(res1.rsquared, res2.rsquared)
        assert_allclose(res1.total_ss, res2.total_ss)
        assert_allclose(res1.resid_ss, res2.resid_ss)
        assert_allclose(res1.model_ss, res2.model_ss)


def assert_frame_similar(result, expected):
    r = result.copy()
    r.iloc[:, :] = 0.0
    e = expected.copy()
    e.iloc[:, :] = 0.0
    assert_frame_equal(r, e)


def access_attributes(result):
    d = dir(result)
    for key in d:
        if not key.startswith("_") and key not in ("wald_test",):
            val = getattr(result, key)
            if callable(val):
                val()
