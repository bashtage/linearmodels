from itertools import product

import numpy as np
import pytest
from numpy.linalg import pinv
from numpy.testing import assert_allclose, assert_equal
from pandas import DataFrame, date_range, Categorical, get_dummies, Series, datetime
from pandas.testing import assert_frame_equal

from linearmodels.compat.pandas import is_string_dtype

try:
    import xarray as xr
except ImportError:
    pass

from linearmodels.compat.numpy import lstsq
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS
from linearmodels.tests.panel._utility import generate_data, datatypes, \
    MISSING_XARRAY
from linearmodels.utility import panel_to_frame

pytestmark = pytest.mark.filterwarnings('ignore::linearmodels.utility.MissingValueWarning')

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = datatypes


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]),
                             product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype, ntk=(231, 7, 5))


@pytest.fixture
def mi_df():
    np.random.seed(12345)
    n, t, k = 11, 7, 3
    x = np.random.standard_normal((k, t, n))
    major = date_range('12-31-1999', periods=7)
    items = ['var.{0}'.format(i) for i in range(1, k + 1)]
    minor = ['entities.{0}'.format(i) for i in range(1, n + 1)]
    return panel_to_frame(x, items, major, minor, swap=True)


def test_numpy_3d():
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    dh = PanelData(x)
    assert_equal(x, dh.values3d)
    assert dh.nentity == n
    assert dh.nobs == t
    assert dh.nvar == k
    assert_equal(np.reshape(x.T, (n * t, k)), dh.values2d)
    items = ['entity.{0}'.format(i) for i in range(n)]
    obs = [i for i in range(t)]
    var_names = ['x.{0}'.format(i) for i in range(k)]
    expected_frame = panel_to_frame(np.reshape(x, (k, t, n)), items=var_names,
                                    major_axis=obs, minor_axis=items, swap=True)
    expected_frame.index.levels[0].name = 'entity'
    expected_frame.index.levels[1].name = 'time'
    assert_frame_equal(dh.dataframe, expected_frame)


def test_numpy_1d():
    n = 11
    x = np.random.random(n)
    with pytest.raises(ValueError):
        PanelData(x)


def test_numpy_2d():
    n, t, k = 11, 7, 1
    x = np.random.random((t, n))
    dh = PanelData(x)
    assert dh.nentity == n
    assert dh.nobs == t
    assert dh.nvar == k
    assert_equal(np.reshape(x.T, (n * t, k)), dh.values2d)
    assert_equal(np.reshape(x, (k, t, n)), dh.values3d)


def test_pandas_multiindex_dataframe():
    n, t, k = 11, 7, 3
    x = np.random.random((n, t, k))
    major = date_range('12-31-1999', periods=7)
    minor = ['var.{0}'.format(i) for i in range(1, k + 1)]
    items = ['item.{0}'.format(i) for i in range(1, n + 1)]
    x = panel_to_frame(x, items=items, major_axis=major, minor_axis=minor, swap=True)
    PanelData(x)


def test_pandas_dataframe():
    t, n = 11, 7
    x = np.random.random((t, n))
    index = date_range('12-31-1999', periods=t)
    cols = ['entity.{0}'.format(i) for i in range(1, n + 1)]
    x = DataFrame(x, columns=cols, index=index)
    PanelData(x)


def test_existing_panel_data():
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    major = date_range('12-31-1999', periods=7)
    items = ['var.{0}'.format(i) for i in range(1, k + 1)]
    minor = ['entities.{0}'.format(i) for i in range(1, n + 1)]
    x = panel_to_frame(x, items=items, major_axis=major, minor_axis=minor, swap=True)
    dh = PanelData(x)
    dh2 = PanelData(dh)
    assert_frame_equal(dh.dataframe, dh2.dataframe)


@pytest.mark.skipif(MISSING_XARRAY, reason='xarray is not installed')
def test_xarray_2d():
    n, t = 11, 7
    x = np.random.random((t, n))
    x = xr.DataArray(x, dims=('time', 'entity'),
                     coords={
                         'entity': list('firm.' + str(i) for i in range(n))})
    dh = PanelData(x)
    assert_equal(dh.values2d, np.reshape(x.values.T, (n * t, 1)))


@pytest.mark.skipif(MISSING_XARRAY, reason='xarray is not installed')
def test_xarray_3d():
    n, t, k = 11, 7, 13
    x = np.random.random((k, t, n))
    x = xr.DataArray(x, dims=('var', 'time', 'entity'),
                     coords={
                         'entity': list('firm.' + str(i) for i in range(n)),
                         'var': list('x.' + str(i) for i in range(k))})
    dh = PanelData(x)
    assert_equal(np.reshape(x.values.T, (n * t, k)), dh.values2d)


def test_dimensions(mi_df):
    dh = PanelData(mi_df)
    assert dh.nentity == len(mi_df.index.levels[0])
    assert dh.nvar == mi_df.shape[1]
    assert dh.nobs == len(mi_df.index.levels[1])


def test_drop(mi_df):
    dh = PanelData(mi_df)
    orig = dh.dataframe.copy()
    sel = np.zeros(orig.shape[0], dtype=np.bool)
    sel[::3] = True
    dh.drop(sel)
    assert dh.dataframe.shape[0] == len(sel) - sel.sum()


def test_labels(mi_df):
    dh = PanelData(mi_df)
    assert dh.vars == list(mi_df.columns)
    assert dh.time == list(mi_df.index.levels[1])
    assert dh.entities == list(mi_df.index.get_level_values(0).unique())


def test_missing(mi_df):
    mi_df = mi_df.copy()
    entities = mi_df.index.levels[0]
    for entity in entities[::3]:
        mi_df.loc[entity] = np.nan
    dh = PanelData(mi_df)
    assert_equal(dh.isnull, np.any(np.isnan(dh.values2d), 1))


def test_incorrect_dataframe():
    grouped = np.array(list([i] * 10 for i in range(10))).ravel()
    df = DataFrame({'a': np.arange(100),
                    'b': grouped,
                    'c': np.random.permutation(grouped),
                    'data': np.random.randn(100)})
    df = df.set_index(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        PanelData(df)


def test_incorrect_types():
    with pytest.raises(TypeError):
        PanelData(list(np.random.randn(10)))


@pytest.mark.skipif(MISSING_XARRAY, reason='xarray is not installed')
def test_incorrect_types_xarray():
    with pytest.raises(ValueError):
        PanelData(xr.DataArray(np.random.randn(10)))


def test_ids(mi_df):
    data = PanelData(mi_df)
    eids = data.entity_ids
    assert eids.shape == (77, 1)
    assert len(np.unique(eids)) == 11
    for i in range(0, len(eids), 7):
        assert np.ptp(eids[i:i + 7]) == 0
        assert np.all((eids[i + 8:] - eids[i]) != 0)

    tids = data.time_ids
    assert tids.shape == (77, 1)
    assert len(np.unique(tids)) == 7
    for i in range(0, 11):
        assert np.ptp(tids[i::7]) == 0


def test_str_repr(mi_df):
    data = PanelData(mi_df)
    assert 'PanelData' in str(data)
    assert str(hex(id(data))) in data.__repr__()


def test_demean(mi_df):
    data = PanelData(mi_df)
    fe = data.demean('entity')
    expected = data.values3d.copy()
    for i in range(3):
        expected[i] -= expected[i].mean(0)
    assert_allclose(fe.values3d, expected)

    te = data.demean('time')
    expected = data.values3d.copy()
    for i in range(3):
        expected[i] -= expected[i].mean(1)[:, None]
    assert_allclose(te.values3d, expected)


def test_demean_against_groupby(data):
    dh = PanelData(data.x)
    df = dh.dataframe

    def demean(x):
        return x - x.mean()

    entity_demean = df.groupby(level=0).transform(demean)
    res = dh.demean('entity')
    assert_allclose(entity_demean.values, res.values2d)

    time_demean = df.groupby(level=1).transform(demean)
    res = dh.demean('time')
    assert_allclose(time_demean.values, res.values2d)


def test_demean_against_dummy_regression(data):
    dh = PanelData(data.x)
    dh.drop(dh.isnull)

    df = dh.dataframe
    no_index = df.reset_index()

    cat = Categorical(no_index[df.index.levels[0].name])
    d = get_dummies(cat, drop_first=False).astype(np.float64)
    dummy_demeaned = df.values - d @ lstsq(d, df.values)[0]
    entity_demean = dh.demean('entity')
    assert_allclose(1 + np.abs(entity_demean.values2d),
                    1 + np.abs(dummy_demeaned))

    cat = Categorical(no_index[df.index.levels[1].name])
    d = get_dummies(cat, drop_first=False).astype(np.float64)
    dummy_demeaned = df.values - d @ lstsq(d, df.values)[0]
    time_demean = dh.demean('time')
    assert_allclose(1 + np.abs(time_demean.values2d),
                    1 + np.abs(dummy_demeaned))

    cat = Categorical(no_index[df.index.levels[0].name])
    d1 = get_dummies(cat, drop_first=False).astype(np.float64)
    cat = Categorical(no_index[df.index.levels[1].name])
    d2 = get_dummies(cat, drop_first=True).astype(np.float64)
    d = np.c_[d1.values, d2.values]
    dummy_demeaned = df.values - d @ lstsq(d, df.values)[0]
    both_demean = dh.demean('both')
    assert_allclose(1 + np.abs(both_demean.values2d),
                    1 + np.abs(dummy_demeaned))


def test_demean_missing(mi_df):
    mi_df.values.flat[::13] = np.nan
    data = PanelData(mi_df)
    fe = data.demean('entity')
    expected = data.values3d.copy()
    for i in range(3):
        expected[i] -= np.nanmean(expected[i], 0)
    assert_allclose(fe.values3d, expected)

    te = data.demean('time')
    expected = data.values3d.copy()
    for i in range(3):
        expected[i] -= np.nanmean(expected[i], 1)[:, None]
    assert_allclose(te.values3d, expected)


def test_demean_many_missing(mi_df):
    entities = mi_df.index.levels[0]
    times = mi_df.index.levels[1]
    skips = (3, 5, 2)
    for column, skip in zip(mi_df, skips):
        for entity in entities[::skip]:
            mi_df.loc[entity, column] = np.nan
        mi_df.index = mi_df.index.swaplevel()
        for time in times[::skip]:
            mi_df.loc[time, column] = np.nan
        mi_df.index = mi_df.index.swaplevel()
    data = PanelData(mi_df)
    fe = data.demean('entity')
    orig_nan = np.isnan(data.values3d.ravel())
    fe_nan = np.isnan(fe.values3d.ravel())
    assert np.all(fe_nan[orig_nan])
    expected = data.values3d.copy()
    for i in range(3):
        mu = np.ones(expected[i].shape[1]) * np.nan
        for j in range(expected[i].shape[1]):
            if np.any(np.isfinite(expected[i][:, j])):
                mu[j] = np.nanmean(expected[i][:, j])
        expected[i] -= mu
    assert_allclose(fe.values3d, expected)

    te = data.demean('time')
    expected = data.values3d.copy()
    for i in range(3):
        mu = np.ones((expected[i].shape[0], 1)) * np.nan
        for j in range(expected[i].shape[0]):
            if np.any(np.isfinite(expected[i][j])):
                mu[j, 0] = np.nanmean(expected[i][j])
        expected[i] -= mu
    assert_allclose(te.values3d, expected)


def test_demean_many_missing_dropped(mi_df):
    entities = mi_df.index.levels[0]
    times = mi_df.index.levels[1]
    column = mi_df.columns[0]
    for entity in entities[::3]:
        mi_df.loc[entity, column] = np.nan
    mi_df.index = mi_df.index.swaplevel()
    for time in times[::3]:
        mi_df.loc[time, column] = np.nan
    mi_df.index = mi_df.index.swaplevel()

    data = PanelData(mi_df)
    data.drop(data.isnull)
    fe = data.demean('entity')

    expected = data.values2d.copy()
    eid = data.entity_ids.ravel()
    for i in np.unique(eid):
        expected[eid == i] -= np.nanmean(expected[eid == i], 0)

    assert_allclose(fe.values2d, expected)


def test_demean_both_large_t():
    x = np.random.standard_normal((1, 100, 10))
    time = date_range('1-1-2000', periods=100)
    entities = ['entity.{0}'.format(i) for i in range(10)]
    data = panel_to_frame(x, ['x'], time, entities, swap=True)
    data = PanelData(data)
    demeaned = data.demean('both')

    df = data.dataframe
    no_index = df.reset_index()
    cat = Categorical(no_index[df.index.levels[0].name])
    d1 = get_dummies(cat, drop_first=False).astype(np.float64)
    cat = Categorical(no_index[df.index.levels[1].name])
    d2 = get_dummies(cat, drop_first=True).astype(np.float64)
    d = np.c_[d1.values, d2.values]
    dummy_demeaned = df.values - d @ pinv(d) @ df.values
    assert_allclose(1 + np.abs(demeaned.values2d),
                    1 + np.abs(dummy_demeaned))


def test_demean_invalid(mi_df):
    data = PanelData(mi_df)
    with pytest.raises(ValueError):
        data.demean('unknown')


def test_dummies(mi_df):
    data = PanelData(mi_df)
    edummy = data.dummies()
    assert edummy.shape == (77, 11)
    assert np.all(edummy.sum(0) == 7)
    tdummy = data.dummies(group='time')
    assert tdummy.shape == (77, 7)
    assert np.all(tdummy.sum(0) == 11)
    tdummy_drop = data.dummies(group='time', drop_first=True)
    assert tdummy_drop.shape == (77, 6)
    assert np.all(tdummy.sum(0) == 11)
    with pytest.raises(ValueError):
        data.dummies('unknown')


def test_roundtrip_3d(data):
    x = data.x
    xpd = PanelData(x)
    xv = x if isinstance(x, np.ndarray) else x.values
    if isinstance(x, DataFrame):
        assert_equal(xpd.values2d, xv)
    else:
        assert_equal(xpd.values3d, xv)


def test_series_multiindex(mi_df):
    from_df = PanelData(mi_df.iloc[:, [0]])
    from_series = PanelData(mi_df.iloc[:, 0])
    assert_frame_equal(from_df.dataframe, from_series.dataframe)


def test_invalid_seires(mi_df):
    si = mi_df.reset_index()
    with pytest.raises(ValueError):
        PanelData(si.iloc[:, 0])


def test_demean_missing_alt_types(data):
    check = isinstance(data.x, (DataFrame, np.ndarray))
    xpd = PanelData(data.x)
    xpd.drop(xpd.isnull)
    entity_demean = xpd.demean('entity')
    expected = xpd.dataframe.groupby(level=0).transform(lambda s: s - s.mean())
    assert_frame_equal(entity_demean.dataframe, expected,
                       check_index_type=check,
                       check_column_type=check)

    time_demean = xpd.demean('time')
    expected = xpd.dataframe.groupby(level=1).transform(lambda s: s - s.mean())
    assert_frame_equal(time_demean.dataframe, expected,
                       check_index_type=check,
                       check_column_type=check)


def test_mean_missing(data):
    xpd = PanelData(data.x)
    xpd.drop(xpd.isnull)
    entity_mean = xpd.mean('entity')
    expected = xpd.dataframe.groupby(level=0).mean()
    expected = expected.loc[xpd.entities]
    expected.columns.name = None
    assert_frame_equal(entity_mean, expected)

    time_mean = xpd.mean('time')
    expected = xpd.dataframe.groupby(level=1).mean()
    expected = expected.loc[xpd.time]
    expected.columns.name = None
    assert_frame_equal(time_mean, expected)


def test_count(data):
    xpd = PanelData(data.x)
    xpd.drop(xpd.isnull)
    entity_mean = xpd.count('entity')
    expected = xpd.dataframe.groupby(level=0).count()
    expected = expected.loc[xpd.entities]
    expected.columns.name = None
    expected = expected.astype(np.int64)
    assert_frame_equal(entity_mean, expected)

    time_mean = xpd.count('time')
    expected = xpd.dataframe.groupby(level=1).count()
    expected = expected.loc[xpd.time]
    expected.columns.name = None
    expected = expected.astype(np.int64)
    assert_frame_equal(time_mean, expected)


def test_first_difference(data):
    x = PanelData(data.x)
    x.first_difference()


def test_demean_simple_weighted(data):
    x = PanelData(data.x)
    w = PanelData(data.w)
    missing = x.isnull | w.isnull
    x.drop(missing)
    w.drop(missing)
    w.dataframe.iloc[:, 0] = 1
    unweighted_entity_demean = x.demean('entity')
    weighted_entity_demean = x.demean('entity', weights=w)
    assert_allclose(unweighted_entity_demean.dataframe,
                    weighted_entity_demean.dataframe)

    unweighted_entity_demean = x.demean('time')
    weighted_entity_demean = x.demean('time', weights=w)
    assert_allclose(unweighted_entity_demean.dataframe,
                    weighted_entity_demean.dataframe)


def test_demean_weighted(data):
    x = PanelData(data.x)
    w = PanelData(data.w)
    missing = x.isnull | w.isnull
    x.drop(missing)
    w.drop(missing)

    entity_demean = x.demean('entity', weights=w)
    d = get_dummies(Categorical(x.index.labels[0]))
    d = d.values
    root_w = np.sqrt(w.values2d)
    wx = root_w * x.values2d
    wd = d * root_w
    mu = wd @ lstsq(wd, wx)[0]
    e = wx - mu
    assert_allclose(1 + np.abs(entity_demean.values2d),
                    1 + np.abs(e))

    time_demean = x.demean('time', weights=w)
    d = get_dummies(Categorical(x.index.labels[1]))
    d = d.values
    root_w = np.sqrt(w.values2d)
    wx = root_w * x.values2d
    wd = d * root_w
    mu = wd @ lstsq(wd, wx)[0]
    e = wx - mu
    assert_allclose(1 + np.abs(time_demean.values2d),
                    1 + np.abs(e))


def test_mean_weighted(data):
    x = PanelData(data.x)
    w = PanelData(data.w)
    missing = x.isnull | w.isnull
    x.drop(missing)
    w.drop(missing)
    entity_mean = x.mean('entity', weights=w)
    c = x.index.levels[0][x.index.labels[0]]
    d = get_dummies(Categorical(c, ordered=True))
    d = d[entity_mean.index]
    d = d.values
    root_w = np.sqrt(w.values2d)
    wx = root_w * x.values2d
    wd = d * root_w
    mu = lstsq(wd, wx)[0]
    assert_allclose(entity_mean, mu)

    time_mean = x.mean('time', weights=w)
    c = x.index.levels[1][x.index.labels[1]]
    d = get_dummies(Categorical(c, ordered=True))
    d = d[time_mean.index]
    d = d.values
    root_w = np.sqrt(w.values2d)
    wx = root_w * x.values2d
    wd = d * root_w
    mu = pinv(wd) @ wx
    assert_allclose(time_mean, mu)


def test_categorical_conversion():
    t, n = 3, 1000
    string = np.random.choice(['a', 'b', 'c'], (t, n))
    num = np.random.randn(t, n)
    time = date_range('1-1-2000', periods=t)
    entities = ['entity.{0}'.format(i) for i in range(n)]
    p = panel_to_frame(None, items=['a', 'b'], major_axis=time,
                       minor_axis=entities, swap=True)
    p['a'] = string.T.ravel()
    p['b'] = num.T.ravel()
    p = p[['a', 'b']]
    panel = PanelData(p, convert_dummies=False)
    df = panel.dataframe.copy()
    df['a'] = Categorical(df['a'])
    panel = PanelData(df, convert_dummies=True)

    df = panel.dataframe
    assert df.shape == (3000, 3)
    s = string.T.ravel()
    a_locs = np.where(s == 'a')
    b_locs = np.where(s == 'b')
    c_locs = np.where(s == 'c')
    assert np.all(df.loc[:, 'a.b'].values[a_locs] == 0.0)
    assert np.all(df.loc[:, 'a.b'].values[b_locs] == 1.0)
    assert np.all(df.loc[:, 'a.b'].values[c_locs] == 0.0)

    assert np.all(df.loc[:, 'a.c'].values[a_locs] == 0.0)
    assert np.all(df.loc[:, 'a.c'].values[b_locs] == 0.0)
    assert np.all(df.loc[:, 'a.c'].values[c_locs] == 1.0)


def test_string_conversion():
    t, n = 3, 1000
    string = np.random.choice(['a', 'b', 'c'], (t, n))
    num = np.random.randn(t, n)
    time = date_range('1-1-2000', periods=t)
    entities = ['entity.{0}'.format(i) for i in range(n)]
    p = panel_to_frame(None, items=['a', 'b'], major_axis=time, minor_axis=entities,
                       swap=True)
    p['a'] = string.T.ravel()
    p['b'] = num.T.ravel()
    p = p[['a', 'b']]
    panel = PanelData(p, var_name='OtherEffect')
    df = panel.dataframe
    assert df.shape == (3000, 3)
    s = string.T.ravel()
    a_locs = np.where(s == 'a')
    b_locs = np.where(s == 'b')
    c_locs = np.where(s == 'c')
    assert np.all(df.loc[:, 'a.b'].values[a_locs] == 0.0)
    assert np.all(df.loc[:, 'a.b'].values[b_locs] == 1.0)
    assert np.all(df.loc[:, 'a.b'].values[c_locs] == 0.0)

    assert np.all(df.loc[:, 'a.c'].values[a_locs] == 0.0)
    assert np.all(df.loc[:, 'a.c'].values[b_locs] == 0.0)
    assert np.all(df.loc[:, 'a.c'].values[c_locs] == 1.0)


def test_string_nonconversion():
    t, n = 3, 1000
    string = np.random.choice(['a', 'b', 'c'], (t, n))
    num = np.random.randn(t, n)
    time = date_range('1-1-2000', periods=t)
    entities = ['entity.{0}'.format(i) for i in range(n)]
    p = panel_to_frame(None, items=['a', 'b'], major_axis=time, minor_axis=entities,
                       swap=True)
    p['a'] = string.T.ravel()
    p['b'] = num.T.ravel()
    panel = PanelData(p, var_name='OtherEffect', convert_dummies=False)
    assert is_string_dtype(panel.dataframe['a'].dtype)
    assert np.all(panel.dataframe['a'] == string.T.ravel())


def test_repr_html(mi_df):
    data = PanelData(mi_df)
    html = data._repr_html_()
    assert '<br/>' in html


def test_general_demean_oneway(mi_df):
    y = PanelData(mi_df)
    dm1 = y.demean('entity')
    g = DataFrame(y.entity_ids, index=y.index)
    dm2 = y.general_demean(g)
    assert_allclose(dm1.values2d, dm2.values2d)

    dm1 = y.demean('time')
    g = DataFrame(y.time_ids, index=y.index)
    dm2 = y.general_demean(g)
    assert_allclose(dm1.values2d, dm2.values2d)

    g = DataFrame(np.random.randint(0, 10, g.shape), index=y.index)
    dm2 = y.general_demean(g)
    g = Categorical(g.iloc[:, 0])
    d = get_dummies(g)
    dm1 = y.values2d - d @ lstsq(d, y.values2d)[0]
    assert_allclose(dm1, dm2.values2d)


def test_general_demean_twoway(mi_df):
    y = PanelData(mi_df)
    dm1 = y.demean('both')
    g = DataFrame(y.entity_ids, index=y.index)
    g['column2'] = Series(y.time_ids.squeeze(), index=y.index)
    dm2 = y.general_demean(g)
    assert_allclose(dm1.values2d, dm2.values2d)

    g = DataFrame(np.random.randint(0, 10, g.shape), index=y.index)
    dm2 = y.general_demean(g)
    g1 = Categorical(g.iloc[:, 0])
    d1 = get_dummies(g1)
    g2 = Categorical(g.iloc[:, 1])
    d2 = get_dummies(g2, drop_first=True)
    d = np.c_[d1, d2]
    dm1 = y.values2d - d @ lstsq(d, y.values2d)[0]
    assert_allclose(dm1 - dm2.values2d, np.zeros_like(dm2.values2d), atol=1e-7)


def test_general_unit_weighted_demean_oneway(mi_df):
    y = PanelData(mi_df)
    dm1 = y.demean('entity')
    g = PanelData(DataFrame(y.entity_ids, index=y.index))
    weights = PanelData(g).copy()
    weights.dataframe.iloc[:, :] = 1
    dm2 = y.general_demean(g, weights)
    assert_allclose(dm1.values2d, dm2.values2d)
    dm3 = y.general_demean(g)
    assert_allclose(dm3.values2d, dm2.values2d)

    dm1 = y.demean('time')
    g = PanelData(DataFrame(y.time_ids, index=y.index))
    dm2 = y.general_demean(g, weights)
    assert_allclose(dm1.values2d, dm2.values2d)
    dm3 = y.general_demean(g)
    assert_allclose(dm3.values2d, dm2.values2d)

    g = PanelData(DataFrame(np.random.randint(0, 10, g.dataframe.shape),
                            index=y.index))
    dm2 = y.general_demean(g, weights)
    dm3 = y.general_demean(g)
    g = Categorical(g.dataframe.iloc[:, 0])
    d = get_dummies(g)
    dm1 = y.values2d - d @ lstsq(d, y.values2d)[0]
    assert_allclose(dm1, dm2.values2d)
    assert_allclose(dm3.values2d, dm2.values2d)


def test_general_weighted_demean_oneway(mi_df):
    y = PanelData(mi_df)
    weights = DataFrame(
        np.random.chisquare(10, (y.dataframe.shape[0], 1)) / 10, index=y.index)
    w = PanelData(weights)

    dm1 = y.demean('entity', weights=w)
    g = PanelData(DataFrame(y.entity_ids, index=y.index))
    dm2 = y.general_demean(g, w)
    assert_allclose(dm1.values2d, dm2.values2d)

    dm1 = y.demean('time', weights=w)
    g = PanelData(DataFrame(y.time_ids, index=y.index))
    dm2 = y.general_demean(g, w)
    assert_allclose(dm1.values2d, dm2.values2d)

    g = PanelData(DataFrame(np.random.randint(0, 10, g.dataframe.shape),
                            index=y.index))
    dm2 = y.general_demean(g, w)
    g = Categorical(g.dataframe.iloc[:, 0])
    d = get_dummies(g)
    wd = np.sqrt(w.values2d) * d
    wy = np.sqrt(w.values2d) * y.values2d
    dm1 = wy - wd @ lstsq(wd, wy)[0]
    assert_allclose(dm1, dm2.values2d, atol=1e-14)


def test_general_unit_weighted_demean_twoway(mi_df):
    np.random.seed(12345)
    y = PanelData(mi_df)
    weights = DataFrame(
        np.random.chisquare(10, (y.dataframe.shape[0], 1)) / 10, index=y.index)
    w = PanelData(weights)

    dm1 = y.demean('both', weights=w)
    g = DataFrame(y.entity_ids, index=y.index)
    g['column2'] = Series(y.time_ids.squeeze(), index=y.index)
    dm2 = y.general_demean(g, weights=w)
    assert_allclose(dm1.values2d - dm2.values2d, np.zeros_like(dm2.values2d),
                    atol=1e-7)

    g = DataFrame(np.random.randint(0, 10, g.shape), index=y.index)
    dm2 = y.general_demean(g, weights=w)
    g1 = Categorical(g.iloc[:, 0])
    d1 = get_dummies(g1)
    g2 = Categorical(g.iloc[:, 1])
    d2 = get_dummies(g2, drop_first=True)
    d = np.c_[d1, d2]
    wd = np.sqrt(w.values2d) * d
    wy = np.sqrt(w.values2d) * y.values2d
    dm1 = wy - wd @ lstsq(wd, wy)[0]
    assert_allclose(dm1 - dm2.values2d, np.zeros_like(dm2.values2d), atol=1e-7)


def test_original_unmodified(data):
    pre_y = data.y.copy()
    pre_x = data.x.copy()
    pre_w = data.w.copy()
    mod = PanelOLS(data.y, data.x, weights=data.w)
    mod.fit(debiased=True)
    if isinstance(data.y, (DataFrame)):
        for after, before in ((data.y, pre_y), (data.x, pre_x), (data.w, pre_w)):
            assert_frame_equal(before, after)

        mi_df_y = PanelData(data.y).dataframe
        mi_df_x = PanelData(data.x).dataframe
        mi_df_y.index.names = ['firm', 'period']
        mi_df_x.index.names = ['firm', 'period']
        mi_df_w = PanelData(data.w).dataframe
        pre_y = mi_df_y.copy()
        pre_x = mi_df_x.copy()
        pre_w = mi_df_w.copy()
        mod = PanelOLS(mi_df_y, mi_df_x, weights=mi_df_w)
        mod.fit(debiased=True)
        assert_frame_equal(mi_df_w, pre_w)
        assert_frame_equal(mi_df_y, pre_y)
        assert_frame_equal(mi_df_x, pre_x)
    elif isinstance(data.y, np.ndarray):
        assert_allclose(data.y, pre_y)
        assert_allclose(data.x, pre_x)
        assert_allclose(data.w, pre_w)
    else:
        xr.testing.assert_identical(data.y, pre_y)
        xr.testing.assert_identical(data.w, pre_w)
        xr.testing.assert_identical(data.x, pre_x)


def test_incorrect_time_axis():
    x = np.random.randn(3, 3, 1000)
    entities = ['entity.{0}'.format(i) for i in range(1000)]
    time = ['time.{0}'.format(i) for i in range(3)]
    var_names = ['var.{0}'.format(i) for i in range(3)]
    p = panel_to_frame(x, items=var_names, major_axis=time, minor_axis=entities,
                       swap=True)
    with pytest.raises(ValueError):
        PanelData(p)

    time = [1, 2, 3]
    var_names = ['var.{0}'.format(i) for i in range(3)]
    p = panel_to_frame(x, items=var_names, major_axis=time, minor_axis=entities,
                       swap=True)
    p.index = p.index.set_levels([1, datetime(1960, 1, 1), 'a'], 1)
    with pytest.raises(ValueError):
        PanelData(p)


@pytest.mark.skipif(MISSING_XARRAY, reason='xarray is not installed')
def test_incorrect_time_axis_xarray():
    x = np.random.randn(3, 3, 1000)
    entities = ['entity.{0}'.format(i) for i in range(1000)]
    time = ['time.{0}'.format(i) for i in range(3)]
    vars = ['x.{0}'.format(i) for i in range(3)]
    da = xr.DataArray(x, coords={'entities': entities, 'time': time,
                                 'vars': vars},
                      dims=['vars', 'time', 'entities'])
    with pytest.raises(ValueError):
        PanelData(da)

    da = xr.DataArray(x, coords={'entities': entities, 'time': time,
                                 'vars': vars},
                      dims=['vars', 'time', 'entities'])
    with pytest.raises(ValueError):
        PanelData(da)


def test_named_index(data):
    pdata = PanelData(data.x)
    if isinstance(data.x, DataFrame):
        assert pdata.dataframe.index.levels[0].name == data.x.index.levels[0].name
        assert pdata.dataframe.index.levels[1].name == data.x.index.levels[1].name

        data.x.index.levels[0].name = None
        data.x.index.levels[1].name = None
        pdata = PanelData(data.x)

    assert pdata.dataframe.index.levels[0].name == 'entity'
    assert pdata.dataframe.index.levels[1].name == 'time'
