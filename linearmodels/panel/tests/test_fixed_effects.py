import string
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_equal, assert_almost_equal

from linearmodels.panel.fixed_effects import EntityEffect, TimeEffect, GroupEffect, \
    SaturatedEffectWarning, DummyVariableIterator, FixedEffect, RequiredSubclassingError


def check_groups_pd(groups, data, cols):
    col1 = col2 = None
    if len(cols) >= 1:
        col1 = cols[0]
    if len(cols) >= 2:
        col2 = cols[1]

    df = data.swapaxes(0, 2).to_frame()
    count = np.sum((groups == np.arange(groups.max() + 1)[:, None]))

    assert_equal(groups.max() + 1, len(np.unique(groups)))
    assert_equal(count, groups.shape[0])
    for i in range(np.max(groups)):
        locs = groups == i
        if col1 is not None:
            if col2 is not None:
                temp = df.iloc[locs][[col1, col2]]
            else:
                temp = df.iloc[locs][[col1]]
            if temp.shape[0] > 1:
                assert (temp[col1] == temp[col1].iloc[0]).all()
            if col2 is not None:
                assert (temp[col2] == temp[col2].iloc[0]).all()


def demean_1col(x, col):
    _x = x.copy()
    n, t, k = _x.shape
    _x = _x.reshape(n * t, k)
    orig_values = _x[:, col].copy()
    uniques = np.unique(_x[:, col])
    for u in uniques:
        locs = _x[:, col] == u
        temp = _x[locs]
        _x[locs] = temp - np.nanmean(temp, 0)
    _x[:, col] = orig_values

    return _x.reshape((n, t, k))


class BaseTestClass(object):
    @classmethod
    def setup_class(cls):
        cls.np0 = np.random.randn(2, 10, 3).reshape(2, 10, 3)
        cls.np0[0, :, 0] = 0
        cls.np0[1, :, 0] = 1
        cls.np0[0, :5, 1] = 0
        cls.np0[1, :5, 1] = 0
        cls.np0[0, 5:, 1] = 1
        cls.np0[1, 5:, 1] = 1

        exp_grp1_np0 = cls.np0.copy()
        exp_grp1_np0[0, :, 1:] -= exp_grp1_np0[0, :, 1:].mean(0)
        exp_grp1_np0[1, :, 1:] -= exp_grp1_np0[1, :, 1:].mean(0)

        cls.expected_group2_np0 = cls.np0.copy()
        cls.expected_group2_np0[0, :5, 1:] -= exp_grp1_np0[0, :5, 1:].mean(0)
        cls.expected_group2_np0[0, 5:, 1:] -= exp_grp1_np0[0, 5:, 1:].mean(0)

        cls.expected_group2_np0[1, :5, 1:] -= exp_grp1_np0[1, :5, 1:].mean(0)
        cls.expected_group2_np0[1, 5:, 1:] -= exp_grp1_np0[1, 5:, 1:].mean(0)

        cls.expected_group1_np0 = exp_grp1_np0

        cls.np1 = np.arange(3 * 100 * 17.0).reshape((100, 3, 17))
        cls.np2 = np.arange(10000 * 2 * 10.0).reshape((10000, 2, 10))
        modulus = [19, 11, 13, 23]
        for i in range(4):
            cls.np1[:, :, i] = cls.np1[:, :, i] % modulus[i]
            cls.np2[:, :, i] = cls.np2[:, :, i] % modulus[i]
        cls.np1 += np.random.random_sample(cls.np1.shape)
        cls.np2 += np.random.random_sample(cls.np2.shape)

        np3 = np.zeros((1000, 17, 4))
        np3 = np3.reshape((17000, 4))
        np3[:, 0] = np.arange(17000) % 13
        for i in range(13):
            np3[np3[:, 0] == i, 1:] = np.random.randn(3)
        np3 = np3.reshape((1000, 17, 4))
        cls.np3 = np3
        cls.np3 += np.random.random_sample(cls.np3.shape)

        cls.np1_missing = cls.np1.copy()
        cls.np1_missing[:, ::3, :] = np.nan

        cls.np1_missing_warning = cls.np1.copy()
        cls.np1_missing_warning[:, 0::2, :] = np.nan

        cls.pd1 = pd.Panel(cls.np1.copy())
        cls.pd2 = pd.Panel(cls.np2.copy())
        cls.pd3 = pd.Panel(cls.np3.copy())

        cls.xr1 = xr.DataArray(cls.np1.copy(),
                               coords=[np.arange(cls.np1.shape[0]),
                                       np.arange(cls.np1.shape[1]),
                                       np.arange(cls.np1.shape[2])])
        cls.xr2 = xr.DataArray(cls.np2.copy(),
                               coords=[np.arange(cls.np2.shape[0]),
                                       np.arange(cls.np2.shape[1]),
                                       np.arange(cls.np2.shape[2])])


class TestEntityEffect(BaseTestClass):
    def test_invalid_data(self):
        with pytest.raises(ValueError):
            EntityEffect(self.np1[0])
        with pytest.raises(ValueError):
            EntityEffect(self.pd1.iloc[0])
        with pytest.raises(ValueError):
            EntityEffect(self.xr1[0])

    def test_numpy(self):
        ee = EntityEffect(self.np1)
        out = ee.orthogonalize()
        expected = self.np1.copy()
        for i in range(expected.shape[0]):
            expected[i] -= expected[i].mean(0)
        assert_almost_equal(expected, out)

        ee = EntityEffect(self.np2)
        out = ee.orthogonalize()
        expected = self.np2.copy()
        for i in range(expected.shape[0]):
            expected[i] -= expected[i].mean(0)
        assert_almost_equal(expected, out)

    def test_numpy_exclude(self):
        ee = EntityEffect(self.np1)
        out = ee.orthogonalize(np.arange(10, self.np1.shape[2]))
        for c in range(10):
            assert (self.np1[:, :, c] != out[:, :, c]).any()

        for c in range(10, self.np1.shape[2]):
            assert_equal(self.np1[:, :, c], out[:, :, c])

        indices = [2, 5, 9]
        ee = EntityEffect(self.np2)
        out = ee.orthogonalize(indices)
        for c in list(range(self.np2.shape[2])):
            if c in indices:
                continue
            assert (self.np2[:, :, c] != out[:, :, c]).any()
        for c in indices:
            assert_equal(self.np2[:, :, c], out[:, :, c])

    def test_numpy_missing(self):
        ee = EntityEffect(self.np1_missing)

        out = ee.orthogonalize()
        expected = self.np1_missing.copy()
        for i in range(expected.shape[0]):
            expected[i] -= np.nanmean(expected[i], 0)
        assert_almost_equal(expected, out)

    def test_numpy_missing_warning(self):
        ee = EntityEffect(self.np1_missing_warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ee.orthogonalize()
            assert len(w) >= 1
            assert issubclass(w[0].category, SaturatedEffectWarning)

    def test_pandas(self):
        ee = EntityEffect(self.pd1)
        out = ee.orthogonalize()
        expected_values = EntityEffect(self.np1).orthogonalize()
        assert_almost_equal(out.values, expected_values)

        ee = EntityEffect(self.pd2)
        out = ee.orthogonalize()
        expected_values = EntityEffect(self.np2).orthogonalize()
        assert_almost_equal(out.values, expected_values)

    def test_pandas_mixed(self):
        panel = {}
        letters = list(string.ascii_lowercase * 1000)
        shape = len(letters)
        for i in range(10):
            df = pd.DataFrame({'var0': pd.Categorical(letters),
                               'var1': np.random.randn(shape),
                               'var2': np.random.randn(shape)})
            df['var0'] = pd.Categorical(df['var0'])
            panel['panel_' + str(i)] = df
        panel = pd.Panel(panel)

        ee = EntityEffect(panel)
        ee.orthogonalize()

    def test_xarray(self):
        ee = EntityEffect(self.xr1)
        out = ee.orthogonalize()
        expected_values = EntityEffect(self.np1).orthogonalize()
        assert_almost_equal(out.values, expected_values)

        ee = EntityEffect(self.xr2)
        out = ee.orthogonalize()
        expected_values = EntityEffect(self.np2).orthogonalize()
        assert_almost_equal(out.values, expected_values)

    def test_dummies(self):
        ee = EntityEffect(self.np1)
        dummies = ee.dummies()
        n, t, k = self.np1.shape
        assert_equal(dummies.shape, (n * t, n))

    def test_dummies_drop(self):
        ee = EntityEffect(self.np1)
        dummies = ee.dummies(drop=True)
        n, t, k = self.np1.shape
        assert_equal(dummies.shape, (n * t, n - 1))

    def test_dummy_iterator(self):
        ee = EntityEffect(self.np1)
        dummies = ee.dummies(iterator=True)
        assert isinstance(dummies, DummyVariableIterator)

    def test_estimate_numpy(self):
        ee = EntityEffect(self.np1)
        ee.estimate(5)
        ee.estimate([5], drop=True)
        ee.estimate(5, exog=[6, 7, 8, 9])
        ee.estimate([5], exog=[6, 7, 8, 9], drop=True)

    def test_estimate_pandas(self):
        endog = self.pd1.iloc[:, :, [5]]
        ee = TimeEffect(endog)
        exog = self.pd1.iloc[:, :, 6:]
        cols = self.pd1.minor_axis
        ee.estimate(cols[5])
        ee.estimate([cols[5]], drop=True)
        ee.estimate(cols[5], exog=list(cols[6:10]))
        ee.estimate([cols[5]], exog=list(cols[6:10]), drop=True)

    def test_estimate_xarray(self):
        endog = self.xr1[:, :, [5]]
        ee = TimeEffect(endog)
        exog = self.xr1[:, :, 6:]
        ee.estimate(5)
        ee.estimate([5], drop=True)
        ee.estimate(5, exog=[6, 7, 8, 9])
        ee.estimate([5], exog=[6, 7, 8, 9], drop=True)

    def test_estimate_numpy_group(self):
        ge = GroupEffect(self.np1, [0])
        ge.estimate(5)
        ge.estimate([5], drop=True)
        ge.estimate(5, exog=[6, 7, 8, 9])
        ge.estimate([5], exog=[6, 7, 8, 9], drop=True)


class TestTimeEffect(BaseTestClass):
    def test_numpy(self):
        te = TimeEffect(self.np1)
        te.orthogonalize()
        te = TimeEffect(self.np1)
        te.orthogonalize()

    def test_pandas(self):
        te = TimeEffect(self.pd1)
        te.orthogonalize()
        te = TimeEffect(self.pd2)

    def test_xarray(self):
        te = TimeEffect(self.xr1)
        te.orthogonalize()
        te = TimeEffect(self.xr2)
        te.orthogonalize()

    def test_numpy_missing_warning(self):
        temp = self.np2.copy()
        temp[1:, 1, -1] = np.nan
        te = TimeEffect(temp)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            te.orthogonalize()
            assert len(w) >= 1
            assert issubclass(w[0].category, SaturatedEffectWarning)


class TestFixedEffectSet(BaseTestClass):
    def test_wrong_data(self):
        ee = EntityEffect(self.np1)
        ge = GroupEffect(self.np1, [0])
        te = TimeEffect(self.np2)
        with pytest.raises(ValueError):
            ee + te
        fes = ee + ge
        with pytest.raises(ValueError):
            fes + te

        fes2 = te + EntityEffect(self.np2)
        with pytest.raises(ValueError):
            fes + fes2

    def test_add(self):
        ee = EntityEffect(self.np1)
        te = TimeEffect(self.np1)
        feset = ee + te
        feset = feset + feset
        ge = GroupEffect(self.np1, [0])
        feset += ge

    def test_add_error(self):
        ee = EntityEffect(self.np1)
        te = TimeEffect(self.np1)
        feset = ee + te
        with pytest.raises(ValueError):
            feset + 5
        with pytest.raises(ValueError):
            ee + 'a'
        with pytest.raises(ValueError):
            te + np.arange(10)

    def test_subtract(self):
        ee = EntityEffect(self.np1)
        te = TimeEffect(self.np1)
        feset = ee + te
        feset = feset - ee
        assert_equal(len(feset), 1)
        feset = ee + te
        ge = GroupEffect(self.np1, [0, 1])
        feset3 = ee + te + ge
        assert_equal(len(feset3), 3)
        feset1 = feset3 - feset
        assert_equal(len(feset1), 1)

    def test_subtract_error(self):
        ee = EntityEffect(self.np1)
        te = TimeEffect(self.np1)
        feset = ee + te
        feset = feset - ee
        with pytest.raises(ValueError):
            feset - ee
        with pytest.raises(ValueError):
            feset - 'a'
        ge = GroupEffect(self.np1, [0, 1])
        ge2 = GroupEffect(self.np1, [0, 2])
        feset = ee + te + ge
        feset2 = ee + ge2
        with pytest.raises(ValueError):
            feset - feset2

    def test_orthogonalize_pandas(self):
        ee = EntityEffect(self.pd1)
        te = TimeEffect(self.pd1)
        feset = ee + te
        feset.orthogonalize()

        ee = EntityEffect(self.pd2)
        te = TimeEffect(self.pd2)
        feset = ee + te
        feset.orthogonalize()

    def test_orthogonalize_xarray(self):
        ee = EntityEffect(self.xr1)
        te = TimeEffect(self.xr1)
        feset = ee + te
        feset.orthogonalize()
        ee = EntityEffect(self.xr2)
        te = TimeEffect(self.xr2)
        feset.orthogonalize()

    def test_str(self):
        ee = EntityEffect(self.np1)
        te = TimeEffect(self.np1)
        feset = ee + te
        assert 'FixedEffectSet' in feset.__str__()
        assert_equal(feset.__str__(), feset.__repr__())
        assert '<b>FixedEffectSet</b>' in feset._repr_html_()

    def test_direct_indirect(self):
        fes = EntityEffect(self.np2) + TimeEffect(self.np2)
        direct = fes.orthogonalize()

        ee = EntityEffect(self.np2)
        indirect = ee.orthogonalize()
        te = TimeEffect(indirect)
        indirect = te.orthogonalize()
        assert_almost_equal(direct, indirect)

        indirect = TimeEffect(self.np2).orthogonalize()
        indirect = EntityEffect(indirect).orthogonalize()
        assert_almost_equal(direct, indirect)


class TestGroupEffects(BaseTestClass):
    def test_numpy_time_effects(self):
        ge = GroupEffect(self.np1, [], time=True)
        out = ge.orthogonalize()
        assert_almost_equal(out, TimeEffect(self.np1).orthogonalize())
        ge = GroupEffect(self.np2, [], time=True)
        out = ge.orthogonalize()
        assert_almost_equal(out, TimeEffect(self.np2).orthogonalize())

    def test_numpy_entity_effects(self):
        ge = GroupEffect(self.np1, [], entity=True)
        out = ge.orthogonalize()
        assert_almost_equal(out, EntityEffect(self.np1).orthogonalize())
        ge = GroupEffect(self.np2, [], entity=True)
        out = ge.orthogonalize()
        assert_almost_equal(out, EntityEffect(self.np2).orthogonalize())

    def test_numpy_single_group(self):
        ge = GroupEffect(self.np1, [0])
        np1 = ge.orthogonalize()
        ge = GroupEffect(self.np2, [0])
        np2 = ge.orthogonalize()
        expected1 = demean_1col(self.np1, 0)
        assert_almost_equal(np1, expected1)
        expected2 = demean_1col(self.np2, 0)
        assert_almost_equal(np2, expected2)

    def test_numpy_double_group(self):
        ge = GroupEffect(self.np1, [0, 1])
        ge.orthogonalize()
        ge = GroupEffect(self.np2, [0, 1])
        ge.orthogonalize(self.np2)

    def test_numpy_group_time(self):
        ge = GroupEffect(self.np1, [0], time=True)
        ge.orthogonalize()
        ge = GroupEffect(self.np2, [0], time=True)
        ge.orthogonalize(self.np2)

    def test_numpy_group_entity(self):
        ge = GroupEffect(self.np1, [0], entity=True)
        ge.orthogonalize()
        ge = GroupEffect(self.np2, [0], entity=True)
        ge.orthogonalize()

    def test_pandas(self):
        ge = GroupEffect(self.pd1, [0])
        pd1 = ge.orthogonalize()
        ge = GroupEffect(self.pd2, [0])
        pd2 = ge.orthogonalize()

        ge = GroupEffect(self.np1, [0])
        np1 = ge.orthogonalize()
        ge = GroupEffect(self.np2, [0])
        np2 = ge.orthogonalize()

        assert_almost_equal(np1, pd1.values)
        assert_almost_equal(np2, pd2.values)

    def test_pandas_group_double(self):
        ge = GroupEffect(self.pd1, [0, 1])
        pd1 = ge.orthogonalize()
        ge = GroupEffect(self.pd2, [0, 1])
        pd2 = ge.orthogonalize()

        ge = GroupEffect(self.np1, [0, 1])
        np1 = ge.orthogonalize()
        ge = GroupEffect(self.np2, [0, 1])
        np2 = ge.orthogonalize()

        assert_almost_equal(np1, pd1.values)
        assert_almost_equal(np2, pd2.values)

    def test_pandas_time_effects(self):
        te = GroupEffect(self.pd1, [], time=True)
        out = te.orthogonalize()
        expected = TimeEffect(self.pd1).orthogonalize()
        assert_almost_equal(out.values, expected.values)
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(expected.minor_axis)

        te = GroupEffect(self.pd2, [], time=True)
        out = te.orthogonalize()
        expected = TimeEffect(self.pd2).orthogonalize()
        assert_almost_equal(out.values, expected.values)
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(expected.minor_axis)

    def test_pandas_group_effects(self):
        ge = GroupEffect(self.pd1, [], entity=True)
        out = ge.orthogonalize()
        expected = EntityEffect(self.pd1).orthogonalize()
        assert_almost_equal(out.values, expected.values)
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(expected.minor_axis)

        ge = GroupEffect(self.pd2, [], entity=True)
        out = ge.orthogonalize()
        expected = EntityEffect(self.pd2).orthogonalize()
        assert_almost_equal(out.values, expected.values)
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(expected.minor_axis)

    def test_group_time_entity_joint(self):
        ge = GroupEffect(self.pd2, [0], entity=True)
        out = ge.orthogonalize()
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(self.pd2.minor_axis)

        ge = GroupEffect(self.pd2, [0], time=True)
        out = ge.orthogonalize()
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(self.pd2.minor_axis)

    def test_group_time_entity_joint_existing_name(self):
        pd2 = self.pd2.copy()
        cols = list(pd2.minor_axis)
        cols[1] = '___entity_or_time__'
        pd2.minor_axis = cols

        ge = GroupEffect(pd2, [0], entity=True)
        out = ge.orthogonalize()
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(pd2.minor_axis)

        ge = GroupEffect(pd2, [0], time=True)
        out = ge.orthogonalize()
        assert isinstance(out, pd.Panel)
        assert list(out.minor_axis) == list(pd2.minor_axis)

    def test_xarray_single(self):
        ge = GroupEffect(self.xr1, [0])
        xr1 = ge.orthogonalize()
        ge = GroupEffect(self.xr2, [0])
        xr2 = ge.orthogonalize()
        ge = GroupEffect(self.np1, [0])
        np1 = ge.orthogonalize()
        ge = GroupEffect(self.np2, [0])
        np2 = ge.orthogonalize()
        assert_almost_equal(np1, xr1.values)
        assert_almost_equal(np2, xr2.values)

    def test_xarray_double(self):
        ge = GroupEffect(self.xr1, [0, 1])
        xr1 = ge.orthogonalize()
        ge = GroupEffect(self.xr2, [0, 1])
        xr2 = ge.orthogonalize()
        ge = GroupEffect(self.np1, [0, 1])
        np1 = ge.orthogonalize()
        ge = GroupEffect(self.np2, [0, 1])
        np2 = ge.orthogonalize()
        assert_almost_equal(np1, xr1.values)
        assert_almost_equal(np2, xr2.values)

    def test_str(self):
        ge = GroupEffect(self.pd1, ['var0'])
        assert 'var0' in ge.__str__()
        ge = GroupEffect(self.pd1, ['var0', 'var1'])
        assert 'var0' in ge.__str__()
        assert 'var1' in ge.__str__()
        ge = GroupEffect(self.np1, [0, 1])
        assert '0' in ge.__str__()
        assert '1' in ge.__str__()
        ge = GroupEffect(self.pd1, ['var0', 1])
        assert 'var0' in ge.__str__()
        assert '1' in ge.__str__()

    def test_errors(self):
        with pytest.raises(ValueError):
            GroupEffect(self.np1, [], entity=True, time=True)
        with pytest.raises(ValueError):
            GroupEffect(self.np1, np.arange(2))

    def test_pandas_categorical(self):
        panel = {}
        letters = list(string.ascii_lowercase * 1000)
        shape = len(letters)
        for i in range(10):
            df = pd.DataFrame({'var0': pd.Categorical(letters),
                               'var1': np.random.randn(shape),
                               'var2': np.random.randn(shape),
                               'var3': pd.Categorical(letters)})
            df['var0'] = pd.Categorical(df['var0'])
            panel['panel_' + str(i)] = df
        panel = pd.Panel(panel)

        te = TimeEffect(panel)
        te.orthogonalize()

        ee = EntityEffect(panel)
        ee.orthogonalize()

        ge = GroupEffect(panel, ['var0'])
        ge.orthogonalize()


class TestGroups(BaseTestClass):
    def test_numpy_smoke_entity(self):
        ee = EntityEffect(self.np1)
        ee.groups()
        ee = EntityEffect(self.np2)
        ee.groups()
        ee = EntityEffect(self.np3)
        ee.groups()

    def test_numpy_smoke_time(self):
        te = TimeEffect(self.np1)
        te.groups()
        te = TimeEffect(self.np2)
        te.groups()
        te = TimeEffect(self.np3)
        te.groups()

    def test_numpy_smoke_group(self):
        ge = GroupEffect(self.np1, [0])
        ge.groups()
        ge = GroupEffect(self.np2, [0])
        ge.groups()
        ge = GroupEffect(self.np3, [0])
        ge.groups()

    def test_numpy_smoke_group_time(self):
        ge = GroupEffect(self.np1, [0], time=True)
        ge.groups()
        ge = GroupEffect(self.np2, [0], time=True)
        ge.groups()
        ge = GroupEffect(self.np3, [0], time=True)
        ge.groups()

    def test_numpy_smoke_group_entity(self):
        ge = GroupEffect(self.np1, [0], entity=True)
        ge.groups()
        ge = GroupEffect(self.np2, [0], entity=True)
        ge.groups()
        ge = GroupEffect(self.np3, [0], entity=True)
        ge.groups()

    def test_pandas_smoke_entity(self):
        ee = EntityEffect(self.pd1)
        ee.groups()
        ee = EntityEffect(self.pd2)
        ee.groups()
        ee = EntityEffect(self.pd3)
        ee.groups()

    def test_pandas_smoke_time(self):
        te = TimeEffect(self.pd1)
        te.groups()
        te = TimeEffect(self.pd2)
        te.groups()
        te = TimeEffect(self.pd3)
        te.groups()

    def test_pandas_smoke_group(self):
        ge = GroupEffect(self.pd1, [0])
        groups = ge.groups()
        check_groups_pd(groups, self.pd1, [0])

        ge = GroupEffect(self.pd2, [0])
        groups = ge.groups()
        check_groups_pd(groups, self.pd2, [0])

        ge = GroupEffect(self.pd3, [0])
        groups = ge.groups()
        check_groups_pd(groups, self.pd3, [0])

    def test_pandas_smoke_group_2d(self):
        ge = GroupEffect(self.pd1, [0, 1])
        groups = ge.groups()
        check_groups_pd(groups, self.pd1, [0, 1])

        ge = GroupEffect(self.pd2, [0, 1])
        groups = ge.groups()
        check_groups_pd(groups, self.pd2, [0, 1])

        ge = GroupEffect(self.pd3, [0, 1])
        groups = ge.groups()
        check_groups_pd(groups, self.pd3, [0, 1])

    def test_pandas_smoke_group_time(self):
        ge = GroupEffect(self.pd1, [0], time=True)
        groups = ge.groups()
        check_groups_pd(groups, self.pd1, [0])
        ge = GroupEffect(self.pd2, [0], time=True)
        groups = ge.groups()
        check_groups_pd(groups, self.pd2, [0])
        ge = GroupEffect(self.pd3, [0], time=True)
        groups = ge.groups()
        check_groups_pd(groups, self.pd3, [0])

    def test_pandas_existing_column(self):
        pd2 = self.pd2.copy()
        cols = list(pd2.minor_axis)
        cols[1] = '___entity_or_time__'
        pd2.minor_axis = cols

        ge = GroupEffect(pd2, [0], entity=True)
        ge.groups()

    def test_pandas_smoke_group_entity(self):
        ge = GroupEffect(self.pd1, [0], entity=True)
        ge.groups()
        ge = GroupEffect(self.pd2, [0], entity=True)
        ge.groups()
        ge = GroupEffect(self.pd3, [0], entity=True)
        ge.groups()

    def test_pands_missing_groups(self):
        pd1_missing = self.pd1.copy()
        df = pd1_missing.swapaxes(0, 2).to_frame()
        df[1].loc[df[1] == 1] = 0
        pd1_missing = df.to_panel().swapaxes(0, 2)
        ge = GroupEffect(pd1_missing, [0, 1])
        groups = ge.groups()
        check_groups_pd(groups, pd1_missing, [0, 1])

    def test_xarray_smoke_group_2d(self):
        ge = GroupEffect(self.xr1, [0, 1])
        groups = ge.groups()

        ge = GroupEffect(self.xr2, [0, 1])
        groups = ge.groups()


class TestFixedEffect(BaseTestClass):
    def test_notimplemented(self):
        fe = FixedEffect(self.np1)
        with pytest.raises(RequiredSubclassingError):
            str(fe)
