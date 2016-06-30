import string
import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
from nose.tools import assert_true, assert_is_instance
from numpy.testing import assert_equal, assert_raises, assert_almost_equal

from panel.fixed_effects import EntityEffect, TimeEffect, GroupEffect, \
    SaturatedEffectWarning, DummyVariableIterator


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
                assert_true((temp[col1] == temp[col1].iloc[0]).all())
            if col2 is not None:
                assert_true((temp[col2] == temp[col2].iloc[0]).all())


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


class BaseTestClass(TestCase):
    @classmethod
    def setUpClass(cls):
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

        np3 = np.zeros((1000, 17, 4))
        np3 = np3.reshape((17000, 4))
        np3[:, 0] = np.arange(17000) % 13
        for i in range(13):
            np3[np3[:, 0] == i, 1:] = np.random.randn(3)
        np3 = np3.reshape((1000, 17, 4))
        cls.np3 = np3

        cls.np1_missing = cls.np1.copy()
        cls.np1_missing[:, ::3, :] = np.nan

        cls.np1_missing_warning = cls.np1.copy()
        cls.np1_missing_warning[:, 0::2, :] = np.nan

        cls.pd1 = pd.Panel(cls.np1.copy())
        cls.pd2 = pd.Panel(cls.np2.copy())
        cls.pd3 = pd.Panel(cls.np3.copy())

        cls.xr1 = xr.DataArray(cls.np1.copy())
        cls.xr2 = xr.DataArray(cls.np2.copy())


class TestEntityEffect(BaseTestClass):
    def test_invalid_data(self):
        ge = EntityEffect()
        assert_raises(ValueError, ge.orthogonalize, self.np1[0])
        assert_raises(ValueError, ge.orthogonalize, self.pd1.iloc[0])
        assert_raises(ValueError, ge.orthogonalize, self.xr1[0])

    def test_numpy(self):
        ee = EntityEffect()
        out = ee.orthogonalize(self.np1)
        expected = self.np1.copy()
        for i in range(expected.shape[0]):
            expected[i] -= expected[i].mean(0)
        assert_almost_equal(expected, out)

        out = ee.orthogonalize(self.np2)
        expected = self.np2.copy()
        for i in range(expected.shape[0]):
            expected[i] -= expected[i].mean(0)
        assert_almost_equal(expected, out)

    def test_numpy_exclude(self):
        ee = EntityEffect()
        out = ee.orthogonalize(self.np1, np.arange(10, self.np1.shape[2]))
        for c in range(10):
            assert_true((self.np1[:, :, c] != out[:, :, c]).any())

        for c in range(10, self.np1.shape[2]):
            assert_equal(self.np1[:, :, c], out[:, :, c])

        indices = [2, 5, 9]
        out = ee.orthogonalize(self.np2, indices)
        for c in list(range(self.np2.shape[2])):
            if c in indices:
                continue
            assert_true((self.np2[:, :, c] != out[:, :, c]).any())
        for c in indices:
            assert_equal(self.np2[:, :, c], out[:, :, c])

    def test_numpy_missing(self):
        ee = EntityEffect()

        out = ee.orthogonalize(self.np1_missing)
        expected = self.np1_missing.copy()
        for i in range(expected.shape[0]):
            expected[i] -= np.nanmean(expected[i], 0)
        assert_almost_equal(expected, out)

    def test_numpy_missing_warning(self):
        ee = EntityEffect()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ee.orthogonalize(self.np1_missing_warning)
            assert_true(len(w) >= 1)
            assert_true(issubclass(w[0].category, SaturatedEffectWarning))

    def test_pandas(self):
        ee = EntityEffect()
        out = ee.orthogonalize(self.pd1)
        expected_values = ee.orthogonalize(self.np1)
        assert_almost_equal(out.values, expected_values)

        out = ee.orthogonalize(self.pd2)
        expected_values = ee.orthogonalize(self.np2)
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

        ee = EntityEffect()
        ee.orthogonalize(panel)

    def test_xarray(self):
        ee = EntityEffect()
        out = ee.orthogonalize(self.xr1)
        expected_values = ee.orthogonalize(self.np1)
        assert_almost_equal(out.values, expected_values)

        out = ee.orthogonalize(self.xr2)
        expected_values = ee.orthogonalize(self.np2)
        assert_almost_equal(out.values, expected_values)

    def test_dummies(self):
        ee = EntityEffect()
        dummies = ee.dummies(self.np1)
        n,t,k = self.np1.shape
        assert_equal(dummies.shape, (n*t, n))

    def test_dummies_drop(self):
        ee = EntityEffect()
        dummies = ee.dummies(self.np1, drop=True)
        n, t, k = self.np1.shape
        assert_equal(dummies.shape, (n * t, n - 1))

    def test_dummy_iterator(self):
        ee = EntityEffect()
        dummies = ee.dummies(self.np1, iterator=True)
        assert_is_instance(dummies, DummyVariableIterator)

    def test_estimate(self):
        ee = EntityEffect()
        exog = self.np1[:,:,6:]
        endog = self.np1[:, :, [5]]
        ee.estimate(endog,None)
        ee.estimate(endog, None, True)
        ee.estimate(endog, exog)
        ee.estimate(endog, exog, True)

class TestTimeEffect(BaseTestClass):
    def test_numpy(self):
        te = TimeEffect()
        te.orthogonalize(self.np1)
        te.orthogonalize(self.np2)

    def test_pandas(self):
        te = TimeEffect()
        te.orthogonalize(self.pd1)
        te.orthogonalize(self.pd2)

    def test_xarray(self):
        te = TimeEffect()
        te.orthogonalize(self.xr1)
        te.orthogonalize(self.xr2)

    def test_numpy_missing_warning(self):
        te = TimeEffect()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            temp = self.np2.copy()
            temp[1:, 1, -1] = np.nan
            te.orthogonalize(temp)
            assert_true(len(w) >= 1)
            assert_true(issubclass(w[0].category, SaturatedEffectWarning))


class TestFixedEffectSet(BaseTestClass):
    def test_add(self):
        ee = EntityEffect()
        te = TimeEffect()
        feset = ee + te
        feset = feset + feset
        ge = GroupEffect([0])
        feset += ge

    def test_add_error(self):
        ee = EntityEffect()
        te = TimeEffect()
        feset = ee + te
        assert_raises(ValueError, feset.__add__, 5)
        assert_raises(ValueError, ee.__add__, 'a')
        assert_raises(ValueError, te.__add__, np.arange(10))

    def test_subtract(self):
        ee = EntityEffect()
        te = TimeEffect()
        feset = ee + te
        feset = feset - ee
        assert_equal(len(feset.fixed_effects), 1)
        feset = ee + te
        ge = GroupEffect([0, 1])
        feset3 = ee + te + ge
        assert_equal(len(feset3.fixed_effects), 3)
        feset1 = feset3 - feset
        assert_equal(len(feset1.fixed_effects), 1)

    def test_subtract_error(self):
        ee = EntityEffect()
        te = TimeEffect()
        feset = ee + te
        feset = feset - ee
        assert_raises(ValueError, feset.__sub__, ee)
        assert_raises(ValueError, feset.__sub__, 'a')
        ge = GroupEffect([0, 1])
        ge2 = GroupEffect([0, 2])
        feset = ee + te + ge
        feset2 = ee + ge2
        assert_raises(ValueError, feset.__sub__, feset2)

    def test_orthogonalize(self):
        ee = EntityEffect()
        te = TimeEffect()
        feset = ee + te
        feset.orthogonalize(self.xr1)
        feset.orthogonalize(self.xr2)
        feset.orthogonalize(self.pd1)
        feset.orthogonalize(self.pd2)

    def test_str(self):
        ee = EntityEffect()
        te = TimeEffect()
        feset = ee + te
        assert_true('FixedEffectSet' in feset.__str__())
        assert_equal(feset.__str__(), feset.__repr__())
        assert_true('<b>FixedEffectSet</b>' in feset._repr_html_())

    def test_direct_indirect(self):
        fes = EntityEffect() + TimeEffect()
        direct = fes.orthogonalize(self.np2)

        ee = EntityEffect()
        te = TimeEffect()
        indirect = ee.orthogonalize(self.np2)
        indirect = te.orthogonalize(indirect)
        assert_almost_equal(direct, indirect)

        indirect = te.orthogonalize(self.np2)
        indirect = ee.orthogonalize(indirect)
        assert_almost_equal(direct, indirect)


class TestGroupEffects(BaseTestClass):
    def test_numpy_time_effects(self):
        ge = GroupEffect([], time=True)
        out = ge.orthogonalize(self.np1)
        assert_almost_equal(out, TimeEffect().orthogonalize(self.np1))
        out = ge.orthogonalize(self.np2)
        assert_almost_equal(out, TimeEffect().orthogonalize(self.np2))

    def test_numpy_entity_effects(self):
        ge = GroupEffect([], entity=True)
        out = ge.orthogonalize(self.np1)
        assert_almost_equal(out, EntityEffect().orthogonalize(self.np1))
        out = ge.orthogonalize(self.np2)
        assert_almost_equal(out, EntityEffect().orthogonalize(self.np2))

    def test_numpy_single_group(self):
        ge = GroupEffect([0])
        np1 = ge.orthogonalize(self.np1)
        np2 = ge.orthogonalize(self.np2)
        expected1 = demean_1col(self.np1, 0)
        assert_almost_equal(np1, expected1)
        expected2 = demean_1col(self.np2, 0)
        assert_almost_equal(np2, expected2)

    def test_numpy_double_group(self):
        ge = GroupEffect([0, 1])
        ge.orthogonalize(self.np1)
        ge.orthogonalize(self.np2)

    def test_numpy_group_time(self):
        ge = GroupEffect([0], time=True)
        ge.orthogonalize(self.np1)
        ge.orthogonalize(self.np2)

    def test_numpy_group_entity(self):
        ge = GroupEffect([0], entity=True)
        ge.orthogonalize(self.np1)
        ge.orthogonalize(self.np2)

    def test_pandas(self):
        ge = GroupEffect([0])
        pd1 = ge.orthogonalize(self.pd1)
        pd2 = ge.orthogonalize(self.pd2)

        np1 = ge.orthogonalize(self.np1)
        np2 = ge.orthogonalize(self.np2)

        assert_almost_equal(np1, pd1.values)
        assert_almost_equal(np2, pd2.values)

        ge = GroupEffect([0, 1])
        pd1 = ge.orthogonalize(self.pd1)
        pd2 = ge.orthogonalize(self.pd2)

        np1 = ge.orthogonalize(self.np1)
        np2 = ge.orthogonalize(self.np2)

        assert_almost_equal(np1, pd1.values)
        assert_almost_equal(np2, pd2.values)

    def test_pandas_time_effects(self):
        te = GroupEffect([], time=True)
        out = te.orthogonalize(self.pd1)
        expected = TimeEffect().orthogonalize(self.pd1)
        assert_almost_equal(out.values, expected.values)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(expected.minor_axis))

        te = GroupEffect([], time=True)
        out = te.orthogonalize(self.pd2)
        expected = TimeEffect().orthogonalize(self.pd2)
        assert_almost_equal(out.values, expected.values)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(expected.minor_axis))

    def test_pandas_group_effects(self):
        ge = GroupEffect([], entity=True)
        out = ge.orthogonalize(self.pd1)
        expected = EntityEffect().orthogonalize(self.pd1)
        assert_almost_equal(out.values, expected.values)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(expected.minor_axis))

        ge = GroupEffect([], entity=True)
        out = ge.orthogonalize(self.pd2)
        expected = EntityEffect().orthogonalize(self.pd2)
        assert_almost_equal(out.values, expected.values)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(expected.minor_axis))

    def test_group_time_entity_joint(self):
        ge = GroupEffect([0], entity=True)
        out = ge.orthogonalize(self.pd2)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(self.pd2.minor_axis))

        ge = GroupEffect([0], time=True)
        out = ge.orthogonalize(self.pd2)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(self.pd2.minor_axis))

    def test_group_time_entity_joint_existing_name(self):
        ge = GroupEffect([0], entity=True)
        pd2 = self.pd2.copy()
        cols = list(pd2.minor_axis)
        cols[1] = '___entity_or_time__'
        pd2.minor_axis = cols
        out = ge.orthogonalize(pd2)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(pd2.minor_axis))

        ge = GroupEffect([0], time=True)
        out = ge.orthogonalize(pd2)
        assert_is_instance(out, pd.Panel)
        assert_true(list(out.minor_axis) == list(pd2.minor_axis))

    def test_xarray(self):
        ge = GroupEffect([0])
        xr1 = ge.orthogonalize(self.xr1)
        xr2 = ge.orthogonalize(self.xr2)
        np1 = ge.orthogonalize(self.np1)
        np2 = ge.orthogonalize(self.np2)
        assert_almost_equal(np1, xr1.values)
        assert_almost_equal(np2, xr2.values)

        ge = GroupEffect([0, 1])
        xr1 = ge.orthogonalize(self.xr1)
        xr2 = ge.orthogonalize(self.xr2)
        np1 = ge.orthogonalize(self.np1)
        np2 = ge.orthogonalize(self.np2)
        assert_almost_equal(np1, xr1.values)
        assert_almost_equal(np2, xr2.values)

    def test_str(self):
        ge = GroupEffect(['var0'])
        assert_true('var0' in ge.__str__())
        ge = GroupEffect(['var0', 'var1'])
        assert_true('var0' in ge.__str__())
        assert_true('var1' in ge.__str__())
        ge = GroupEffect([0, 1])
        assert_true('0' in ge.__str__())
        assert_true('1' in ge.__str__())
        ge = GroupEffect(['var0', 1])
        assert_true('var0' in ge.__str__())
        assert_true('1' in ge.__str__())

    def test_errors(self):
        assert_raises(ValueError, GroupEffect, [], entity=True, time=True)
        assert_raises(ValueError, GroupEffect, np.arange(2))

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

        te = TimeEffect()
        te.orthogonalize(panel)

        ee = EntityEffect()
        ee.orthogonalize(panel)

        ge = GroupEffect(['var0'])
        ge.orthogonalize(panel)


class TestGroups(BaseTestClass):
    def test_numpy_smoke(self):
        ee = EntityEffect()
        ee.groups(self.np1)
        ee.groups(self.np2)
        ee.groups(self.np3)

        te = TimeEffect()
        te.groups(self.np1)
        te.groups(self.np2)
        te.groups(self.np3)

        ge = GroupEffect([0])
        ge.groups(self.np1)
        ge.groups(self.np2)
        ge.groups(self.np3)

        ge = GroupEffect([0], time=True)
        ge.groups(self.np1)
        ge.groups(self.np2)
        ge.groups(self.np3)

        ge = GroupEffect([0], entity=True)
        ge.groups(self.np1)
        ge.groups(self.np2)
        ge.groups(self.np3)

    def test_pandas_smoke(self):
        ee = EntityEffect()
        ee.groups(self.pd1)
        ee.groups(self.pd2)
        ee.groups(self.pd3)

        te = TimeEffect()
        te.groups(self.pd1)
        te.groups(self.pd2)
        te.groups(self.pd3)

        ge = GroupEffect([0])
        groups = ge.groups(self.pd1)
        check_groups_pd(groups, self.pd1, [0])
        ge = GroupEffect([0])
        groups = ge.groups(self.pd2)
        check_groups_pd(groups, self.pd2, [0])
        ge = GroupEffect([0])
        groups = ge.groups(self.pd3)
        check_groups_pd(groups, self.pd3, [0])

        ge = GroupEffect([0, 1])
        groups = ge.groups(self.pd1)
        check_groups_pd(groups, self.pd1, [0, 1])
        ge = GroupEffect([0, 1])
        groups = ge.groups(self.pd2)
        check_groups_pd(groups, self.pd2, [0, 1])
        ge = GroupEffect([0, 1])
        groups = ge.groups(self.pd3)
        check_groups_pd(groups, self.pd3, [0, 1])

        ge = GroupEffect([0], time=True)
        groups = ge.groups(self.pd1)
        check_groups_pd(groups, self.pd1, [0])
        ge = GroupEffect([0], time=True)
        groups = ge.groups(self.pd2)
        check_groups_pd(groups, self.pd2, [0])
        ge = GroupEffect([0], time=True)
        groups = ge.groups(self.pd3)
        check_groups_pd(groups, self.pd3, [0])

        ge = GroupEffect([0], entity=True)
        ge.groups(self.pd1)
        ge = GroupEffect([0], entity=True)
        ge.groups(self.pd2)
        ge = GroupEffect([0], entity=True)
        ge.groups(self.pd3)

    def test_pands_missing_groups(self):
        ge = GroupEffect([0, 1])
        pd1_missing = self.pd1.copy()
        df = pd1_missing.swapaxes(0, 2).to_frame()
        df[1].loc[df[1] == 1] = 0
        pd1_missing = df.to_panel().swapaxes(0, 2)
        groups = ge.groups(pd1_missing)

        check_groups_pd(groups, pd1_missing, [0, 1])

