import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
import scipy.sparse.coo
import scipy.sparse.csc
import scipy.sparse.csr

from linearmodels.panel.utility import (dummy_matrix, in_2core_graph,
                                        in_2core_graph_slow, preconditioner)

formats = {
    "csc": scipy.sparse.csc.csc_matrix,
    "csr": scipy.sparse.csr.csr_matrix,
    "coo": scipy.sparse.coo.coo_matrix,
    "array": np.ndarray,
}

pytestmark = pytest.mark.filterwarnings(
    "ignore:the matrix subclass:PendingDeprecationWarning"
)


@pytest.fixture(scope="module", params=formats)
def format(request):
    return request.param, formats[request.param]


def test_dummy_format(format):
    code, expected_type = format
    cats = np.zeros([15, 2], dtype=np.int8)
    cats[5:, 0] = 1
    cats[10:, 0] = 2
    cats[:, 1] = np.arange(15) % 5
    out, cond = dummy_matrix(cats, format=code, precondition=False)
    assert isinstance(out, expected_type)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 3, 3, 3, 3], dtype=np.int32)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)
    assert_array_equal(cond, np.ones(out.shape[1]))


def test_dummy_last():
    cats = np.zeros([15, 2], dtype=np.int8)
    cats[5:, 0] = 1
    cats[10:, 0] = 2
    cats[:, 1] = np.arange(15) % 5
    cats[-1, 1] = 0
    out, _ = dummy_matrix(cats, drop="last", precondition=False)
    assert isinstance(out, scipy.sparse.csc.csc_matrix)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 4, 3, 3, 3], dtype=np.int32)
    assert out.shape == (15, 3 + 5 - 1)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)


def test_invalid_format():
    cats = np.zeros([10, 1], dtype=np.int8)
    cats[5:, 0] = 1
    with pytest.raises(ValueError):
        dummy_matrix(cats, format="unknown", precondition=False)


def test_dummy_pandas():
    c1 = pd.Series(pd.Categorical(["a"] * 5 + ["b"] * 5 + ["c"] * 5))
    c2 = pd.Series(pd.Categorical(["A", "B", "C", "D", "E"] * 3))
    cats = pd.concat([c1, c2], 1)
    out, _ = dummy_matrix(cats, drop="last", precondition=False)
    assert isinstance(out, scipy.sparse.csc.csc_matrix)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 3, 3, 3, 3], dtype=np.int32)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)


def test_dummy_precondition():
    c1 = pd.Series(pd.Categorical(["a"] * 5 + ["b"] * 5 + ["c"] * 5))
    c2 = pd.Series(pd.Categorical(["A", "B", "C", "D", "E"] * 3))
    cats = pd.concat([c1, c2], 1)
    out_arr, cond_arr = dummy_matrix(
        cats, format="array", drop="last", precondition=True
    )
    out_csc, cond_csc = dummy_matrix(cats, format="csc", drop="last", precondition=True)
    out_csr, cond_csr = dummy_matrix(cats, format="csr", drop="last", precondition=True)
    assert_allclose((out_arr ** 2).sum(0), np.ones(out_arr.shape[1]))
    assert_allclose((out_csc.multiply(out_csc)).sum(0).A1, np.ones(out_arr.shape[1]))
    assert_allclose(cond_arr, cond_csc)
    assert_allclose(cond_csr, cond_csc)
    assert isinstance(out_csr, scipy.sparse.csr_matrix)


def test_drop_singletons_single():
    rs = np.random.RandomState(0)
    cats = rs.randint(0, 10000, (40000, 1))
    retain = in_2core_graph_slow(cats)
    nonsingletons = cats[retain]
    cats = pd.Series(cats[:, 0])
    vc = cats.value_counts()
    expected = np.sort(np.asarray(vc.index[vc > 1]))
    assert_array_equal(np.unique(nonsingletons), expected)
    assert vc[vc > 1].sum() == nonsingletons.shape[0]
    singletons = np.asarray(vc.index[vc == 1])
    assert nonsingletons.shape[0] == (40000 - singletons.shape[0])
    assert not np.any(np.isin(nonsingletons, singletons))


def test_drop_singletons_slow():
    rs = np.random.RandomState(0)
    c1 = rs.randint(0, 10000, (40000, 1))
    c2 = rs.randint(0, 20000, (40000, 1))
    cats = np.concatenate([c1, c2], 1)
    retain = in_2core_graph_slow(cats)
    nonsingletons = cats[retain]
    for col in (c1, c2):
        uniq, counts = np.unique(col, return_counts=True)
        assert not np.any(np.isin(col[retain], uniq[counts == 1]))

    idx = np.arange(40000)

    cols = {"c1": c1.copy(), "c2": c2.copy()}
    for i in range(40000):
        last = cols["c1"].shape[0]
        for col in cols:
            keep = in_2core_graph_slow(cols[col])
            for col2 in cols:
                cols[col2] = cols[col2][keep]
            idx = idx[keep]
        if cols["c1"].shape[0] == last:
            break

    expected = np.concatenate([c1[idx], c2[idx]], 1)
    assert_array_equal(nonsingletons, expected)
    expected = np.concatenate([cols["c1"], cols["c2"]], 1)
    assert_array_equal(nonsingletons, expected)

    dummies, _ = dummy_matrix(cats, format="csr", precondition=False)
    to_drop = dummies[~retain]
    assert to_drop.sum() == 2 * (~retain).sum()


def test_drop_singletons():
    rs = np.random.RandomState(0)
    c1 = rs.randint(0, 10000, (40000, 1))
    c2 = rs.randint(0, 20000, (40000, 1))
    cats = np.concatenate([c1, c2], 1)
    remain = in_2core_graph(cats)
    expected = in_2core_graph_slow(cats)
    assert_array_equal(remain, expected)


def test_drop_singletons_large():
    rs = np.random.RandomState(1234)
    m = 2000000
    c1 = rs.randint(0, m // 3, m)
    c2 = rs.randint(0, m // 20, m)
    cats = np.column_stack([c1, c2])
    retain = in_2core_graph(cats)
    expected = in_2core_graph_slow(cats)
    assert_array_equal(retain, expected)


def test_drop_singletons_pandas():
    rs = np.random.RandomState(0)
    c1 = rs.randint(0, 10000, (40000, 1))
    c2 = rs.randint(0, 20000, (40000, 1))
    df = [
        pd.Series(["{0}{1}".format(let, c) for c in cat.ravel()], dtype="category")
        for let, cat in zip("AB", (c1, c2))
    ]
    df = pd.concat(df, 1)
    df.columns = ["cat1", "cat2"]
    cats = df
    remain = in_2core_graph(cats)
    expected = in_2core_graph_slow(cats)
    assert_array_equal(remain, expected)


def test_preconditioner_copy():
    rs = np.random.RandomState(0)
    values = rs.standard_normal((100, 10))
    orig = values.copy()
    val_cond, cond = preconditioner(values, copy=True)
    assert_allclose(np.sqrt((orig ** 2).sum(0)), cond)
    assert id(val_cond) != id(values)
    assert_array_equal(orig, values)


def test_preconditioner_sparse():
    rs = np.random.RandomState(0)
    values = scipy.sparse.csc_matrix(rs.standard_normal((100, 10)))
    orig = values.copy()
    val_cond, cond = preconditioner(values, copy=True)
    assert_allclose(np.sqrt((values.multiply(values)).sum(0).A1), cond)
    assert id(val_cond) != id(values)
    assert_array_equal(orig.A, values.A)


def test_preconditioner_subclass():
    class subarray(np.ndarray):
        pass

    rs = np.random.RandomState(0)
    values = rs.standard_normal((100, 10))
    values = values.view(subarray)
    val_cond, cond = preconditioner(values, copy=True)
    assert_allclose(np.sqrt((values ** 2).sum(0)), cond)
    assert type(val_cond) == type(values)
