from linearmodels.compat.scipy import coo_array, csc_array, csr_array

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from linearmodels.panel import PanelOLS
from linearmodels.panel.utility import (
    AbsorbingEffectError,
    PanelModelData,
    check_absorbed,
    dummy_matrix,
    generate_panel_data,
    in_2core_graph,
    in_2core_graph_slow,
    not_absorbed,
    preconditioner,
)

formats = {
    "csc": csc_array,
    "csr": csr_array,
    "coo": coo_array,
}


@pytest.fixture(scope="module", params=formats)
def dummy_format(request):
    return request.param, formats[request.param]


def test_dummy_format(dummy_format):
    code, expected_type = dummy_format
    cats = np.zeros([15, 2], dtype=np.int8)
    cats[5:, 0] = 1
    cats[10:, 0] = 2
    cats[:, 1] = np.arange(15) % 5
    out, cond = dummy_matrix(cats, output_format=code, precondition=False)
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
    assert isinstance(out, csc_array)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 4, 3, 3, 3], dtype=np.int32)
    assert out.shape == (15, 3 + 5 - 1)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)


def test_invalid_format():
    cats = np.zeros([10, 1], dtype=np.int8)
    cats[5:, 0] = 1
    with pytest.raises(ValueError, match=r"Unknown format"):
        dummy_matrix(cats, output_format="unknown", precondition=False)


def test_dummy_pandas():
    c1 = pd.Series(pd.Categorical(["a"] * 5 + ["b"] * 5 + ["c"] * 5))
    c2 = pd.Series(pd.Categorical(["A", "B", "C", "D", "E"] * 3))
    cats = pd.concat([c1, c2], axis=1)
    out, _ = dummy_matrix(cats, drop="last", precondition=False)
    assert isinstance(out, csc_array)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 3, 3, 3, 3], dtype=np.int32)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)


def test_dummy_precondition():
    c1 = pd.Series(pd.Categorical(["a"] * 5 + ["b"] * 5 + ["c"] * 5))
    c2 = pd.Series(pd.Categorical(["A", "B", "C", "D", "E"] * 3))
    cats = pd.concat([c1, c2], axis=1)
    csc = dummy_matrix(cats, output_format="csc", drop="last", precondition=True)
    out_csc: csc_array = csc[0]
    cond_csc: np.ndarray = csc[1]
    csr = dummy_matrix(cats, output_format="csr", drop="last", precondition=True)
    out_csr: csr_array = csr[0]
    cond_csr: np.ndarray = csr[1]
    assert_allclose(
        np.asarray((out_csc.multiply(out_csc)).sum(0)).ravel(),
        np.ones(out_csc.shape[1]),
    )
    assert_allclose(cond_csr, cond_csc)
    assert isinstance(out_csc, csc_array)
    assert isinstance(out_csr, csr_array)


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
    for _ in range(40000):
        last = cols["c1"].shape[0]
        for col_value in cols.values():
            keep = in_2core_graph_slow(col_value)
            for col2, col2_value in cols.items():
                cols[col2] = col2_value[keep]
            idx = idx[keep]
        if cols["c1"].shape[0] == last:
            break

    expected = np.concatenate([c1[idx], c2[idx]], 1)
    assert_array_equal(nonsingletons, expected)
    expected = np.concatenate([cols["c1"], cols["c2"]], 1)
    assert_array_equal(nonsingletons, expected)

    dummies, _ = dummy_matrix(cats, output_format="csr", precondition=False)
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
        pd.Series([f"{let}{c}" for c in cat.ravel()], dtype="category")
        for let, cat in zip("AB", (c1, c2), strict=False)
    ]
    df = pd.concat(df, axis=1)
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
    assert_allclose(np.sqrt((orig**2).sum(0)), cond)
    assert id(val_cond) != id(values)
    assert_array_equal(orig, values)


def test_preconditioner_sparse():
    rs = np.random.RandomState(0)
    values = csc_array(rs.standard_normal((100, 10)))
    orig = values.copy()
    val_cond, cond = preconditioner(values, copy=True)
    assert_allclose(np.asarray(np.sqrt((values.multiply(values)).sum(0))).ravel(), cond)
    assert id(val_cond) != id(values)
    assert_array_equal(orig.toarray(), values.toarray())


def test_preconditioner_subclass():
    class SubArray(np.ndarray):
        pass

    rs = np.random.RandomState(0)
    values = rs.standard_normal((100, 10))
    values = values.view(SubArray)
    val_cond, cond = preconditioner(values, copy=True)
    assert_allclose(np.sqrt((values**2).sum(0)), cond)
    assert type(val_cond) is type(values)
    # Test in-place
    val_cond, cond = preconditioner(values, copy=False)
    assert_allclose(np.sqrt((values**2).sum(0)), np.ones(10))
    assert type(val_cond) is type(values)


@pytest.mark.parametrize("missing", [0, 0.2])
@pytest.mark.parametrize("const", [True, False])
@pytest.mark.parametrize("other_effects", [0, 1, 2])
@pytest.mark.parametrize("cat_list", [True, False])
@pytest.mark.parametrize(
    "random_state", [None, np.random.RandomState(0), np.random.default_rng(0)]
)
def test_generate_panel_data(missing, const, other_effects, cat_list, random_state):
    if cat_list:
        ncats: list[int] | int = [13] * other_effects
    else:
        ncats = 21

    dataset = generate_panel_data(
        missing=missing,
        const=const,
        other_effects=other_effects,
        ncats=ncats,
        rng=random_state,
    )

    assert isinstance(dataset, PanelModelData)
    if missing > 0:
        assert np.any(np.asarray(np.isnan(dataset.data)))
    if const:
        assert "const" in dataset.data
        assert (dataset.data["const"].dropna() == 1.0).all()
    assert dataset.other_effects.shape == (dataset.data.shape[0], other_effects)


@pytest.mark.parametrize("make_rng", [np.random.RandomState, np.random.default_rng])
def test_generate_panel_data_rng(make_rng):
    # A Generator has integers and not randint. The same seed gives the same data
    options = {"nentity": 30, "ntime": 5, "other_effects": 2, "ncats": [3, 4]}
    first = generate_panel_data(rng=make_rng(1234), **options)
    second = generate_panel_data(rng=make_rng(1234), **options)
    other = generate_panel_data(rng=make_rng(4321), **options)
    for result in (first, second, other):
        assert result.data.shape == (150, 6)
        assert result.other_effects.shape == (150, 2)
    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.other_effects, second.other_effects)
    pd.testing.assert_frame_equal(first.clusters, second.clusters)
    assert not first.data.equals(other.data)
    # The categories of the effects are integers in the ranges
    effects = first.other_effects
    assert set(effects["cat.0"].unique()) <= set(range(3))
    assert set(effects["cat.1"].unique()) <= set(range(4))


def test_generate_panel_data_random_state_is_unchanged():
    # The draws of a RandomState do not depend on Generators being accepted
    result = generate_panel_data(
        nentity=6,
        ntime=3,
        nexog=2,
        other_effects=2,
        ncats=[3, 5],
        rng=np.random.RandomState(7),
    )
    effects = result.other_effects.to_numpy().T
    assert_array_equal(
        effects,
        [
            [1, 0, 1, 2, 1, 0, 1, 1, 0, 2, 0, 2, 2, 1, 1, 0, 1, 1],
            [2, 1, 1, 0, 3, 1, 2, 1, 3, 0, 0, 2, 2, 0, 0, 1, 1, 2],
        ],
    )


def test_generate_panel_data_clusters_random_state_is_unchanged():
    # The clusters are the draws that the second column of the clusters had
    result = generate_panel_data(
        nentity=6,
        ntime=3,
        nexog=2,
        other_effects=2,
        ncats=[3, 5],
        rng=np.random.RandomState(7),
    )
    first_periods = result.clusters.iloc[::3]
    assert_array_equal(first_periods["varcat.0"], [2, 2, 0, 2, 0, 0])
    assert_array_equal(first_periods["varcat.1"], [1, 0, 1, 0, 2, 2])


@pytest.mark.parametrize("missing", [0, 0.1])
def test_generate_panel_data_clusters(missing):
    # The clusters have one row for each observation, and two variables
    result = generate_panel_data(nentity=20, ntime=5, missing=missing)
    assert result.clusters.shape == (result.data.shape[0], 2)
    assert result.clusters.index.equals(result.data.index)
    assert not result.clusters.isnull().any().any()
    assert result.clusters.to_numpy().max() < 10
    assert list(result.clusters.columns) == ["varcat.0", "varcat.1"]
    # They are the clusters of the entities: the same in all periods of an entity
    assert (result.clusters.groupby(level=0).nunique() == 1).all().all()
    # And can be used in a covariance
    data = result.data.dropna()
    clusters = result.clusters.loc[data.index]
    res = PanelOLS(data["y"], data[["x0", "x1"]], entity_effects=True).fit(
        cov_type="clustered", clusters=clusters
    )
    assert np.all(np.isfinite(res.params))


def test_not_absorbed_const():
    x = np.random.standard_normal((200, 3))
    x[:, 0] = 0
    na = not_absorbed(x, True, 0)
    assert na == [0, 1, 2]
    x[:, 0] = x[:, 1]
    x[:, 1] = 0
    na = not_absorbed(x, True, 1)
    assert na == [0, 1, 2]


def test_all_absorbed_const():
    x = np.zeros((200, 3))
    na = not_absorbed(x, True, 0)
    assert na == [0]
    na = not_absorbed(x, True, 1)
    assert na == [1]


def test_all_absorbed_exception():
    x_orig = np.random.standard_normal((200, 3))
    x = x_orig * 1e-32
    with pytest.raises(AbsorbingEffectError, match=r"All exog variables have been"):
        check_absorbed(x, ["a", "b", "c"], x_orig)


def test_dummy_matrix_bad_order():
    cats = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="Unknown drop option: bad"):
        dummy_matrix(cats, drop="bad")


def test_case_insensitive_drop_option():
    cats = np.array([[0, 1], [1, 0]])
    # Should not raise an error for "First" or "LAST"
    first_0 = dummy_matrix(cats, drop="first")
    first_1 = dummy_matrix(cats, drop="First")
    last_0 = dummy_matrix(cats, drop="last")
    last_1 = dummy_matrix(cats, drop="LAST")
    assert_allclose(first_0[0].toarray(), first_1[0].toarray())
    assert_allclose(last_0[0].toarray(), last_1[0].toarray())
