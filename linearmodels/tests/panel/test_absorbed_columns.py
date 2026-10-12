"""
The test of absorbed regressors does not depend on the units of the variables.

The residual of a regressor that the effects absorb is not 0. It is the rounding error of
the demeaning, or the error of the iterative algorithm that removes the effects, and it
grows with the size of the regressor. Whether it is treated as a regressor must not
depend on the units of the regressor or of the other regressors.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels import PanelOLS
from linearmodels.panel.utility import (
    AbsorbingEffectError,
    AbsorbingEffectWarning,
    check_absorbed,
    not_absorbed,
)

ABSORBED = "absorbed_col"
PATHS = {
    "default": {},
    "low_memory": {"low_memory": True},
    "lsdv": {"use_lsdv": True},
    "lsmr": {"use_lsmr": True},
}


def make_panel(unbalanced, unit=1.0, const=False, seed=0, nentity=60, ntime=8):
    """A panel and a regressor that is the sum of an entity and a time variable"""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(ntime)], names=["entity", "time"]
    )
    if unbalanced:
        index = index[rng.random(len(index)) < 0.7]
    entity = rng.standard_normal(nentity)[index.codes[0]]
    time = rng.standard_normal(ntime)[index.codes[1]]
    x = pd.DataFrame(
        {
            "x1": rng.standard_normal(len(index)),
            "x2": rng.standard_normal(len(index)),
            ABSORBED: (entity + time) * unit,
        },
        index=index,
    )
    if const:
        x.insert(0, "const", 1.0)
    y = pd.Series(
        0.5 * x["x1"] - 0.3 * x["x2"] + entity + time + rng.standard_normal(len(index)),
        index=index,
        name="y",
    )
    return y, x


def absorbed_messages(record):
    return [
        str(w.message) for w in record if issubclass(w.category, AbsorbingEffectWarning)
    ]


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("unit", [1.0, 1e8, 1e-8])
@pytest.mark.parametrize("const", [False, True])
def test_absorbed_regressor_is_dropped(path, unbalanced, unit, const):
    y, x = make_panel(unbalanced, unit, const)
    options = PATHS[path]
    kept = [c for c in x if c != ABSORBED]
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED) as record:
        res = mod.fit(**options)
    assert len(absorbed_messages(record)) == 1
    assert list(res.params.index) == kept
    expected = PanelOLS(y, x[kept], entity_effects=True, time_effects=True)
    expected_res = expected.fit(**options)
    assert_allclose(res.params, expected_res.params, rtol=1e-6)
    assert_allclose(res.std_errors, expected_res.std_errors, rtol=1e-5)


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("unit", [1.0, 1e8])
def test_absorbed_regressor_raises_without_drop_absorbed(path, unbalanced, unit):
    y, x = make_panel(unbalanced, unit)
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    with pytest.raises(AbsorbingEffectError, match=ABSORBED):
        mod.fit(**PATHS[path])


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("const", [False, True])
@pytest.mark.parametrize("unit", [1.0, 1e8])
def test_absorbed_regressor_with_weights(path, const, unit):
    y, x = make_panel(True, unit, const)
    rng = np.random.default_rng(1)
    weights = pd.Series(np.exp(0.5 * rng.standard_normal(len(y))), index=y.index)
    options = PATHS[path]
    mod = PanelOLS(
        y,
        x,
        weights=weights,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED):
        res = mod.fit(**options)
    kept = [c for c in x if c != ABSORBED]
    expected = PanelOLS(
        y, x[kept], weights=weights, entity_effects=True, time_effects=True
    ).fit(**options)
    assert list(res.params.index) == kept
    assert_allclose(res.params, expected.params, rtol=1e-6)
    assert_allclose(res.std_errors, expected.std_errors, rtol=1e-6)
    with pytest.raises(AbsorbingEffectError, match=ABSORBED):
        PanelOLS(y, x, weights=weights, entity_effects=True, time_effects=True).fit(
            **options
        )


@pytest.mark.parametrize("unit", [1.0, 1e8])
def test_absorbed_regressor_with_other_effects(unit):
    y, x = make_panel(True, unit)
    rng = np.random.default_rng(2)
    group = pd.Series(rng.integers(0, 7, len(y)), index=y.index, name="group")
    shifts = rng.standard_normal(7)[group.to_numpy()]
    # The entity part is absorbed by the entity effects and the rest by the groups
    entity = x[ABSORBED].groupby(level=0).transform("mean")
    x[ABSORBED] = entity + shifts * unit
    mod = PanelOLS(y, x, entity_effects=True, other_effects=group, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED):
        res = mod.fit()
    kept = [c for c in x if c != ABSORBED]
    expected = PanelOLS(y, x[kept], entity_effects=True, other_effects=group).fit()
    assert list(res.params.index) == kept
    assert_allclose(res.params, expected.params, rtol=1e-6)


@pytest.mark.parametrize("units", [(1e-4, 1.0, 1e4), (1e-8, 1.0, 1e8)])
@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("drop_absorbed", [True, False])
def test_regressors_in_any_units_are_retained(units, path, drop_absorbed):
    y, x = make_panel(True)
    rng = np.random.default_rng(3)
    x = x[["x1", "x2"]].assign(x3=rng.standard_normal(len(y)))
    scaled = x * np.array(units)
    mod = PanelOLS(
        y,
        scaled,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=drop_absorbed,
        check_rank=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", AbsorbingEffectWarning)
        res = mod.fit(**PATHS[path])
    assert list(res.params.index) == ["x1", "x2", "x3"]
    if max(units) / min(units) > 1e8:
        return  # The least squares step is not accurate if the units differ by 1e16
    base = PanelOLS(y, x, entity_effects=True, time_effects=True).fit(**PATHS[path])
    assert_allclose(res.params * np.array(units), base.params, rtol=1e-6)


def test_constant_is_retained_when_the_absorbed_regressor_comes_first():
    y, x = make_panel(False, const=True)
    # Only the entity effects: use the part of the regressor that is entity-only
    x = x[[ABSORBED, "const", "x1", "x2"]].copy()
    x[ABSORBED] = x[ABSORBED].groupby(level=0).transform("mean")
    mod = PanelOLS(y, x, entity_effects=True, drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match=ABSORBED):
        res = mod.fit()
    assert list(res.params.index) == ["const", "x1", "x2"]
    expected = PanelOLS(y, x[["const", "x1", "x2"]], entity_effects=True).fit()
    assert_allclose(res.params, expected.params, rtol=1e-8)


UNITS = [(1e-8, 1.0, 1e8), (1e8, 1e-8, 1.0), (1e-12, 1.0, 1e12)]


@pytest.mark.parametrize("units", UNITS)
def test_not_absorbed_keeps_columns_in_any_units(units):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((300, 3)) * np.array(units)
    assert not_absorbed(x) == [0, 1, 2]
    assert not_absorbed(x, x_orig=x * 1000.0) == [0, 1, 2]
    check_absorbed(x, ["a", "b", "c"])
    check_absorbed(x, ["a", "b", "c"], x * 1000.0)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_not_absorbed_collinear_columns_in_any_units(scale):
    rng = np.random.default_rng(5)
    base = rng.standard_normal((300, 2))
    # The third column is a combination of the others, in other units
    x = np.column_stack([base * scale, (base[:, 0] + 2 * base[:, 1]) * scale * 1e-4])
    assert not_absorbed(x) == [0, 1]
    with pytest.raises(AbsorbingEffectError, match="a, b, c"):
        check_absorbed(x, ["a", "b", "c"])


def residual_of_share(rng, x_orig, share):
    """The first column of x_orig after it was absorbed, leaving a share of its norm"""
    noise = rng.standard_normal(x_orig.shape[0])
    return share * np.linalg.norm(x_orig[:, 1]) * noise / np.linalg.norm(noise)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("share", [0.0, 1e-9, 1e-7, 1e-6])
def test_not_absorbed_drops_the_error_of_the_algorithm(scale, share):
    rng = np.random.default_rng(6)
    x_orig = rng.standard_normal((300, 3)) * scale
    x = x_orig.copy()
    x[:, 1] = residual_of_share(rng, x_orig, share)
    assert not_absorbed(x, x_orig=x_orig) == [0, 2]
    with pytest.raises(AbsorbingEffectError):
        check_absorbed(x, ["a", "b", "c"], x_orig)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("share", [1e-3, 1e-4])
def test_not_absorbed_keeps_a_column_that_is_partly_absorbed(scale, share):
    rng = np.random.default_rng(7)
    x_orig = rng.standard_normal((300, 3)) * scale
    x = x_orig.copy()
    x[:, 1] = residual_of_share(rng, x_orig, share)
    assert not_absorbed(x, x_orig=x_orig) == [0, 1, 2]
    check_absorbed(x, ["a", "b", "c"], x_orig)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_regressor_restored_as_the_grand_mean_is_dropped_with_a_constant(position):
    # The effects absorb the third regressor and the mean is added back, so what is left
    # of the regressor is a multiple of the constant
    rng = np.random.default_rng(8)
    nobs = 300
    pure = rng.standard_normal((nobs, 2))
    between = rng.standard_normal((nobs, 3))
    means = np.array([0.5, -1.5, 3.0])
    x_orig = np.column_stack(
        [pure + between[:, :2] + means[:2], between[:, 2] + means[2], np.ones(nobs)]
    )
    x = np.column_stack(
        [
            pure + means[:2],
            means[2] + 1e-10 * rng.standard_normal(nobs),
            np.ones(nobs),
        ]
    )
    names = ["x0", "x1", "absorbed", "const"]
    order = [0, 1, 2, 3]
    order.insert(position, order.pop(3))  # Move the constant
    names = [names[i] for i in order]
    x, x_orig = x[:, order], x_orig[:, order]
    const, absorbed = names.index("const"), names.index("absorbed")
    expected = sorted(set(range(4)) - {absorbed})

    assert not_absorbed(x, True, const, x_orig) == expected
    with pytest.raises(AbsorbingEffectError) as exc_info:
        check_absorbed(x, names, x_orig)
    # The relation that makes the regressor redundant is the constant
    rows = [r.strip() for r in str(exc_info.value).splitlines() if "absorbed" in r]
    row = [r for r in rows if "," in r]
    assert ", ".join(sorted(["absorbed", "const"], key=names.index)) in row


def test_not_absorbed_without_original_data_finds_exact_zero_columns():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((100, 3))
    x[:, 1] = 0.0
    assert not_absorbed(x) == [0, 2]
    assert not_absorbed(np.zeros((100, 3))) == []
    assert not_absorbed(np.zeros((100, 3)), True, 1) == [1]
