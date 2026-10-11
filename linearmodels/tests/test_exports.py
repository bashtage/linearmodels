"""
Tests of the names that linearmodels exports

Every estimator that a sub-package exports is also exported from the top-level
package, the first-stage rank tests are exported from ``linearmodels.iv``, and
``linearmodels.iv.common`` and ``linearmodels.iv.results`` use ``__all__`` to
list only the names that are part of their public API.
"""

import importlib
import inspect

import pytest

import linearmodels
import linearmodels.iv
import linearmodels.iv.common
import linearmodels.iv.model
import linearmodels.iv.results

SUBPACKAGES = ["iv", "panel", "system", "asset_pricing"]

# Every estimator in the public API. An estimator is a class that a sub-package
# lists in its __all__ and that has a fit method. The tests find the estimators
# in this way, so that a new estimator is checked without being added here.
ESTIMATORS = {
    "AbsorbingLS",
    "BetweenOLS",
    "FamaMacBeth",
    "FirstDifferenceOLS",
    "IV2SLS",
    "IV3SLS",
    "IVGMM",
    "IVGMMCUE",
    "IVJIVE",
    "IVLIML",
    "IVSystemGMM",
    "LinearFactorModel",
    "LinearFactorModelGMM",
    "PanelOLS",
    "PooledOLS",
    "RandomEffects",
    "SUR",
    "TradedFactorModel",
}

RANK_TESTS = ["cragg_donald", "cragg_donald_f", "kleibergen_paap", "kleibergen_paap_f"]

RESULTS = [
    "AbsorbingLSResults",
    "FirstStageResults",
    "IVGMMResults",
    "IVModelComparison",
    "IVResults",
    "OLSResults",
    "compare",
]

MODULES_WITH_ALL = [
    "linearmodels",
    "linearmodels.iv",
    "linearmodels.iv.common",
    "linearmodels.iv.results",
    "linearmodels.panel",
    "linearmodels.system",
    "linearmodels.asset_pricing",
]


def _exported_estimators() -> list[tuple[str, str]]:
    """The (sub-package, name) of every estimator listed in a sub-package's __all__"""
    found = []
    for subpackage in SUBPACKAGES:
        module = importlib.import_module(f"linearmodels.{subpackage}")
        for name in module.__all__:
            obj = getattr(module, name, None)
            if inspect.isclass(obj) and hasattr(obj, "fit"):
                found.append((subpackage, name))
    return found


EXPORTED_ESTIMATORS = _exported_estimators()


def test_estimators_are_found():
    # With nothing to test, the parametrized test below is reported as xfail
    # (empty_parameter_set_mark) rather than failing, so check that the estimators
    # are found. This also fails if an estimator is dropped from the public API.
    found = {name for _, name in EXPORTED_ESTIMATORS}
    assert found >= ESTIMATORS


@pytest.mark.parametrize(
    ("subpackage", "name"),
    EXPORTED_ESTIMATORS,
    ids=[f"{subpackage}.{name}" for subpackage, name in EXPORTED_ESTIMATORS],
)
def test_estimator_exported_from_top_level(subpackage, name):
    module = importlib.import_module(f"linearmodels.{subpackage}")
    assert name in linearmodels.__all__
    assert getattr(linearmodels, name) is getattr(module, name)


def test_ivjive_exported_from_top_level():
    assert "IVJIVE" in linearmodels.__all__
    assert linearmodels.IVJIVE is linearmodels.iv.model.IVJIVE
    assert linearmodels.IVJIVE is linearmodels.iv.IVJIVE


@pytest.mark.parametrize("name", RANK_TESTS)
def test_rank_tests_exported_from_iv(name):
    assert name in linearmodels.iv.__all__
    assert getattr(linearmodels.iv, name) is getattr(linearmodels.iv.common, name)


def test_common_all():
    common = linearmodels.iv.common
    assert sorted(common.__all__) == sorted(RANK_TESTS)
    # Importable, since the models use them, but not part of the public API
    for helper in ("find_constant", "f_statistic"):
        assert callable(getattr(common, helper))
        assert helper not in common.__all__


def test_results_all():
    results = linearmodels.iv.results
    assert sorted(results.__all__) == sorted(RESULTS)
    # Defined in results, not imported into it
    for name in RESULTS:
        assert getattr(results, name).__module__ == "linearmodels.iv.results"
    # The rank tests are imported by results but are part of iv.common
    for name in RANK_TESTS:
        assert name not in results.__all__
    # Importable, since other modules use them, but not part of the public API
    for helper in ("stub_concat", "table_concat"):
        assert callable(getattr(results, helper))
        assert helper not in results.__all__


@pytest.mark.parametrize("module_name", MODULES_WITH_ALL)
def test_all_is_valid(module_name):
    module = importlib.import_module(module_name)
    assert len(module.__all__) == len(set(module.__all__))
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing
