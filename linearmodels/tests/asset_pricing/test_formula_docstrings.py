"""
Tests of the examples in the docstrings of from_formula

Each example must run, and must create the model that the docstring documents.
"""

import doctest
import re

import pytest

from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)

MODELS = [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]


def examples_of(model):
    finder = doctest.DocTestFinder()
    tests = finder.find(model.from_formula, f"{model.__name__}.from_formula", globs={})
    assert len(tests) == 1
    return tests[0]


@pytest.mark.parametrize("model", MODELS)
def test_example_runs(model):
    test = examples_of(model)
    assert len(test.examples) > 0
    output = []
    runner = doctest.DocTestRunner(verbose=False)
    result = runner.run(test, out=output.append)
    assert result.failed == 0, "".join(output)


@pytest.mark.parametrize("model", MODELS)
def test_example_uses_the_documented_model(model):
    source = "".join(example.source for example in examples_of(model).examples)
    assert f"import {model.__name__}\n" in source
    calls = re.findall(r"(\w+)\.from_formula\(", source)
    assert len(calls) == 2
    assert set(calls) == {model.__name__}
