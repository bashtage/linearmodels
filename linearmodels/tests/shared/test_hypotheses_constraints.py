import numpy as np
import pandas as pd
import pytest

from linearmodels.shared.hypotheses import (
    _parse_single,
    _reparse_constraint_formula,
    quadratic_form_test,
)


def test_parse_single_simple():
    expr, val = _parse_single("x1 = 1")
    assert expr == "x1"
    assert val == 1.0


def test_parse_single_expression_with_plus():
    expr, val = _parse_single("x1 + x2 = 2.5")
    assert expr == "x1 + x2"
    assert val == 2.5


def test_parse_single_multiple_equals():
    expr, val = _parse_single("a = b = 1")
    assert expr == "a = b"
    assert val == 1.0


def test_parse_single_no_equals_raises():
    with pytest.raises(ValueError, match="required syntax"):
        _parse_single("x1")


def test_parse_single_non_float_rhs_raises():
    with pytest.raises(TypeError, match="required syntax"):
        _parse_single("x1 = not_a_number")


def test_reparse_dict_passthrough():
    spec = {"x1": 0.0, "x2": 1.0}
    out = _reparse_constraint_formula(spec)
    assert out == spec


def test_reparse_single_constraint_string_unchanged():
    s = "x1 + x2 = 1"
    assert _reparse_constraint_formula(s) is s


def test_reparse_multiple_equals_without_comma():
    out = _reparse_constraint_formula("x1 = x2 = 0")
    assert out == {"x1": 0.0, "x2": 0.0}


def test_reparse_comma_separated():
    out = _reparse_constraint_formula("x1 = 1, x2 = 2")
    assert out == {"x1": 1.0, "x2": 2.0}


def test_reparse_list_of_strings():
    out = _reparse_constraint_formula(["x1 = 1", "x2 = 2"])
    assert out == {"x1": 1.0, "x2": 2.0}


def test_reparse_comma_with_multiple_equals():
    out = _reparse_constraint_formula("a=b=1,c=2")
    assert out == {"a=b": 1.0, "c": 2.0}


def test_quadratic_form_formula_end_to_end():
    params = pd.Series([0.0, 1.0], index=["x0", "x1"])
    cov = np.eye(2)
    res = quadratic_form_test(params, cov, formula="x0=0")
    assert res.stat == 0.0
    assert res.df == 1


def test_quadratic_form_formula_and_restriction_exclusive():
    params = pd.Series([0.0, 0.0], index=["a", "b"])
    cov = np.eye(2)
    r = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="cannot be used simultaneously"):
        quadratic_form_test(params, cov, restriction=r, formula="a=0")


def test_quadratic_form_formula_requires_series_params():
    with pytest.raises(TypeError, match="pandas Series"):
        quadratic_form_test(np.array([0.0, 1.0]), np.eye(2), formula="x0=0")


def test_reparse_name_with_comma_single_equals():
    # One "=" in the string: pass through to formulaic as a single constraint.
    out = _reparse_constraint_formula("my,var = 1")
    assert out == "my,var = 1"
