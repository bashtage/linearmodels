"""
Tests of the error for a name that is not supported, and of the functions that make it
"""

import copy
import pickle
import traceback

import pytest

from linearmodels.shared.exceptions import (
    UnknownOptionError,
    lookup_option,
    unknown_option_error,
)

TABLE = {"robust": 1, "clustered": 2, "Kernel": 3, "unadjusted": 4}


def test_unknown_option_error_is_a_value_error_and_a_key_error():
    error = UnknownOptionError("message")
    assert isinstance(error, ValueError)
    assert isinstance(error, KeyError)
    # Code written for releases that raised either of them keeps working
    for exception in (ValueError, KeyError, LookupError, Exception):
        with pytest.raises(exception, match="message"):
            raise error


def test_str_is_the_message_and_is_not_quoted():
    message = "Unknown cov_type: 'foo'. The supported values of cov_type are: 'a'."
    error = UnknownOptionError(message)
    assert str(error) == message
    # The str of a KeyError is the repr of its message, which is the reason to override
    assert str(KeyError(message)) == repr(message)
    assert repr(error) == f"UnknownOptionError({message!r})"
    # What Python prints for an uncaught error and what IPython shows
    printed = traceback.format_exception_only(UnknownOptionError, error)
    expected = f"linearmodels.shared.exceptions.UnknownOptionError: {message}\n"
    assert printed == [expected]
    assert str(UnknownOptionError()) == ""


def test_error_can_be_pickled_and_copied():
    # Models that are fit in other processes raise errors that are pickled
    error = unknown_option_error("cov_type", "foo", TABLE)
    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
        assert type(clone) is UnknownOptionError
        assert str(clone) == str(error)


def test_message():
    error = unknown_option_error("cov_type", "foo", TABLE)
    assert type(error) is UnknownOptionError
    assert str(error) == (
        "Unknown cov_type: 'foo'. The supported values of cov_type are: "
        "'Kernel', 'clustered', 'robust', 'unadjusted'."
    )


def test_names_are_sorted_and_listed_once():
    error = unknown_option_error("kernel", "z", iter(["b", "C", "a", "B"]))
    assert str(error).endswith(
        "The supported values of kernel are: 'B', 'C', 'a', 'b'."
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("", "''"),
        ("robust ", "'robust '"),
        (None, "None"),
        (3, "3"),
        (("a",), "('a',)"),
    ],
)
def test_the_value_is_repeated_unambiguously(value, text):
    error = unknown_option_error("cov_type", value, TABLE)
    assert str(error).startswith(f"Unknown cov_type: {text}. ")


@pytest.mark.parametrize("value", [None, 3, ("robust",), b"robust"])
def test_only_a_string_is_compared_with_the_names(value):
    assert "Did you mean" not in str(unknown_option_error("cov_type", value, TABLE))


@pytest.mark.parametrize(
    ("value", "suggestion"),
    [
        ("cluster", "clustered"),
        ("clustered ", "clustered"),
        ("Clustered", "clustered"),
        ("robsut", "robust"),
        ("kernel", "Kernel"),
        ("foo", None),
        ("", None),
    ],
)
def test_close_name_is_suggested(value, suggestion):
    error = unknown_option_error("cov_type", value, TABLE)
    suggested = f"Did you mean {suggestion!r}? " if suggestion else ""
    expected = (
        f"Unknown cov_type: {value!r}. {suggested}The supported values of cov_type "
        "are: 'Kernel', 'clustered', 'robust', 'unadjusted'."
    )
    assert str(error) == expected


def test_only_the_closest_name_is_suggested():
    error = unknown_option_error("cov_type", "cluster", ["clustered", "clusters"])
    assert str(error).count("Did you mean") == 1


def test_problem_replaces_the_first_sentence():
    problem = "Requested covariance estimator is not supported for the made-up-class."
    error = unknown_option_error("cov_type", "kernel", TABLE, problem=problem)
    assert str(error) == (
        f"{problem} Did you mean 'Kernel'? The supported values of cov_type are: "
        "'Kernel', 'clustered', 'robust', 'unadjusted'."
    )


def test_lookup_option():
    assert lookup_option(TABLE, "clustered", "cov_type") == 2
    with pytest.raises(UnknownOptionError) as info:
        lookup_option(TABLE, "cluster", "cov_type")
    error = info.value
    assert isinstance(error, KeyError)
    assert isinstance(error, ValueError)
    assert str(error) == str(unknown_option_error("cov_type", "cluster", TABLE))
    # The KeyError of the dict is not shown, as it would be in "During handling ..."
    assert error.__suppress_context__
    assert "Did you mean 'clustered'?" in str(error)


def test_lookup_option_does_not_hide_other_errors():
    # A dict raises a TypeError for a value that cannot be a key
    with pytest.raises(UnknownOptionError, match="Unknown cov_type"):
        lookup_option(TABLE, ["robust"], "cov_type")  # type: ignore[arg-type]
