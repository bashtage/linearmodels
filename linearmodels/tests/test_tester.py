import pytest

import linearmodels


def test_runner():
    status = linearmodels.test(
        location="tests/shared/test_typed_getters.py", exit=False
    )
    assert status == 0


def test_runner_exception():
    with pytest.raises(RuntimeError):
        linearmodels.test(location="tests/shared/unknown_test_file.py")


def test_extra_args():
    status = linearmodels.test(
        "--tb=short",
        append=False,
        location="tests/shared/test_typed_getters.py",
        exit=False,
    )

    assert status == 0

    status = linearmodels.test(
        ["-r", "a"],
        append=True,
        location="tests/shared/test_typed_getters.py",
        exit=False,
    )

    assert status == 0
