from __future__ import annotations

import logging
import os

import pandas as pd
import pytest

logger = logging.getLogger(__name__)

cow = bool(os.environ.get("LM_TEST_COPY_ON_WRITE", False))
if cow:
    try:
        pd.options.mode.copy_on_write = cow
    except AttributeError:
        cow = False
if cow:
    logger.critical("Copy on Write testing enabled")


def pytest_configure(config):
    # Minimal config to simplify running tests from lm.test()
    config.addinivalue_line("markers", "example: mark a test as an example")
    config.addinivalue_line("markers", "slow: mark a test as slow")
    config.addinivalue_line(
        "markers", "smoke: mark a test as a coding error (smoke) test"
    )
    config.addinivalue_line(
        "filterwarnings", "ignore:Method .ptp is deprecated:FutureWarning"
    )


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true", help="skip slow tests")
    parser.addoption("--only-slow", action="store_true", help="run only slow tests")
    parser.addoption("--skip-smoke", action="store_true", help="skip smoke tests")
    parser.addoption("--only-smoke", action="store_true", help="run only smoke tests")
    parser.addoption("--skip-examples", action="store_true", help="skip examples tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and item.config.getoption("--skip-slow"):
        pytest.skip("skipping due to --skip-slow")

    if "slow" not in item.keywords and item.config.getoption("--only-slow"):
        pytest.skip("skipping due to --only-slow")

    if "smoke" in item.keywords and item.config.getoption("--skip-smoke"):
        pytest.skip("skipping due to --skip-smoke")

    if "smoke" not in item.keywords and item.config.getoption("--only-smoke"):
        pytest.skip("skipping due to --only-smoke")

    if "example" in item.keywords and item.config.getoption("--skip-examples"):
        pytest.skip("skipping due to --skip-examples")
