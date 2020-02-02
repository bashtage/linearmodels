#!/usr/bin/env bash

# Source to configure pip prerelease CI builds
# Wheelhouse for various packages missing from pypi.
EXTRA_WHEELS="https://5cf40426d9f06eb7461d-6fe47d9331aba7cd62fc36c7196769e4.ssl.cf2.rackcdn.com"
# Wheelhouse for daily builds of some packages.
PRE_WHEELS="https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
PIP_FLAGS="--pre --find-links $EXTRA_WHEELS --find-links $PRE_WHEELS"
pip install ${PIP_FLAGS} scipy pandas matplotlib matplotlib seaborn cython statsmodels --upgrade
