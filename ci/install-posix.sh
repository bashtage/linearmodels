#!/usr/bin/env bash

CMD="python -m pip install setuptools_scm[toml]>=7 formulaic numpy"
if [[ -n ${NUMPY} ]]; then CMD="$CMD~=${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD~=${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD~=${PANDAS}"; fi;
CMD="$CMD statsmodels"
if [[ -n ${STATSMODELS} ]]; then CMD="$CMD~=${STATSMODELS}"; fi
if [[ -n ${XARRAY} ]]; then CMD="$CMD xarray~=${XARRAY}"; fi
if [[ -n ${FORMULAIC} ]]; then CMD="$CMD formulaic~=${FORMULAIC}"; fi
if [[ -n ${XXHASH} ]]; then CMD="$CMD xxhash"; fi
echo "$CMD"
eval "$CMD"

if [ "${PIP_PRE}" = true ]; then
  python -m pip uninstall -y numpy pandas scipy matplotlib statsmodels
  python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy pandas scipy matplotlib --upgrade --use-deprecated=legacy-resolver
  python -m pip install git+https://github.com/statsmodels/statsmodels.git --upgrade --no-build-isolation -v
fi