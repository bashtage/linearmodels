#!/usr/bin/env bash

CMD="python -m pip install numpy"
if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD==${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD==${PANDAS}"; fi;
CMD="$CMD statsmodels"
if [[ -n ${STATSMODELS} ]]; then CMD="$CMD==${STATSMODELS}"; fi
CMD="$CMD xarray"
if [[ -n ${XARRAY} ]]; then CMD="$CMD==${XARRAY}"; fi
if [[ -n ${XXHASH} ]]; then CMD="$CMD xxhash"; fi
echo $CMD
eval $CMD
