#!/usr/bin/env bash
conda update --all -q
conda create -n linearmodels-test python
source activate linearmodels-test
pip install numpy scipy pandas matplotlib matplotlib seaborn cython
pip install statsmodels
if [[ ! -z ${XARRAY} ]]; then pip install xarray==${XARRAY}; fi
