#!/usr/bin/env bash

export BLAS_CONFIG="mkl blas=*=mkl"
if [[ ${OPENBLAS} == "1" ]]; then
  export BLAS_CONFIG="nomkl blas=*=openblas"
fi;
conda create -n linearmodels-test python=${PYTHON} ${BLAS_CONFIG} numpy=${NUMPY} scipy=${SCIPY} pandas=${PANDAS} statsmodels=${STATSMODELS} matplotlib seaborn
source activate linearmodels-test
if [[ ! -z ${XARRAY} ]]; then conda install xarray=${XARRAY}; fi
conda install --yes --quiet sphinx ipython jupyter nbconvert nbformat seaborn
pip install pytest pytest-xdist coverage pytest-cov codecov sphinx doctr nbsphinx guzzle_sphinx_theme -q
pip install --upgrade --no-deps sphinx
