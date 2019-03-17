#!/usr/bin/env bash

export BLAS_CONFIG="mkl blas=*=mkl"
if [[ ${OPENBLAS} == "1" ]]; then
  export BLAS_CONFIG="nomkl blas=*=openblas"
fi;
conda update --all -q
conda create -n linearmodels-test python=${PYTHON} ${BLAS_CONFIG} numpy=${NUMPY} scipy=${SCIPY} pandas=${PANDAS} statsmodels=${STATSMODELS} matplotlib seaborn cython
source activate linearmodels-test
if [[ ! -z ${XARRAY} ]]; then conda install xarray=${XARRAY}; fi
