#!/usr/bin/env bash

export BLAS_CONFIG="mkl blas=*=mkl"
if [[ ${OPENBLAS} == "1" ]]; then
  export BLAS_CONFIG="nomkl blas=*=openblas"
fi;
conda update --all -q
echo conda create -n linearmodels-test python=${PYTHON} ${BLAS_CONFIG} numpy=${NUMPY} scipy=${SCIPY} pandas=${PANDAS} statsmodels=${STATSMODELS} matplotlib
conda create -n linearmodels-test python=${PYTHON} ${BLAS_CONFIG} numpy=${NUMPY} scipy=${SCIPY} pandas=${PANDAS} statsmodels=${STATSMODELS} matplotlib
source activate linearmodels-test
if [[ ! -z ${MKL} ]]; then conda install mkl=${MKL} intel-openmp=${MKL} --no-update-deps; fi
if [[ ! -z ${XARRAY} ]]; then conda install xarray=${XARRAY}; fi
