#!/usr/bin/env bash
if [[ ! -z ${NUMPY} ]]; then pip install numpy==${NUMPY}; else pip install numpy; fi
pip install scipy pandas matplotlib matplotlib seaborn cython statsmodels
if [[ ! -z ${XARRAY} ]]; then pip install xarray==${XARRAY}; fi
if [[ ! -z ${PIP_PACKAGES} ]]; then pip install ${PIP_PACKAGES}; fi
