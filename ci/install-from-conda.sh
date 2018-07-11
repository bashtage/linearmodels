#!/usr/bin/env bash
conda create -n linearmodels-test python=${PYTHON} numpy=${NUMPY} scipy=${SCIPY} pandas=${PANDAS} statsmodels=${STATSMODELS} matplotlib seaborn
source activate linearmodels-test
if [[ ! -z ${XARRAY} ]]; then conda install xarray=${XARRAY}; fi
conda install --yes --quiet sphinx ipython jupyter nbconvert nbformat seaborn
pip install pytest pytest-xdist coverage pytest-cov codecov sphinx doctr nbsphinx guzzle_sphinx_theme -q
pip install --upgrade --no-deps sphinx
