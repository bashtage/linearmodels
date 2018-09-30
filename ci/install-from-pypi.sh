#!/usr/bin/env bash
conda create -n linearmodels-test python
source activate linearmodels-test
pip install numpy scipy pandas matplotlib matplotlib seaborn
pip install statsmodels
if [[ ! -z ${XARRAY} ]]; then pip install xarray; fi
pip install ipython jupyter nbconvert nbformat -q
pip install pytest pytest-xdist coverage pytest-cov codecov sphinx doctr nbsphinx guzzle_sphinx_theme cached_property -q
