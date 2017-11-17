import glob
import os
import sys

import pytest

try:
    import xarray  # flake8: noqa

    MISSING_XARRAY = False
except ImportError:
    MISSING_XARRAY = True

try:
    import jupyter_client
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:  # pragma: no cover
    pytest.mark.skip(reason='Required packages not available')

kernels = jupyter_client.kernelspec.find_kernel_specs()
kernel_name = 'python%s' % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOKS_USING_XARRAY = ['panel_data-formats.ipynb']
NOTEBOOK_DIR = os.path.abspath(os.path.join(head, '..', '..', 'examples'))

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, '*.ipynb')))
ids = list(map(lambda s: os.path.split(s)[-1].split('.')[0], nbs))
if not nbs:  # pragma: no cover
    pytest.mark.skip(reason='No notebooks found so not tests run')


@pytest.fixture(params=nbs, ids=ids)
def notebook(request):
    return request.param


@pytest.mark.slow
def test_notebook(notebook):
    nb_name = os.path.split(notebook)[-1]
    if MISSING_XARRAY and nb_name in NOTEBOOKS_USING_XARRAY:
        pytest.skip('xarray is required to test {0}'.format(notebook))

    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(allow_errors=False,
                             timeout=120,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
