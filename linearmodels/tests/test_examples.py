import glob
import os
import sys

import pytest

try:
    import xarray  # noqa: F401

    MISSING_XARRAY = False
except ImportError:
    MISSING_XARRAY = True

try:
    import jupyter_client
    import matplotlib  # noqa: F401
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat
    import seaborn  # noqa: F401

    kernels = jupyter_client.kernelspec.find_kernel_specs()
    SKIP = False
except ImportError:  # pragma: no cover
    SKIP = True
    pytestmark = pytest.mark.skip(reason="Required packages not available")

kernel_name = "python%s" % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOKS_USING_XARRAY = ["panel_data-formats.ipynb"]
NOTEBOOK_DIR = os.path.abspath(os.path.join(head, "..", "..", "examples"))

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))
ids = [os.path.split(nb)[-1].split(".")[0] for nb in nbs]
if not nbs:  # pragma: no cover
    pytest.mark.skip(reason="No notebooks found so not tests run")


@pytest.fixture(params=nbs, ids=ids)
def notebook(request):
    return request.param


@pytest.mark.slow
@pytest.mark.skipif(SKIP, reason="Required packages not available")
def test_notebook(notebook):
    nb_name = os.path.split(notebook)[-1]
    if MISSING_XARRAY and nb_name in NOTEBOOKS_USING_XARRAY:
        pytest.skip("xarray is required to test {0}".format(notebook))

    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(allow_errors=False, timeout=120, kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})
