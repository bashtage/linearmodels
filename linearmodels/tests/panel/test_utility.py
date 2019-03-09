import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest
import scipy.sparse.coo
import scipy.sparse.csc
import scipy.sparse.csr

from linearmodels.panel.utility import dummy_matrix

formats = {'csc': scipy.sparse.csc.csc_matrix, 'csr': scipy.sparse.csr.csr_matrix,
           'coo': scipy.sparse.coo.coo_matrix, 'array': np.ndarray}


@pytest.fixture('module', params=formats)
def format(request):
    return request.param, formats[request.param]


def test_dummy_format(format):
    code, expected_type = format
    cats = np.zeros([15, 2], dtype=np.int8)
    cats[5:, 0] = 1
    cats[10:, 0] = 2
    cats[:, 1] = np.arange(15) % 5
    out = dummy_matrix(cats, format=code)
    assert isinstance(out, expected_type)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 3, 3, 3, 3], dtype=np.int32)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)


def test_dummy_last():
    cats = np.zeros([15, 2], dtype=np.int8)
    cats[5:, 0] = 1
    cats[10:, 0] = 2
    cats[:, 1] = np.arange(15) % 5
    cats[-1, 1] = 0
    out = dummy_matrix(cats, drop='last')
    assert isinstance(out, scipy.sparse.csc.csc_matrix)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 4, 3, 3, 3], dtype=np.int32)
    assert out.shape == (15, 3 + 5 - 1)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)


def test_invalid_format():
    cats = np.zeros([10, 1], dtype=np.int8)
    cats[5:, 0] = 1
    with pytest.raises(ValueError):
        dummy_matrix(cats, format='unknown')


def test_dummy_pandas():
    c1 = pd.Series(pd.Categorical(['a'] * 5 + ['b'] * 5 + ['c'] * 5))
    c2 = pd.Series(pd.Categorical(['A', 'B', 'C', 'D', 'E'] * 3))
    cats = pd.concat([c1, c2], 1)
    out = dummy_matrix(cats, drop='last')
    assert isinstance(out, scipy.sparse.csc.csc_matrix)
    assert out.shape == (15, 3 + 5 - 1)
    expected = np.array([5, 5, 5, 3, 3, 3, 3], dtype=np.int32)
    assert_array_equal(np.squeeze(np.asarray(out.sum(0), dtype=np.int32)), expected)
