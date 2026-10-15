"""
Sparse array classes that work in all supported versions of SciPy.
"""

import packaging.version
import scipy

SP_VERSION = packaging.version.parse(scipy.__version__)
SP_LT_2 = SP_VERSION < packaging.version.parse("1.99.99")

if not SP_LT_2:
    from scipy.sparse import coo_array, csc_array, csr_array, diags_array, lil_array
else:
    import scipy.sparse

    # The matrix classes are not the array classes that the stubs have for SciPy 2
    csr_array = scipy.sparse.csr_matrix  # type: ignore[misc,assignment]
    csc_array = scipy.sparse.csc_matrix  # type: ignore[misc,assignment]
    coo_array = scipy.sparse.coo_matrix  # type: ignore[misc,assignment]
    diags_array = scipy.sparse.diags  # type: ignore[assignment]
    lil_array = scipy.sparse.lil_matrix  # type: ignore[misc,assignment]

__all__ = ["coo_array", "csc_array", "csr_array", "diags_array", "lil_array"]
