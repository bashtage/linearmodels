import packaging.version
import scipy

SP_VERSION = packaging.version.parse(scipy.__version__)
SP_LT_2 = SP_VERSION < packaging.version.parse("1.99.99")

if not SP_LT_2:
    from scipy.sparse import coo_array, csc_array, csr_array, diags_array, lil_array
else:
    import scipy.sparse

    csr_array = scipy.sparse.csr_matrix
    csc_array = scipy.sparse.csc_matrix
    coo_array = scipy.sparse.coo_matrix
    diags_array = scipy.sparse.diags
    lil_array = scipy.sparse.lil_matrix

__all__ = ["coo_array", "csc_array", "csr_array", "diags_array", "lil_array"]
