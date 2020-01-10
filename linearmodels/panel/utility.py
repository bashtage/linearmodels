from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from linearmodels.panel._utility import _drop_singletons

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


class AbsorbingEffectError(Exception):
    pass


absorbing_error_msg = """
The model cannot be estimated. The included effects have fully absorbed
one or more of the variables. This occurs when one or more of the dependent
variable is perfectly explained using the effects included in the model.

The following variables or variable combinations have been fully absorbed
or have become perfectly collinear after effects are removed:

{absorbed_variables}

Set drop_absorbed=True to automatically drop absorbed variables.
"""


class AbsorbingEffectWarning(Warning):
    pass


absorbing_warn_msg = """
Variables have been fully absorbed and have removed from the regression:

{absorbed_variables}
"""


def preconditioner(d, *, copy=False):
    """
    Parameters
    ----------
    d : array_like
        Array to precondition
    copy : bool
        Flag indicating whether the operation should be in-place, if possible.
        If True, when a new array will always be returned.

    Returns
    -------
    d : array_like
        Array with same type as input array. If copy is False, and d is
        an ndarray or a csc_matrix, then the operation is inplace
    cond : ndarray
        Array of conditioning numbers defined as the square root of the column
        2-norms (nvar,)
    """
    # Dense path
    if not sp.issparse(d):
        klass = None if type(d) == np.ndarray else d.__class__
        d_id = id(d)
        d = np.asarray(d)
        if id(d) == d_id or copy:
            d = d.copy()
        cond = np.sqrt((d ** 2).sum(0))
        d /= cond
        if klass is not None:
            d = d.view(klass)
        return d, cond

    klass = None
    if not isinstance(d, sp.csc_matrix):
        klass = d.__class__
        d = sp.csc_matrix(d)
    elif copy:
        d = d.copy()

    cond = np.sqrt(d.multiply(d).sum(0)).A1
    locs = np.zeros_like(d.indices)
    locs[d.indptr[1:-1]] = 1
    locs = np.cumsum(locs)
    d.data /= np.take(cond, locs)
    if klass is not None:
        d = klass(d)

    return d, cond


def dummy_matrix(
    cats, *, format="csc", drop="first", drop_all=False, precondition=True
):
    """
    Parameters
    ----------
    cats: {DataFrame, ndarray}
        Array containing the category codes of pandas categoricals
        (nobs, ncats)
    format: {'csc', 'csr', 'coo', 'array'}
        Output format. Default is csc (csc_matrix). Supported output
        formats are:

        * 'csc' - sparse matrix in compressed column form
        * 'csr' - sparse matrix in compressed row form
        * 'coo' - sparse matrix in coordinate form
        * 'array' - dense numpy ndarray

    drop: {'first', 'last'}
        Exclude either the first or last category. This only applies when
        cats contains more than one column, unless `drop_all` is True.
    drop_all : bool
        Flag indicating whether all sets of dummies should exclude one category
    precondition : bool
        Flag indicating whether the columns of the dummy matrix should be
        preconditioned to have unit 2-norm.

    Returns
    -------
    dummies : array_like
        Array, either sparse or dense, of size nobs x ncats containing the
        dummy variable values
    cond : ndarray
        Conditioning number of each column
    """
    if isinstance(cats, pd.DataFrame):
        codes = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    else:
        codes = cats

    data = defaultdict(list)
    total_dummies = 0
    nobs, ncats = codes.shape
    for i in range(ncats):
        rows = np.arange(nobs)
        ucats, inverse = np.unique(codes[:, i], return_inverse=True)
        ncategories = len(ucats)
        bits = min(
            [i for i in (8, 16, 32, 64) if i - 1 > np.log2(ncategories + total_dummies)]
        )
        replacements = np.arange(ncategories, dtype="int{:d}".format(bits))
        cols = replacements[inverse]
        if i == 0 and not drop_all:
            retain = np.arange(nobs)
        elif drop == "first":
            # remove first
            retain = cols != 0
        else:  # drop == 'last'
            # remove last
            retain = cols != (ncategories - 1)
        rows = rows[retain]
        col_adj = -1 if (drop == "first" and i > 0) else 0
        cols = cols[retain] + total_dummies + col_adj
        values = np.ones(rows.shape)
        data["values"].append(values)
        data["rows"].append(rows)
        data["cols"].append(cols)
        total_dummies += ncategories - (i > 0)

    if format in ("csc", "array"):
        fmt = sp.csc_matrix
    elif format == "csr":
        fmt = sp.csr_matrix
    elif format == "coo":
        fmt = sp.coo_matrix
    else:
        raise ValueError("Unknown format: {0}".format(format))
    out = fmt(
        (
            np.concatenate(data["values"]),
            (np.concatenate(data["rows"]), np.concatenate(data["cols"])),
        )
    )
    if format == "array":
        out = out.toarray()

    if precondition:
        out, cond = preconditioner(out, copy=False)
    else:
        cond = np.ones(out.shape[1])

    return out, cond


def _remove_node(node, meta, orig_dest):
    """
    Parameters
    ----------
    node : int
        ID of the node to remove
    meta : ndarray
        Array with rows containing node, count, and address where
        address is used to find the first occurrence in orig_desk
    orig_dest : ndarray
        Array with rows containing origin and destination nodes

    Returns
    -------
    next_node : int
        ID of the next node in the branch
    next_count : int
        Count of the next node in the branch
    Notes
    -----
    Node has 1 link, so:
        1. Remove the forward link
        2. Remove the backward link
        3. Decrement node's count
        4. Decrement next_node's count
    """
    # 3. Decrement
    meta[node, 1] -= 1
    # 1. Remove forewrd link
    next_offset = meta[node, 2]
    orig, next_node = orig_dest[next_offset]
    while next_node == -1:
        # Increment since this could have been previously deleted
        next_offset += 1
        next_orig, next_node = orig_dest[next_offset]
        assert orig == next_orig
    # 4. Remove next_node's link
    orig_dest[next_offset, 1] = -1

    # 2. Remove the backward link
    # Set reverse to -1
    reverse_offset = meta[next_node, 2]
    reverse_node = orig_dest[reverse_offset, 1]
    while reverse_node != orig:
        reverse_offset += 1
        reverse_node = orig_dest[reverse_offset, 1]
    orig_dest[reverse_offset, 1] = -1

    # Step forward
    meta[next_node, 1] -= 1
    next_count = meta[next_node, 1]
    return next_node, next_count


def _py_drop_singletons(meta, orig_dest):
    """
    Loop through the nodes and recursively drop singleton chains

    Parameters
    ----------
    meta : ndarray
        Array with rows containing node, count, and address where
        address is used to find the first occurrence in orig_desk
    orig_dest : ndarray
        Array with rows containing origin and destination nodes
    """
    for i in range(meta.shape[0]):
        if meta[i, 1] == 1:
            next_node = i
            next_count = 1
            while next_count == 1:
                # Follow singleton chains
                next_node, next_count = _remove_node(next_node, meta, orig_dest)


if not HAS_CYTHON:
    _drop_singletons = _py_drop_singletons  # noqa: F811


def in_2core_graph(cats):
    """
    Parameters
    ----------
    cats: {DataFrame, ndarray}
        Array containing the category codes of pandas categoricals
        (nobs, ncats)

    Returns
    -------
    retain : ndarray
        Boolean array that marks non-singleton entries as True
    """
    if isinstance(cats, pd.DataFrame):
        cats = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    if cats.shape[1] == 1:
        # Fast, simple path
        ucats, counts = np.unique(cats, return_counts=True)
        retain = ucats[counts > 1]
        return np.isin(cats, retain).ravel()

    nobs, ncats = cats.shape
    zero_cats = []
    # Switch to 0 based indexing
    for col in range(ncats):
        u, inv = np.unique(cats[:, col], return_inverse=True)
        zero_cats.append(np.arange(u.shape[0])[inv])
    zero_cats = np.column_stack(zero_cats)
    # 2 tables
    # a.
    #    origin_id, dest_id
    max_cat = zero_cats.max(0)
    shift = np.r_[0, max_cat[:-1] + 1]
    zero_cats += shift
    orig_dest = []
    for i in range(ncats):
        col_order = list(range(ncats))
        col_order.remove(i)
        col_order = [i] + col_order
        temp = zero_cats[:, col_order]
        idx = np.argsort(temp[:, 0])
        orig_dest.append(temp[idx])
        if i == 0:
            inverter = np.empty_like(zero_cats[:, 0])
            inverter[idx] = np.arange(nobs)
    orig_dest = np.concatenate(orig_dest, 0)
    # b.
    #    node_id, count, offset
    node_id, count = np.unique(orig_dest[:, 0], return_counts=True)
    offset = np.r_[0, np.where(np.diff(orig_dest[:, 0]) != 0)[0] + 1]

    def min_dtype(*args):
        bits = max([np.log2(max(arg.max(), 1)) for arg in args])
        return "int{0}".format(min([i for i in (8, 16, 32, 64) if bits < (i - 1)]))

    dtype = min_dtype(offset, node_id, count, orig_dest)
    meta = np.column_stack(
        [node_id.astype(dtype), count.astype(dtype), offset.astype(dtype)]
    )
    orig_dest = orig_dest.astype(dtype)

    singletons = np.any(meta[:, 1] == 1)
    while singletons:
        _drop_singletons(meta, orig_dest)
        singletons = np.any(meta[:, 1] == 1)

    sorted_cats = orig_dest[:nobs]
    unsorted_cats = sorted_cats[inverter]
    retain = unsorted_cats[:, 1] > 0

    return retain


def in_2core_graph_slow(cats):
    """
    Parameters
    ----------
    cats: {DataFrame, ndarray}
        Array containing the category codes of pandas categoricals
        (nobs, ncats)

    Returns
    -------
    retain : ndarray
        Boolean array that marks non-singleton entries as True

    Notes
    -----
    This is a reference implementation that can be very slow to remove
    all singleton nodes in some graphs.
    """
    if isinstance(cats, pd.DataFrame):
        cats = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    if cats.shape[1] == 1:
        return in_2core_graph(cats)
    nobs, ncats = cats.shape
    retain_idx = np.arange(cats.shape[0])
    num_singleton = 1
    while num_singleton > 0 and cats.shape[0] > 0:
        singleton = np.zeros(cats.shape[0], dtype=np.bool)
        for i in range(ncats):
            ucats, counts = np.unique(cats[:, i], return_counts=True)
            singleton |= np.isin(cats[:, i], ucats[counts == 1])
        num_singleton = singleton.sum()
        if num_singleton:
            cats = cats[~singleton]
            retain_idx = retain_idx[~singleton]
    retain = np.zeros(nobs, dtype=np.bool)
    retain[retain_idx] = True
    return retain


def check_absorbed(x: np.ndarray, variables: List[str]):
    """
    Check a regressor matrix for variables absorbed

    Parameters
    ----------
    x : ndarray
        Regressor matrix to check
    variables : List[str]
        List of variable names
    """
    rank = np.linalg.matrix_rank(x)
    if rank < x.shape[1]:
        xpx = x.T @ x
        vals, vecs = np.linalg.eigh(xpx)
        nabsorbed = x.shape[1] - rank
        tol = np.sort(vals)[nabsorbed - 1]
        absorbed = vals <= tol
        absorbed_vecs = vecs[:, absorbed]
        rows = []
        for i in range(nabsorbed):
            abs_vec = np.abs(absorbed_vecs[:, i])
            tol = abs_vec.max() * np.finfo(np.float64).eps * abs_vec.shape[0]
            vars_idx = np.where(np.abs(absorbed_vecs[:, i]) > tol)[0]
            rows.append(" " * 10 + ", ".join((variables[vi] for vi in vars_idx)))
        absorbed_variables = "\n".join(rows)
        msg = absorbing_error_msg.format(absorbed_variables=absorbed_variables)
        raise AbsorbingEffectError(msg)


def not_absorbed(x: np.ndarray):
    """
    Construct a list of the indices of regressors that are not absorbed

    Parameters
    ----------
    x : ndarray
        Regressor matrix to check

    Returns
    -------
    retain : list[int]
        List of columns to retain
    """
    if np.linalg.matrix_rank(x) == x.shape[1]:
        return list(range(x.shape[1]))
    xpx = x.T @ x
    vals, vecs = np.linalg.eigh(xpx)
    tol = vals.max() * x.shape[1] * np.finfo(np.float64).eps
    absorbed = vals < tol
    nabsorbed = absorbed.sum()
    q, r = np.linalg.qr(x)
    threshold = np.sort(np.abs(np.diag(r)))[nabsorbed]
    drop = np.where(np.abs(np.diag(r)) < threshold)[0]
    retain = set(range(x.shape[1])).difference(drop)
    return sorted(retain)
