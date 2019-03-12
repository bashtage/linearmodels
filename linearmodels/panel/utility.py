from linearmodels.compat.numpy import isin

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


def dummy_matrix(cats, format='csc', drop='first', drop_all=False):
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
        Exclude either the first or last category
    drop_all : bool
        Flag indicating whether all sets of dummies should exclude one category

    Returns
    -------
    dummies : array-like
        Array, either sparse or dense, of size nobs x ncats containing the
        dummy variable values
    labels : list[str]
        List of dummy variable labels
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
        bits = min([i for i in (8, 16, 32, 64) if i - 1 > np.log2(ncategories + total_dummies)])
        replacements = np.arange(ncategories, dtype='int{:d}'.format(bits))
        cols = replacements[inverse]
        if i == 0 and not drop_all:
            retain = np.arange(nobs)
        elif drop == 'first':
            # remove first
            retain = cols != 0
        else:  # drop == 'last'
            # remove last
            retain = cols != (ncategories - 1)
        rows = rows[retain]
        col_adj = -1 if (drop == 'first' and i > 0) else 0
        cols = cols[retain] + total_dummies + col_adj
        values = np.ones(rows.shape)
        data['values'].append(values)
        data['rows'].append(rows)
        data['cols'].append(cols)
        total_dummies += ncategories - (i > 0)

    if format in ('csc', 'array'):
        fmt = sp.csc_matrix
    elif format == 'csr':
        fmt = sp.csr_matrix
    elif format == 'coo':
        fmt = sp.coo_matrix
    else:
        raise ValueError('Unknown format: {0}'.format(format))
    out = fmt((np.concatenate(data['values']),
               (np.concatenate(data['rows']), np.concatenate(data['cols']))))
    if format == 'array':
        out = out.toarray()

    return out


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


def _drop_singletons(meta, orig_dest):
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
        return isin(cats, retain).ravel()

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
        # Only need unique sort for first, since this is used to reverse later
        if i == 0:
            idx = np.lexsort(temp.T[::-1])
        else:
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
    meta = np.column_stack([node_id, count, offset])

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
            singleton |= isin(cats[:, i], ucats[counts == 1])
        num_singleton = singleton.sum()
        if num_singleton:
            cats = cats[~singleton]
            retain_idx = retain_idx[~singleton]
    retain = np.zeros(nobs, dtype=np.bool)
    retain[retain_idx] = True
    return retain
