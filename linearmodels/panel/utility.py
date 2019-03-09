from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


def dummy_matrix(cats, format='csc', drop='first', drop_all=False):
    """
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
        bits = min([i for i in (8, 16, 32, 64) if i - 1 > np.log2(ncategories)])
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
