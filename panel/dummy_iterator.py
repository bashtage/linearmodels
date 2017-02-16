import numpy as np
from scipy.sparse import csc_matrix


class DummyVariableIterator(object):
    """
    Iterator object that produces dummy variables
    """

    def __init__(self, n, t, groups, drop=False, max_size=10, sparse=False):
        self.n = n
        self.t = t
        self.groups = groups
        self.max_size = max_size
        self.sparse = sparse
        self.drop = drop
        self._array_cols = max(1, int(self.max_size * 2.0 ** 20 / 8.0 / (n * t)))
        groups = groups.astype(np.int64)
        self._rows = int(n * t)
        self._index = 0
        ugroups = np.unique(groups)
        if np.min(ugroups) != 0 or np.any(np.diff(ugroups) != 1):
            raise ValueError('groups must contain elements in {0,1,...,max}')
        if len(groups) != self._rows:
            raise ValueError('groups must have n * t elements')
        self._group_index = np.argsort(groups)
        ordered = self._ordered_groups = groups[self._group_index]
        locs = np.argwhere(np.diff(ordered) != 0)
        self._ends = np.concatenate([[[0]], locs + 1, [[len(ordered)]]])
        self._ends = self._ends.ravel()

    def __iter__(self):
        self._iter_count = 0
        self._remaining_cols = self.groups.max() + 1
        return self

    def __next__(self):
        if self._remaining_cols <= 0:
            raise StopIteration

        cols = min(self._remaining_cols, self._array_cols)
        self._remaining_cols -= cols
        ends = self._ends
        rows = self._group_index[ends[self._index]:ends[self._index + cols]]
        group_ids = self._ordered_groups[ends[self._index]:ends[self._index + cols]]
        columns = group_ids - self._index
        self._index += cols
        if not self.sparse:
            out = np.zeros((self._rows, cols))
            out[rows, columns] = 1
        else:
            values = np.ones_like(columns)
            locs = (rows, columns)
            shape = (self._rows, cols)
            out = csc_matrix((values, locs), shape=shape, dtype=np.float64)

        if self.drop and np.any(group_ids == 0):
            out = out[:, 1:]
            # Ensure never return empty column
            if out.shape[1] == 0:
                return self.__next__()

        return out
