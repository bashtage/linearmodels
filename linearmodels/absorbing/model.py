from linearmodels.compat.pandas import is_categorical

from typing import Iterable, List, Union

from numpy import column_stack, dtype, int8, int16, int32, int64, zeros
from numpy.linalg import lstsq
from pandas import Categorical, DataFrame, Series
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr

from linearmodels.iv.data import IVData, iv_data_types
from linearmodels.panel.utility import dummy_matrix
from linearmodels.typing.data import ArrayLike, OptionalArrayLike

SCALAR_DTYPES = {'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64}


def category_interaction(cats: ArrayLike):
    sizes = []
    for c in cats:
        if not is_categorical(cats[c]):
            raise TypeError('cats must contain only categorical variables')
        col = cats[c]
        max_code = col.cat.codes.max()
        size = 1
        while max_code >= 2 ** size:
            size += 1
        sizes.append(size)
    nobs = cats.shape[0]
    total_size = sum(sizes)
    if total_size >= 63:
        raise ValueError('There are too many cats with too many states to use this method.')
    dtype_size = min(filter(lambda v: total_size < (v - 1), (8, 16, 32, 64)))
    dtype_str = 'int{0:d}'.format(dtype_size)
    dtype_val = dtype(dtype_str)
    codes = zeros(nobs, dtype=dtype_val)
    cum_size = 0
    for i, col in enumerate(cats):
        codes += (cats[col].cat.codes.astype(dtype_val) << SCALAR_DTYPES[dtype_str](cum_size))
        cum_size += sizes[i]
    return Series(Categorical(codes), index=cats.index)


def categorical_interaction(cat, cont):
    codes = cat.cat.codes
    dummies = dummy_matrix(codes[:, None])[0]
    dummies.data = cont
    return dummies


def construct_interactions(interaction: IVData):
    intact_df = interaction.pandas
    cat_vars = [col for col in intact_df if is_categorical(intact_df[col])]
    cont_vars = [col for col in intact_df if col not in cat_vars]
    if not cat_vars:
        return csc_matrix(intact_df.to_numpy())
    cat_interact = category_interaction(intact_df[cat_vars])
    out = []
    for col in cont_vars:
        out.append(categorical_interaction(cat_interact, intact_df[col]))
    return sp.hstack(out)


class AbsorbingLS(object):
    def __init__(self, dependent: ArrayLike, exog: OptionalArrayLike = None,
                 absorb: OptionalArrayLike = None,
                 interactions: Union[None, DataFrame, Iterable[DataFrame]] = None):

        self._dependent = IVData(dependent, 'dependent')
        self._exog = IVData(exog, 'exog')
        self._absorb = IVData(absorb, 'absorb', nobs=self._dependent.shape[0],
                              convert_dummies=False)
        self._interactions = interactions
        self._interaction_list = []  # type: List[IVData]
        self._prepare_interactions()

    def _prepare_interactions(self):
        if self._interactions is None:
            return
        elif isinstance(self._interactions, iv_data_types):
            self._interaction_list = [IVData(self._interactions, convert_dummies=False)]
        else:
            for iteract in self._interactions:
                self._interaction_list.append(IVData(iteract, convert_dummies=False))

    def fit(self):
        cats = []
        absorb = self._absorb.pandas
        for c in absorb:
            if is_categorical(absorb[c]):
                cats.append(c)
        if cats:
            x, _ = dummy_matrix(absorb[cats])
            cont = absorb[[c for c in absorb if c not in cats]]
            if cont.shape[1] > 0:
                cont = csc_matrix(cont.to_numpy())
                x = sp.hstack([cont, x])
        else:
            x = self._absorb.ndarray

        if self._interaction_list:
            interacted = [construct_interactions(inter) for inter in self._interaction_list]
            x = sp.hstack([x] + interacted)

        y = self._dependent.ndarray
        y_mean = lsmr(x, y, atol=1e-8, btol=1e-8, show=False)[0]
        y_resid = y - (x.dot(csc_matrix(y_mean[:, None]))).A
        if self._exog is None:
            return y_resid

        x_r = []
        exog = self._exog.pandas
        for col in exog:
            ex = exog[col].to_numpy()[:, None]
            x_mean = lsmr(x, ex, atol=1e-8, btol=1e-8, show=True)[0]
            x_r.append(ex - (x.dot(csc_matrix(x_mean[:, None]))).A)
        x_resids = column_stack(x_r)
        # TODO: check for x absorbed
        b = lstsq(x_resids, y_resid)[0]
        return b


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    n = 1000000
    rs = np.random.RandomState(0)
    cats = pd.concat([pd.Series(pd.Categorical(rs.randint(n // 10, size=n)))], 1)
    x = pd.DataFrame(rs.standard_normal((n, 1)))
    y = pd.DataFrame(rs.standard_normal((n, 1)))
    ia = pd.concat([pd.Series(rs.standard_normal((n))),
                    pd.Series(pd.Categorical(rs.randint(n // 10, size=n)))], 1)
    mod = AbsorbingLS(y, x, cats, ia)
    print(mod.fit())
