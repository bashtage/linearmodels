from linearmodels.compat.numpy import lstsq
from linearmodels.compat.pandas import is_categorical, to_numpy

from typing import Iterable, List, Union

from numpy import (column_stack, dtype, empty, int8, int16, int32, int64,
                   ndarray, zeros)
from numpy.linalg import matrix_rank
from pandas import Categorical, DataFrame, Series
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr

from linearmodels.iv.data import IVData
from linearmodels.panel.model import AbsorbingEffectError, absorbing_error_msg
from linearmodels.panel.utility import dummy_matrix, preconditioner
from linearmodels.typing import AnyPandas
from linearmodels.typing.data import ArrayLike, OptionalArrayLike

SCALAR_DTYPES = {'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64}


def lsmr_annihilate(x: csc_matrix, y: ndarray, **lsmr_options) -> ndarray:
    r"""
    Removes projection of x on y from y

    Parameters
    ----------
    x : csc_matrix
        Sparse array of regressors
    y : ndarray
        Array with shape (nobs, nvar)
    lsmr_options: dict
        Dictionary of options to pass to scipy.sparse.linalg.lsmr

    Returns
    -------
    resids : ndarray
        Returns the residuals from regressing y on x, (nobs, nvar)

    Notes
    -----
    Residuals are estiamted column-by-column as

    .. math::

        \hat{\epsilon}_{j} = y_{j} - x^\prime \hat{\beta}

    where :math:`\hat{\beta}` is computed using lsmr.
    """
    default_opts = dict(atol=1e-8, btol=1e-8, show=False)
    default_opts.update(lsmr_options)
    resids = []
    for i in range(y.shape[1]):
        beta = lsmr(x, y, **default_opts)[0]
        resid = y - (x.dot(csc_matrix(beta[:, None]))).A
        resids.append(resid)

    return column_stack(resids)


def category_product(cats: AnyPandas) -> Series:
    """
    Construct category from all combination of input categories

    Parameters
    ----------
    cats : {Series, DataFrame}
        DataFrame containing categorical variables.  If cats is a Series, cats
        is returned unmodified.

    Returns
    -------
    cp : Series
        Categorical series containing the cartesian product of the categories
        in cats
    """
    if isinstance(cats, Series):
        return cats

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


def category_interaction(cat: Series) -> csc_matrix:
    """
    Parameters
    ----------
    cat : Series
        Categorical series to convert to dummy variables

    Returns
    -------
    dummies : csc_matrix
        Sparse matrix of dummies with unit column norm
    """
    codes = category_product(cat).cat.codes
    return dummy_matrix(codes[:, None])[0]


def category_continuous_interaction(cat: AnyPandas, cont: AnyPandas) -> csc_matrix:
    """
    Parameters
    ----------
    cat : Series
        Categorical series to convert to dummy variables
    cont : {Series, DataFrame}
        Continuous variable values to use in the dummy interaction

    Returns
    -------
    interact : csc_matrix
        Sparse matrix of dummy interactions with unit column norm
    """
    codes = category_product(cat).cat.codes
    dummies = dummy_matrix(codes[:, None], precondition=False)[0]
    dummies.data[:] = to_numpy(cont).flat
    return preconditioner(dummies)[0]


class Interaction(object):
    """
    Class that simplifies specifying interactions

    Parameters
    ----------
    cats : {ndarray, Series, DataFrame, DataArray}, optional
        Variables to treat as categoricals. Best format is a Categorical
        Series or DataFrame containing Categorical Series. Other formats
        are converted to Categorical Series, column-by-column. cats has
        shape (nobs, ncat).
    cont : {ndarray, Series, DataFrame, DataArray}, optional
        Variables to treat as continuous, (nobs, ncont).

    Notes
    -----
    For each variable in `cont`, computes the interaction of the variable
    and the cartesian product of the categories.

    Examples
    --------
    >>> import numpy as np
    >>> from linearmodels.absorbing.model import Interaction
    >>> rs = np.random.RandomState(0)
    >>> n = 100000
    >>> cats = rs.randint(2,size=n)  # binary dummy
    >>> cont = rs.standard_normal((n, 3))
    >>> interact = Interaction(cats, cont)
    >>> interact.sparse.shape  # Get the shape of the dummy matrix
    (100000, 6)

    >>> rs = np.random.RandomState(0)
    >>> import pandas as pd
    >>> cats = pd.concat([pd.Series(pd.Categorical(rs.randint(5,size=n)))
    ...                  for _ in range(4)],1)
    >>> cats.describe()
                     0       1       2       3
    count   100000  100000  100000  100000
    unique       5       5       5       5
    top          3       3       0       4
    freq     20251   20195   20331   20158

    >>> interact = Interaction(cats, cont)
    >>> interact.sparse.shape # Cart product of all cats, 5**4, times ncont, 3
    (100000, 1875)
    """
    _iv_data = IVData(None, 'none', 1)

    def __init__(self, cat: OptionalArrayLike = None, cont: OptionalArrayLike = None,
                 nobs: int = None):
        self._cat = cat
        self._cont = cont
        self._cat_data = self._iv_data
        self._cont_data = self._iv_data
        self._nobs = nobs
        self._check_data()

    def _check_data(self):
        cat, cont = self._cat, self._cont
        cat_nobs = getattr(cat, 'shape', (0,))[0]
        cont_nobs = getattr(cont, 'shape', (0,))[0]
        nobs = max(cat_nobs, cont_nobs)
        if cat is None and cont is None:
            if self._nobs is not None:
                self._cont_data = self._cat_data = IVData(None, 'none', nobs=self._nobs)
            else:
                raise ValueError('nobs must be provided when cat and cont are None')
            return

        self._cat_data = IVData(cat, 'cat', nobs=nobs, convert_dummies=False)
        self._cont_data = IVData(cont, 'cont', nobs=nobs, convert_dummies=False)
        if self._cat_data.shape[1] == self._cont_data.shape[1] == 0:
            raise ValueError('Both cat and cont are empty arrays')
        cat_data = self._cat_data.pandas
        convert = [col for col in cat_data if not (is_categorical(cat_data[col]))]
        if convert:
            cat_data = DataFrame({col: cat_data[col].astype('category') for col in cat_data})
            self._cat_data = IVData(cat_data, 'cat', convert_dummies=False)

    @property
    def cat(self) -> DataFrame:
        """Categorical Variables"""
        return self._cat_data.pandas

    @property
    def cont(self) -> DataFrame:
        """Continusous Variables"""
        return self._cont_data.pandas

    @property
    def sparse(self) -> csc_matrix:
        r"""
        Construct a sparce interaction matrix

        Returns
        -------
        dummy_interact : csc_matrix
            Dummy interaction constructed from teh cartesian product of
            the categories and each of the continuous variables.

        Notes
        -----
        The number of columns in `dummy_interact` is

        .. math::

            ncont \times \prod_{i=1}^{ncat} |c_i|

        where :math:`|c_i|` is the number distinct categories in column i.
        """
        if self.cat.shape[1] and self.cont.shape[1]:
            out = []
            for col in self.cont:
                out.append(category_continuous_interaction(self.cat, self.cont[col]))
            return sp.hstack(out)
        elif self.cat.shape[1]:
            return category_interaction(category_product(self.cat))
        elif self.cont.shape[1]:
            return csc_matrix(self._cont_data.ndarray)
        else:  # empty interaction
            return csc_matrix(empty((self._cat_data.shape[0], 0)))

    @staticmethod
    def from_frame(frame: DataFrame) -> 'Interaction':
        """
        Convenience function the simplifies using a DataFrame

        Parameters
        ----------
        frame : DataFrame
            Frame containing categorical and continuous variables. All
            categorical variables are passed to `cat` and all other
            variables are passed as `cont`.

        Returns
        -------
        interaction : Interaction
            Instance using the columns of frame

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.absorbing.model import Interaction
        >>> import pandas as pd
        >>> rs = np.random.RandomState(0)
        >>> cats = pd.concat([pd.Series(pd.Categorical(rs.randint(i+2,size=n)))
        ...                  for i in range(4)],1)
        >>> cats.columns = ['cat{0}'.format(i) for i in range(4)]
        >>> columns = ['cont{0}'.format(i) for i in range(6)]
        >>> cont = pd.DataFrame(rs.standard_normal((n, 6)), columns=columns)
        >>> frame = pd.concat([cats, cont], 1)
        >>> interact = Interaction.from_frame(frame)
        >>> interact.sparse.shape # Cart product of all cats, 5!, times ncont, 6
        (100000, 720)
        """
        cat_cols = [col for col in frame if is_categorical(frame[col])]
        cont_cols = [col for col in frame if col not in cat_cols]
        return Interaction(frame[cat_cols], frame[cont_cols], nobs=frame.shape[0])


InteractionVar = Union[DataFrame, Interaction]


class AbsorbingLS(object):
    """
    Stub for documentation of AbsorbingLS

    TODO
    """
    def __init__(self, dependent: ArrayLike, exog: OptionalArrayLike = None,
                 absorb: InteractionVar = None,
                 interactions: Union[InteractionVar, Iterable[InteractionVar]] = None):

        self._dependent = IVData(dependent, 'dependent')
        self._exog = IVData(exog, 'exog')
        self._absorb = absorb
        if isinstance(absorb, DataFrame):
            self._absorb_inter = Interaction.from_frame(absorb)
        elif absorb is None:
            self._absorb_inter = Interaction(None, None, self._dependent.shape[0])
        elif isinstance(absorb, Interaction):
            self._absorb_inter = absorb
        else:
            raise TypeError('absort must ba a DataFrame or an Interaction')

        self._interactions = interactions
        self._interaction_list = []  # type: List[Interaction]
        self._prepare_interactions()
        self._absorbed_dependent = None
        self._absorbed_exog = None

    @property
    def absorbed_dependent(self) -> DataFrame:
        """
        Dependent variable with effects absorbed

        Returns
        -------
        dependent : DataFrame
            Dependent after effects have been absorbed

        Raises
        ------
        RuntimeError
            If called before `fit` has been used once
        """
        if self._absorbed_dependent is not None:
            return self._absorbed_dependent
        raise RuntimeError('fit must be called once before absorbed_dependent is available')

    @property
    def absorbed_exog(self) -> DataFrame:
        """
        Exogenous variables with effects absorbed

        Returns
        -------
        exogenous : DataFrame
            Exogenous after effects have been absorbed

        Raises
        ------
        RuntimeError
            If called before `fit` has been used once
        """
        if self._absorbed_exog is not None:
            return self._absorbed_exog
        raise RuntimeError('fit must be called once before absorbed_exog is available')

    def _prepare_interactions(self):
        if self._interactions is None:
            return
        elif isinstance(self._interactions, DataFrame):
            self._interaction_list = [Interaction.from_frame(self._interactions)]
        elif isinstance(self._interactions, Interaction):
            self._interaction_list = [self._interactions]
        else:
            for interact in self._interactions:
                if isinstance(interact, DataFrame):
                    self._interaction_list.append(Interaction.from_frame(interact))
                elif isinstance(interact, Interaction):
                    self._interaction_list.append(interact)
                else:
                    raise TypeError('interactions must contain DataFrames or Interactions')

    def fit(self, lsmr_options: dict = None):
        # Build regressor matrix
        regressors = []
        if self._absorb is not None:
            if self._absorb_inter.cat.shape[1] > 0:
                regressors.append(dummy_matrix(self._absorb_inter.cat)[0])
            if self._absorb_inter.cont.shape[1] > 0:
                regressors.append(csc_matrix(to_numpy(self._absorb_inter.cont)))
        if self._interactions is not None:
            regressors.extend([interact.sparse for interact in self._interaction_list])
        dep = self._dependent.ndarray
        exog = self._exog.ndarray
        lsmr_options = {} if lsmr_options is None else lsmr_options
        if regressors:
            absorb_x = sp.hstack(regressors)
            dep_resid = lsmr_annihilate(absorb_x, dep, **lsmr_options)
            if self._exog is not None:
                exog_resid = lsmr_annihilate(absorb_x, exog, **lsmr_options)
            else:
                exog_resid = empty((dep_resid.shape[0], 0))
        else:
            dep_resid = dep
            exog_resid = exog

        self._absorbed_dependent = DataFrame(dep_resid, index=self._dependent.pandas.index,
                                             columns=self._dependent.pandas.columns)
        self._absorbed_exog = DataFrame(exog_resid, index=self._exog.pandas.index,
                                        columns=self._exog.pandas.columns)
        if self._exog is None:
            # TODO early return, only absorbed
            return np.empty((0,))

        if matrix_rank(dep_resid) < dep_resid.shape[1]:
            # TODO: Refactor algo to find absorbed from panel
            # TODO: Move error to common location
            msg = absorbing_error_msg.format(absorbed_variables='unknown')
            raise AbsorbingEffectError(msg)
        b = lstsq(exog_resid, dep_resid)[0]
        return b


if __name__ == '__main__':
    import numpy as np

    rs = np.random.RandomState(0)
    n = 100000
    cats = rs.randint(2, size=n)  # binary dummy
    cont = rs.standard_normal((n, 3))
    interact = Interaction(cats, cont)
    print(interact.sparse.shape)  # Get the shape of the dummy matrix

    import numpy as np
    import pandas as pd

    n = 5000000
    rs = np.random.RandomState(0)
    cats = pd.concat([pd.Series(pd.Categorical(rs.randint(n // 7, size=n)))], 1)
    effects = rs.standard_normal((n, 1))
    x = pd.DataFrame(rs.standard_normal((n, 1)))
    y = pd.DataFrame(
        x.to_numpy() + rs.standard_normal((n, 1)) + effects[cats.to_numpy().squeeze()])

    ia = pd.concat([pd.Series(rs.standard_normal((n))),
                    pd.Series(pd.Categorical(rs.randint(n // 10, size=n)))], 1)
    mod = AbsorbingLS(y, x, cats, ia)
    print(mod.fit({'show': True}))
