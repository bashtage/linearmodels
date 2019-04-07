from linearmodels.compat.numpy import lstsq
from linearmodels.compat.pandas import is_categorical, to_numpy, get_codes

from collections import defaultdict
from typing import Any, Iterable, List, Union

from numpy import (asarray, average, column_stack, dtype, empty, int8, int16,
                   int32, int64, ndarray, ones, sqrt, zeros)
from numpy.linalg import matrix_rank
from pandas import Categorical, DataFrame, Series
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr

from linearmodels.iv.common import f_statistic, find_constant
from linearmodels.iv.data import IVData
from linearmodels.iv.model import COVARIANCE_ESTIMATORS
from linearmodels.iv.results import OLSResults
from linearmodels.panel.model import AbsorbingEffectError, absorbing_error_msg
from linearmodels.panel.utility import dummy_matrix, preconditioner
from linearmodels.typing import AnyPandas
from linearmodels.typing.data import ArrayLike, OptionalArrayLike

try:
    from xxhash import xxh64 as hash_func
except ImportError:
    from hashlib import sha1 as hash_func

SCALAR_DTYPES = {'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64}

_VARIABLE_CACHE = defaultdict(dict)


def clear_cache():
    """Clear the variable cache"""
    for key in _VARIABLE_CACHE:
        del _VARIABLE_CACHE[key]


def lsmr_annihilate(x: csc_matrix, y: ndarray, use_cache: bool = True, **lsmr_options) -> ndarray:
    r"""
    Removes projection of x on y from y

    Parameters
    ----------
    x : csc_matrix
        Sparse array of regressors
    y : ndarray
        Array with shape (nobs, nvar)
    use_cache : bool
        Flag indicating whether results should be stored in the cache,
        and retrieved if available.
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

    check_cache = False
    regressor_hash = ''

    if use_cache:
        hasher = hash_func()
        hasher.update(x.data.data)
        hasher.update(x.indptr.data)
        hasher.update(x.indptr.data)
        regressor_hash = hasher.hexdigest()
        check_cache = regressor_hash in _VARIABLE_CACHE

    default_opts = dict(atol=1e-8, btol=1e-8, show=False)
    default_opts.update(lsmr_options)
    resids = []
    for i in range(y.shape[1]):
        _y = y[:, i:i + 1]

        variable_digest = ''
        if check_cache:
            hasher = hash_func()
            hasher.update(_y.data)
            variable_digest = hasher.hexdigest()

        if check_cache and variable_digest in _VARIABLE_CACHE[regressor_hash]:
            resid = _VARIABLE_CACHE[regressor_hash][variable_digest]
        else:
            beta = lsmr(x, _y, **default_opts)[0]
            resid = y[:, i:i + 1] - (x.dot(csc_matrix(beta[:, None]))).A
            _VARIABLE_CACHE[regressor_hash][variable_digest] = resid
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
        max_code = get_codes(col.cat).max()
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
        codes += (get_codes(cats[col].cat).astype(dtype_val) << SCALAR_DTYPES[dtype_str](cum_size))
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
    codes = get_codes(category_product(cat).cat)
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
    codes = get_codes(category_product(cat).cat)
    dummies = dummy_matrix(codes[:, None], precondition=False)[0]
    dummies.data[:] = to_numpy(cont).flat
    return preconditioner(dummies)[0]


class Interaction(object):
    """
    Class that simplifies specifying interactions

    Parameters
    ----------
    cat : {ndarray, Series, DataFrame, DataArray}, optional
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

    def hash(self):
        """
        Construct a hash that will be invariant for any permutation of
        inputs that produce the same fit when used as regressors"""
        # Sorted hashes of any categoricals
        hasher = hash_func()
        cat_hashes = []
        cat = self.cat
        for col in cat:
            hasher.update(get_codes(self.cat[col].cat).data)
            cat_hashes.append(hasher.hexdigest())
            hasher.reset()
        cat_hashes = tuple(sorted(cat_hashes))
        hashes = []
        cont = self.cont
        for col in cont:
            hasher.update(to_numpy(cont[col]).data)
            hashes.append(cat_hashes + (hasher.hexdigest(),))
            hasher.reset()
        return sorted(hashes)

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


class AbsorbingRegressor():

    def __init__(self, *, cat: DataFrame = None, cont: DataFrame = None,
                 interactions: List[Interaction] = None):
        self._cat = cat
        self._cont = cont
        self._interactions = interactions
        self._regressors = None

    def hash(self):
        hashes = []
        hasher = hash_func()
        if self._cat is not None:
            for col in self._cat:
                hasher.update(get_codes(self._cat[col].cat).data)
                hashes.append(hasher.hexdigest())
                hasher.reset()
        if self._cont is not None:
            for col in self._cont:
                hasher.update(to_numpy(self._cont[col]).data)
                hashes.append(hasher.hexdigest())
                hasher.reset()
        for interact in self._interactions:
            hashes.append(interact.hash())
        return tuple(sorted(hashes))

    @property
    def regressors(self) -> csc_matrix:
        if self._regressors is not None:
            return self._regressors

        regressors = []

        if self._cat is not None and self._cat.shape[1] > 0:
            regressors.append(dummy_matrix(self._cat)[0])
        if self._cont is not None and self._cont.shape[1] > 0:
            regressors.append(csc_matrix(to_numpy(self._cont)))
        if self._interactions is not None:
            regressors.extend([interact.sparse for interact in self._interactions])

        # TODO: This is wrong
        self._regressors = sp.hstack(regressors)

        return self._regressors


class AbsorbingLS(object):
    """
    Stub for documentation of AbsorbingLS

    TODO
    """

    def __init__(self, dependent: ArrayLike, exog: OptionalArrayLike = None,
                 absorb: InteractionVar = None,
                 interactions: Union[InteractionVar, Iterable[InteractionVar]] = None,
                 weights: OptionalArrayLike = None):

        self._dependent = IVData(dependent, 'dependent')
        nobs = self._dependent.shape[0]
        self._exog = IVData(exog, 'exog')
        self._absorb = absorb
        if isinstance(absorb, DataFrame):
            self._absorb_inter = Interaction.from_frame(absorb)
        elif absorb is None:
            self._absorb_inter = Interaction(None, None, nobs)
        elif isinstance(absorb, Interaction):
            self._absorb_inter = absorb
        else:
            raise TypeError('absorb must ba a DataFrame or an Interaction')
        if weights is None:
            nobs = self._dependent.shape[0]
            self._weights = IVData(ones(nobs), 'weights')
        else:
            self._weights = IVData(weights, 'weights')

        self._interactions = interactions
        self._interaction_list = []  # type: List[Interaction]
        self._prepare_interactions()
        self._absorbed_dependent = None
        self._absorbed_exog = None
        self._x = None

        self._columns = self._exog.cols
        self._index = self._dependent.rows
        self._method = 'OLS'
        self._original_index = self._dependent.pandas.index
        # TODO: check formally
        self._has_constant = True
        self._nobs = self._dependent.shape[0]
        self._num_params = 0

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

    @property
    def weights(self):
        return self._weights

    @property
    def dependent(self):
        return self._dependent

    @property
    def exog(self):
        return self._exog

    @property
    def has_constant(self):
        return self._has_constant

    @property
    def instruments(self):
        return IVData(None, 'instrument', nobs=self._dependent.shape[0])

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

    def fit(self, *, cov_type: str = 'robust', debiased: bool = False, lsmr_options: dict = None,
            use_cache: bool = True, **cov_config: Any):
        """
        Estimate model parameters

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator to use. Supported covariance
            estimators are:

            * 'unadjusted', 'homoskedastic' - Classic homoskedastic inference
            * 'robust', 'heteroskedastic' - Heteroskedasticity robust inference
            * 'kernel' - Heteroskedasticity and autocorrelation robust
              inference
            * 'cluster' - One-way cluster dependent inference.
              Heteroskedasticity robust

        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional parameters to pass to covariance estimator. The list
            of optional parameters differ according to ``cov_type``. See
            the documentation of the alternative covariance estimators for
            the complete list of available commands.
        lsmr_options : dict
            Dictionary of options to pass to scipy.sparse.linalg.lsmr
        use_cache : bool
            Flag indicating whether the variables, once purged from the
            absorbed variables and interactions, should be stored in the cache,
            and retrieved if available. Cache can dramatically speed up
            re-fitting large models when the set of absorbed variables and
            interactions are identical.

        Returns
        -------
        results : OLSResults
            Results container

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        The see the docstring of specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.

        See also
        --------
        linearmodels.iv.covariance.HomoskedasticCovariance
        linearmodels.iv.covariance.HeteroskedasticCovariance
        linearmodels.iv.covariance.KernelCovariance
        linearmodels.iv.covariance.ClusteredCovariance
        """
        areg = AbsorbingRegressor(self._absorb_inter.cat, self._absorb_inter.cont,
                                  self._interaction_list)
        regressors = areg.regressors
        dep = self._dependent.ndarray
        exog = self._exog.ndarray
        lsmr_options = {} if lsmr_options is None else lsmr_options
        if regressors.shape[1] > 0:
            dep_resid = lsmr_annihilate(regressors, dep, use_cache, **lsmr_options)
            if self._exog is not None:
                exog_resid = lsmr_annihilate(regressors, exog, use_cache, **lsmr_options)
            else:
                exog_resid = empty((dep_resid.shape[0], 0))
        else:
            dep_resid = dep
            exog_resid = exog

        self._absorbed_dependent = DataFrame(dep_resid, index=self._dependent.pandas.index,
                                             columns=self._dependent.pandas.columns)
        self._absorbed_exog = DataFrame(exog_resid, index=self._exog.pandas.index,
                                        columns=self._exog.pandas.columns)
        self._x = self._absorbed_exog
        if self._exog is None:
            # TODO early return, only absorbed
            params = empty((0,))
        else:
            if matrix_rank(dep_resid) < dep_resid.shape[1]:
                # TODO: Refactor algo to find absorbed from panel
                # TODO: Move error to common location
                msg = absorbing_error_msg.format(absorbed_variables='unknown')
                raise AbsorbingEffectError(msg)
            params = lstsq(exog_resid, dep_resid)[0]
            self._num_params += dep_resid.shape[1]

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        cov_config['debiased'] = debiased
        cov_config['kappa'] = 0.0
        cov_config_copy = {k: v for k, v in cov_config.items()}
        if 'center' in cov_config_copy:
            del cov_config_copy['center']
        cov_estimator = cov_estimator(exog_resid, dep_resid, dep_resid, params, **cov_config_copy)

        results = {'kappa': 0.0,
                   'liml_kappa': 0.0}
        pe = self._post_estimation(params, cov_estimator, cov_type)
        results.update(pe)

        return OLSResults(results, self)

    def resids(self, params: ndarray):
        """
        Compute model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1)

        Returns
        -------
        resids : ndarray
            Model residuals
        """
        return to_numpy(self._absorbed_dependent) - to_numpy(self._absorbed_exog) @ params

    def wresids(self, params: ndarray):
        # TODO: weights
        return self.resids(params)

    def _f_statistic(self, params: ndarray, cov: ndarray, debiased: bool):
        const_loc = find_constant(self._exog.ndarray)
        resid_df = self._nobs - self._num_params

        return f_statistic(params, cov, debiased, resid_df, const_loc)

    def _post_estimation(self, params: ndarray, cov_estimator, cov_type: str):
        columns = self._columns
        index = self._index
        eps = self.resids(params)
        y = self._absorbed_dependent
        # TODO: save effects
        # TODO: effects in fitted? self._effects
        fitted = DataFrame(asarray(y) - eps, y.index, ['fitted_values'])
        weps = self.wresids(params)
        cov = cov_estimator.cov
        debiased = cov_estimator.debiased

        residual_ss = (weps.T @ weps)

        w = self.weights.ndarray
        e = self._dependent.ndarray * sqrt(w)
        if True:  # self.has_constant: TODO: constant
            e = e - sqrt(self.weights.ndarray) * average(self._dependent.ndarray, weights=w)

        total_ss = float(e.T @ e)
        r2 = 1 - residual_ss / total_ss

        fstat = self._f_statistic(params, cov, debiased)
        out = {'params': Series(params.squeeze(), columns, name='parameter'),
               'eps': Series(eps.squeeze(), index=index, name='residual'),
               'weps': Series(weps.squeeze(), index=index, name='weighted residual'),
               'cov': DataFrame(cov, columns=columns, index=columns),
               's2': float(cov_estimator.s2),
               'debiased': debiased,
               'residual_ss': float(residual_ss),
               'total_ss': float(total_ss),
               'r2': float(r2),
               'fstat': fstat,
               'vars': columns,
               'instruments': [],
               'cov_config': cov_estimator.config,
               'cov_type': cov_type,
               'method': self._method,
               'cov_estimator': cov_estimator,
               'fitted': fitted,
               'original_index': self._original_index}

        return out


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

    n = 1000000
    rs = np.random.RandomState(0)
    cats = pd.concat([pd.Series(pd.Categorical(rs.randint(n // 7, size=n)))], 1)
    effects = rs.standard_normal((n, 1))
    x = pd.DataFrame(rs.standard_normal((n, 3)))
    y = pd.DataFrame(
        x.to_numpy().sum(1)[:, None] + rs.standard_normal((n, 1)) + effects[
            cats.to_numpy().squeeze()])

    ia = pd.concat([pd.Series(rs.standard_normal((n))),
                    pd.Series(pd.Categorical(rs.randint(n // 10, size=n)))], 1)
    interact = Interaction.from_frame(ia)
    print(interact.hash())
    mod = AbsorbingLS(y, x, cats, ia)
    print(mod.fit(lsmr_options={'show': True}))
    print(mod.fit(lsmr_options={'show': True}))
