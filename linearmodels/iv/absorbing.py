from linearmodels.compat.numpy import lstsq
from linearmodels.compat.pandas import get_codes, is_categorical, to_numpy

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Hashable, Iterable, List, Union

from numpy import (any as npany, arange, ascontiguousarray, average,
                   column_stack, dtype, empty, empty_like, int8, int16, int32,
                   int64, nanmean, ndarray, ones, ptp, sqrt, where, zeros)
from pandas import Categorical, DataFrame, Series
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr

from linearmodels.iv.common import f_statistic, find_constant
from linearmodels.iv.data import IVData
from linearmodels.iv.model import COVARIANCE_ESTIMATORS
from linearmodels.iv.results import AbsorbingLSResults
from linearmodels.panel.utility import (check_absorbed, dummy_matrix,
                                        preconditioner)
from linearmodels.typing import AnyPandas
from linearmodels.typing.data import ArrayLike, OptionalArrayLike
from linearmodels.utility import missing_warning

try:
    from xxhash import xxh64 as hash_func
except ImportError:
    from hashlib import sha1 as hash_func

SCALAR_DTYPES = {'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64}

_VARIABLE_CACHE = defaultdict(dict)  # type: DefaultDict[Hashable, Dict[str, ndarray]]


def _reset(hasher):
    try:
        hasher.reset()
        return hasher
    except AttributeError:
        return hash_func()


def clear_cache():
    """Clear the absorbed variable cache"""
    _VARIABLE_CACHE.clear()


def lsmr_annihilate(x: csc_matrix, y: ndarray, use_cache: bool = True, x_hash=None,
                    **lsmr_options) -> ndarray:
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
    x_hash : object
        Hashable object representing the values in x
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

    use_cache = use_cache and x_hash is not None
    regressor_hash = x_hash if x_hash is not None else ''
    default_opts = dict(atol=1e-8, btol=1e-8, show=False)
    default_opts.update(lsmr_options)
    resids = []
    for i in range(y.shape[1]):
        _y = y[:, i:i + 1]

        variable_digest = ''
        if use_cache:
            hasher = hash_func()
            hasher.update(ascontiguousarray(_y.data))
            variable_digest = hasher.hexdigest()

        if use_cache and variable_digest in _VARIABLE_CACHE[regressor_hash]:
            resid = _VARIABLE_CACHE[regressor_hash][variable_digest]
        else:
            beta = lsmr(x, _y, **default_opts)[0]
            resid = y[:, i:i + 1] - (x.dot(csc_matrix(beta[:, None]))).A
            _VARIABLE_CACHE[regressor_hash][variable_digest] = resid
        resids.append(resid)
    if resids:
        return column_stack(resids)
    else:
        return empty_like(y)


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


def category_interaction(cat: Series, precondition: bool = True) -> csc_matrix:
    """
    Parameters
    ----------
    cat : Series
        Categorical series to convert to dummy variables
    precondition : bool
        Flag whether dummies should be preconditioned

    Returns
    -------
    dummies : csc_matrix
        Sparse matrix of dummies with unit column norm
    """
    codes = get_codes(category_product(cat).cat)
    return dummy_matrix(codes[:, None], precondition=precondition)[0]


def category_continuous_interaction(cat: AnyPandas, cont: AnyPandas,
                                    precondition: bool = True) -> csc_matrix:
    """
    Parameters
    ----------
    cat : Series
        Categorical series to convert to dummy variables
    cont : {Series, DataFrame}
        Continuous variable values to use in the dummy interaction
    precondition : bool
        Flag whether dummies should be preconditioned

    Returns
    -------
    interact : csc_matrix
        Sparse matrix of dummy interactions with unit column norm
    """
    codes = get_codes(category_product(cat).cat)
    interact = csc_matrix((to_numpy(cont).flat, (arange(codes.shape[0]), codes)))
    if not precondition:
        return interact
    else:
        return preconditioner(interact)[0]


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
    >>> from linearmodels.iv.absorbing import Interaction
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

    @property
    def nobs(self):
        return self._nobs

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
        self._nobs = nobs

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
    def isnull(self) -> Series:
        return self.cat.isnull().any(1) | self.cont.isnull().any(1)

    def drop(self, locs) -> None:
        self._cat_data.drop(locs)
        self._cont_data.drop(locs)

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
                out.append(category_continuous_interaction(self.cat, self.cont[col],
                                                           precondition=False))
            return sp.hstack(out, format='csc')
        elif self.cat.shape[1]:
            return category_interaction(category_product(self.cat), precondition=False)
        elif self.cont.shape[1]:
            return csc_matrix(self._cont_data.ndarray)
        else:  # empty interaction
            return csc_matrix(empty((self._cat_data.shape[0], 0)))

    @property
    def hash(self):
        """
        Construct a hash that will be invariant for any permutation of
        inputs that produce the same fit when used as regressors"""
        # Sorted hashes of any categoricals
        hasher = hash_func()
        cat_hashes = []
        cat = self.cat
        for col in cat:
            hasher.update(ascontiguousarray(to_numpy(get_codes(self.cat[col].cat)).data))
            cat_hashes.append(hasher.hexdigest())
            hasher = _reset(hasher)
        cat_hashes = tuple(sorted(cat_hashes))

        hashes = []
        cont = self.cont
        for col in cont:
            hasher.update(ascontiguousarray(to_numpy(cont[col]).data))
            hashes.append(cat_hashes + (hasher.hexdigest(),))
            hasher = _reset(hasher)

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
        >>> from linearmodels.iv.absorbing import Interaction
        >>> import pandas as pd
        >>> rs = np.random.RandomState(0)
        >>> n = 100000
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


class AbsorbingRegressor(object):
    """
    Constructed weights sparse matrix from components

    Parameters
    ----------
    cat : DataFrame
        List of categorical variables (factors) to absorb
    cont : DataFrame
        List of continuous variables to absorb
    interactions : List[Interaction]
        List of included interactions
    weights : ndarray
        Weights, if any
    """

    def __init__(self, *, cat: DataFrame = None, cont: DataFrame = None,
                 interactions: List[Interaction] = None,
                 weights: ndarray = None):
        self._cat = cat
        self._cont = cont
        self._interactions = interactions
        self._weights = weights
        self._approx_rank = None

    @property
    def has_constant(self):
        """Flag indicating whether the regressors have a constant equivalent"""
        return self._cat is not None and self._cat.shape[1] > 0

    @property
    def approx_rank(self):
        if self._approx_rank is None:
            self._regressors()
        return self._approx_rank

    @property
    def hash(self):
        hashes = []
        hasher = hash_func()
        if self._cat is not None:
            for col in self._cat:
                hasher.update(ascontiguousarray(to_numpy(get_codes(self._cat[col].cat)).data))
                hashes.append((hasher.hexdigest(),))
                hasher = _reset(hasher)
        if self._cont is not None:
            for col in self._cont:
                hasher.update(ascontiguousarray(to_numpy(self._cont[col]).data))
                hashes.append((hasher.hexdigest(),))
                hasher = _reset(hasher)
        if self._interactions is not None:
            for interact in self._interactions:
                hashes.extend(interact.hash)
        # Add weight hash if provided
        if self._weights is not None:
            hasher = hash_func()
            hasher.update(ascontiguousarray(self._weights.data))
            hashes.append((hasher.hexdigest(),))
        return tuple(sorted(hashes))

    @property
    def regressors(self) -> csc_matrix:
        return self._regressors()

    def _regressors(self) -> csc_matrix:
        regressors = []

        if self._cat is not None and self._cat.shape[1] > 0:
            regressors.append(dummy_matrix(self._cat, precondition=False)[0])
        if self._cont is not None and self._cont.shape[1] > 0:
            regressors.append(csc_matrix(to_numpy(self._cont)))
        if self._interactions is not None:
            regressors.extend([interact.sparse for interact in self._interactions])

        if regressors:
            regressor_mat = sp.hstack(regressors, format='csc')
            approx_rank = regressor_mat.shape[1]
            self._approx_rank = approx_rank
            if self._weights is not None:
                return (sp.diags(sqrt(self._weights.squeeze())).dot(regressor_mat)).asformat('csc')
            return regressor_mat
        else:
            self._approx_rank = 0
            return csc_matrix(empty((0, 0)))


class AbsorbingLS(object):
    r"""
    Linear regression with high-dimensaional effects

    Parameters
    ----------
    dependent : ArrayLike
        Endogenous variables (nobs by 1)
    exog : ArrayLike, optional
        Exogenous regressors  (nobs by nexog)
    absorb: {DataFrame, Interaction}
        The effects or continuous variables to absorb. When using a DataFrame,
        effects must be categorical variables. Other variable types are treated
        as continuous variables that should be absorbed. When using an
        Interaction, variables in the `cat` argument are treaded as effects
        and variables in the `cont` argument are treated as continuous.
    interactions : {DataFrame, Interaction, List[DataFrame, Interaction]}, optional
        Interactions containing both categorical and continuous variables.  Each
        interaction is constructed using the Cartesian product of the categorical
        variables to produce the dummy, which are then separately interacted with
        each continuous variable.

    weights : ArrayLike, optional
        Observation weights used in estimation

    Notes
    -----
    Capable of estimating models with millions of effects.

    Estimates models of the form

    .. math::

      y_i = x_i \beta + z_i \gamma + \epsilon_i

    where :math:`\beta` are parameters of interest and :math:`\gamma`
    are not. z may be high-dimensional, although must have fewer
    variables than the number of observations in y.

    The syntax simplifies specifying high-dimensional z when z consists
    of categorical (factor) variables, also known as effects, or when
    z contains interactions between continuous variables and categorical
    variables, also known as fixed slopes.

    The high-dimensional effects are fit using LSMR which avoids inverting
    or even constructing the inner product of the regressors. This is
    combined with Frish-Waugh-Lovell to orthogonzlize x and y from z.

    z can contain factors that are perfectly linearly dependent. LSMR
    estimates a particular restricted set of parameters that captures the
    effect of non-redundant components in z.

    See also
    --------
    Interaction
    linearmodels.iv.model.IVLIML
    linearmodels.iv.model.IV2SLS
    scipy.sparse.linalg.lsmr

    Examples
    --------
    Estimate a model by absorbing 2 cateogricals and 2 continuous variables

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lineamodels.iv import AbsorbingLS, Interaction
    >>> dep = np.random.standard_normal((20000,1))
    >>> exog = np.random.standard_normal((20000,2))
    >>> cats = pd.DataFrame({i: pd.Categorical(np.random.randint(1000, size=20000))
    ...                      for i in range(2)})
    >>> cont = pd.DataFrame({i+2: np.random.standard_normal(20000) for i in range(2)})
    >>> absorb = pd.concat([cats, cont], 1)
    >>> mod = AbsorbingLS(dep, exog, absorb=absorb)
    >>> res = mod.fit()

    Add interactions between the cartesian product of the categorical and
    each continuous variables

    >>> iaction = Interaction(cat=cats, cont=cont)
    >>> absorb = Interaction(cat=cats) # Other encoding of categoricals
    >>> mod = AbsorbingLS(dep, exog, absorb=absorb, interactions=iaction)
    """

    def __init__(self, dependent: ArrayLike, exog: OptionalArrayLike = None, *,
                 absorb: InteractionVar = None,
                 interactions: Union[InteractionVar, Iterable[InteractionVar]] = None,
                 weights: OptionalArrayLike = None):

        self._dependent = IVData(dependent, 'dependent')
        self._nobs = nobs = self._dependent.shape[0]
        self._exog = IVData(exog, 'exog', nobs=self._nobs)
        self._absorb = absorb
        if isinstance(absorb, DataFrame):
            self._absorb_inter = Interaction.from_frame(absorb)
        elif absorb is None:
            self._absorb_inter = Interaction(None, None, nobs)
        elif isinstance(absorb, Interaction):
            self._absorb_inter = absorb
        else:
            raise TypeError('absorb must ba a DataFrame or an Interaction')
        self._weights = weights
        self._is_weighted = False
        self._check_weights()

        self._interactions = interactions
        self._interaction_list = []  # type: List[Interaction]
        self._prepare_interactions()
        self._absorbed_dependent = None
        self._absorbed_exog = None
        self._x = None

        self._check_shape()
        self._original_index = self._dependent.pandas.index
        self._drop_locs = self._drop_missing()
        self._columns = self._exog.cols
        self._index = self._dependent.rows
        self._method = 'Absorbing LS'

        self._const_col = 0
        self._has_constant = False
        self._has_constant_exog = self._check_constant()
        self._constant_absorbed = False
        self._num_params = 0
        self._regressors = None
        self._regressors_hash = None

    def _drop_missing(self) -> ndarray:
        missing = to_numpy(self.dependent.isnull)
        missing |= to_numpy(self.exog.isnull)
        missing |= to_numpy(self._absorb_inter.cat.isnull().any(1))
        missing |= to_numpy(self._absorb_inter.cont.isnull().any(1))
        for interact in self._interaction_list:
            missing |= to_numpy(interact.isnull)
        if npany(missing):
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self._absorb_inter.drop(missing)
            for interact in self._interaction_list:
                interact.drop(missing)
        missing_warning(missing)
        return missing

    def _check_constant(self):
        col_delta = ptp(self.exog.ndarray, 0)
        has_constant = npany(col_delta == 0)
        self._const_col = where(col_delta == 0)[0][0] if has_constant else None
        return has_constant

    def _check_weights(self):
        if self._weights is None:
            nobs = self._dependent.shape[0]
            self._is_weighted = False
            self._weight_data = IVData(ones(nobs), 'weights')
        else:
            self._is_weighted = True
            weights = IVData(self._weights).ndarray
            weights = weights / nanmean(weights)
            self._weight_data = IVData(weights, var_name='weights', nobs=self._nobs)

    def _check_shape(self):
        nobs = self._nobs
        if self._absorb is not None:
            if self._absorb_inter.nobs != nobs:
                raise ValueError('absorb and dependent have different number of observations')
        for interact in self._interaction_list:
            if interact.nobs != nobs:
                raise ValueError('interactions ({0}) and dependent have different number of '
                                 'observations'.format(str(interact)))

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
        return self._weight_data

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

    def _first_time_fit(self, use_cache, lsmr_options):
        weights = self.weights.ndarray if self._is_weighted else None

        areg = AbsorbingRegressor(cat=self._absorb_inter.cat, cont=self._absorb_inter.cont,
                                  interactions=self._interaction_list, weights=weights)
        areg_constant = areg.has_constant
        self._regressors = preconditioner(areg.regressors)[0]
        self._num_params += areg.approx_rank
        # Do not double count intercept-like terms
        self._has_constant = self._has_constant_exog or areg_constant
        self._num_params -= min(self._has_constant_exog, areg_constant)
        self._regressors_hash = areg.hash
        self._constant_absorbed = self._has_constant_exog and areg_constant

        dep = self._dependent.ndarray
        exog = self._exog.ndarray

        root_w = sqrt(self._weight_data.ndarray)
        dep = root_w * dep
        exog = root_w * exog
        denom = root_w.T @ root_w
        mu_dep = (root_w.T @ dep) / denom
        mu_exog = (root_w.T @ exog) / denom

        lsmr_options = {} if lsmr_options is None else lsmr_options
        if self._regressors.shape[1] > 0:
            dep_resid = lsmr_annihilate(self._regressors, dep, use_cache, self._regressors_hash,
                                        **lsmr_options)
            exog_resid = lsmr_annihilate(self._regressors, exog, use_cache,
                                         self._regressors_hash, **lsmr_options)
        else:
            dep_resid = dep
            exog_resid = exog

        if self._constant_absorbed:
            dep_resid += root_w * mu_dep
            exog_resid += root_w * mu_exog

        self._absorbed_dependent = DataFrame(dep_resid, index=self._dependent.pandas.index,
                                             columns=self._dependent.pandas.columns)
        self._absorbed_exog = DataFrame(exog_resid, index=self._exog.pandas.index,
                                        columns=self._exog.pandas.columns)

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
        results : AbsorbingLSResults
            Results container

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        The see the docstring of specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.

        If use_cache is True, then variables are hashed based on their
        contents using either a 64 bit value (if xxhash is installed) or
        a 256 bit value. This allows variables to be reused in different
        models if the set of absorbing variables and interactions is held
        constant.

        See also
        --------
        linearmodels.iv.covariance.HomoskedasticCovariance
        linearmodels.iv.covariance.HeteroskedasticCovariance
        linearmodels.iv.covariance.KernelCovariance
        linearmodels.iv.covariance.ClusteredCovariance
        """

        if self._absorbed_dependent is None:
            self._first_time_fit(use_cache, lsmr_options)

        self._x = exog_resid = to_numpy(self.absorbed_exog)
        dep_resid = to_numpy(self.absorbed_dependent)
        if self._exog.shape[1] == 0:
            params = empty((0, 1))
        else:
            if exog_resid.shape[1]:
                check_absorbed(exog_resid, self.exog.cols)
            params = lstsq(exog_resid, dep_resid)[0]
            self._num_params += exog_resid.shape[1]

        cov_estimator = COVARIANCE_ESTIMATORS[cov_type]
        cov_config['debiased'] = debiased
        cov_config['kappa'] = 0.0
        cov_config_copy = {k: v for k, v in cov_config.items()}
        if 'center' in cov_config_copy:
            del cov_config_copy['center']
        cov_estimator = cov_estimator(exog_resid, dep_resid, exog_resid, params, **cov_config_copy)

        results = {'kappa': 0.0,
                   'liml_kappa': 0.0}
        pe = self._post_estimation(params, cov_estimator, cov_type)
        results.update(pe)
        results['df_model'] = self._num_params

        return AbsorbingLSResults(results, self)

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
        resids = self.wresids(params)
        return resids / sqrt(self.weights.ndarray)

    def wresids(self, params: ndarray):
        """
        Compute weighted model residuals

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1)

        Returns
        -------
        wresids : ndarray
            Weighted model residuals

        Notes
        -----
        Uses weighted versions of data instead of raw data.  Identical to
        resids if all weights are unity.
        """
        return to_numpy(self._absorbed_dependent) - to_numpy(self._absorbed_exog) @ params

    def _f_statistic(self, params: ndarray, cov: ndarray, debiased: bool):
        const_loc = find_constant(self._exog.ndarray)
        resid_df = self._nobs - self._num_params

        return f_statistic(params, cov, debiased, resid_df, const_loc)

    def _post_estimation(self, params: ndarray, cov_estimator, cov_type: str):
        columns = self._columns
        index = self._index
        eps = self.resids(params)
        fitted = DataFrame(self._dependent.ndarray - eps, index=self._dependent.rows,
                           columns=['fitted_values'])
        absorbed_effects = DataFrame(to_numpy(self._absorbed_dependent) - to_numpy(fitted),
                                     columns=['absorbed_effects'], index=self._dependent.rows)

        weps = self.wresids(params)
        cov = cov_estimator.cov
        debiased = cov_estimator.debiased

        residual_ss = (weps.T @ weps)[0, 0]

        w = self.weights.ndarray
        root_w = sqrt(w)
        e = self._dependent.ndarray * root_w
        if self.has_constant:
            e = e - root_w * average(self._dependent.ndarray, weights=w)

        total_ss = float(e.T @ e)
        r2 = max(1 - residual_ss / total_ss, 0.0)

        e = to_numpy(self._absorbed_dependent)  # already scaled by root_w
        # If absorbing contains a constant, but exog does not, no need to demean
        if self._const_col is not None:
            col = self._const_col
            x = to_numpy(self._absorbed_exog)[:, col:col + 1]
            mu = (lstsq(x, to_numpy(e))[0]).squeeze()
            e = e - x * mu

        aborbed_total_ss = float(e.T @ e)
        r2_absorbed = max(1 - residual_ss / aborbed_total_ss, 0.0)

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
               'original_index': self._original_index,
               'absorbed_effects': absorbed_effects,
               'absorbed_r2': r2_absorbed}

        return out
