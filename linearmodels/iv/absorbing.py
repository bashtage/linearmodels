"""
Linear regression with high-dimensional effects.
"""

from __future__ import annotations

from linearmodels.compat.scipy import csc_array, diags_array

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from copy import deepcopy
import hashlib
from inspect import signature
from math import isnan
from typing import Any, DefaultDict, TypedDict, Union, cast
import warnings

from numpy import (
    any as npany,
    arange,
    asarray,
    ascontiguousarray,
    average,
    column_stack,
    dtype,
    empty,
    empty_like,
    int8,
    int16,
    int32,
    int64,
    ndarray,
    ones,
    ptp,
    require,
    sqrt,
    squeeze,
    unique,
    where,
    zeros,
)
from numpy.linalg import lstsq
from pandas import Categorical, CategoricalDtype, DataFrame, Series
from pandas.api.types import is_numeric_dtype
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

from linearmodels.iv.common import f_statistic
from linearmodels.iv.covariance import KERNEL_LOOKUP
from linearmodels.iv.data import IVData
from linearmodels.iv.model import (
    _LS_COV_KEYS,
    COVARIANCE_ESTIMATORS,
    ClusteredCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    KernelCovariance,
)
from linearmodels.iv.results import AbsorbingLSResults
from linearmodels.panel.utility import (
    AbsorbingEffectWarning,
    absorbing_warn_msg,
    check_absorbed,
    dummy_matrix,
    not_absorbed,
    preconditioner,
)
from linearmodels.shared.alignment import (
    check_alignment,
    is_default_index,
    reference_labels,
    row_labels,
)
from linearmodels.shared.exceptions import (
    check_cov_config,
    lookup_option,
    missing_warning,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.shared.inputs import as_array_like
from linearmodels.shared.rsquared import r_squared
from linearmodels.shared.utility import DataFrameWrapper, SeriesWrapper
from linearmodels.shared.weights import check_weights, normalize_weights
import linearmodels.typing.data

HAVE_XXHASH = False
try:
    from xxhash import xxh64

    HAVE_XXHASH = True
except ImportError:
    pass


def _non_negative(r2: float) -> float:
    """Clip an R2 at 0, which is not clipped if it is NaN since it is not defined."""
    return r2 if isnan(r2) else max(r2, 0.0)


class Hasher:
    """
    Hash the values of an array using xxhash if it is installed, and SHA-256 if not.
    """

    def __init__(self) -> None:
        self._hasher: hashlib._Hash | xxh64
        if HAVE_XXHASH:
            self._hasher = xxh64()
            self._use_xxh64 = True
        else:
            self._hasher = hashlib.sha256()
            self._use_xxh64 = False

    def reset(self) -> None:
        """Reset the hasher to its initial state."""
        if self._use_xxh64:
            assert isinstance(self._hasher, xxh64)
            self._hasher.reset()
        else:
            self._hasher = hashlib.sha256()

    def update(self, data: memoryview) -> None:
        """
        Add data to the hash.

        Parameters
        ----------
        data : memoryview
            The values to hash.
        """
        self._hasher.update(data)

    def digest(self) -> bytes:
        """
        Return the digest of the data added so far.

        Returns
        -------
        bytes
            The digest.
        """
        return self._hasher.digest()

    def hexdigest(self) -> str:
        """
        Return the digest of the data added so far as a string of hexadecimal digits.

        Returns
        -------
        str
            The digest.
        """
        return self._hasher.hexdigest()


_VARIABLE_CACHE: DefaultDict[Hashable, dict[str, ndarray]] = defaultdict(dict)


def clear_cache() -> None:
    """Clear the absorbed variable cache."""
    _VARIABLE_CACHE.clear()


def _options_key(options: Mapping[str, Any]) -> tuple[Any, ...]:
    """
    Hashable key that identifies options of an absorption algorithm.

    Parameters
    ----------
    options : dict
        The options. A value can be an array or a dict.

    Returns
    -------
    tuple
        Key that is equal when the options are equal. The ``show`` option is
        left out since it only controls what is printed.
    """
    key = []
    for name in sorted(options):
        value = options[name]
        if name == "show":
            continue
        if isinstance(value, dict):
            value = _options_key(value)
        elif isinstance(value, (ndarray, list, tuple)):
            array = asarray(value)
            value = (array.shape, str(array.dtype), array.tobytes())
        key.append((name, value))
    return tuple(key)


def _check_method(method: str) -> str:
    """
    Check the method used to absorb the variables.

    Parameters
    ----------
    method : str
        The method that was passed to fit.

    Returns
    -------
    str
        The method in lower case.

    Raises
    ------
    ValueError
        If method is not "auto", "hdfe" or "lsmr", whatever its case
    """
    if isinstance(method, str) and method.lower() in ("auto", "hdfe", "lsmr"):
        return method.lower()
    raise ValueError(
        'method must be one of "auto", "hdfe" or "lsmr", whatever its case. '
        f"Got {method!r}."
    )


def _absorb_args(
    method: str, use_cache: bool, absorb_options: dict[str, Any] | None
) -> tuple[Any, ...]:
    """The arguments of AbsorbingLS.fit that determine the absorbed variables."""
    options = {} if absorb_options is None else absorb_options
    return method, use_cache, _options_key(options)


def _is_nested(
    effect: linearmodels.typing.data.IntArray,
    cluster: linearmodels.typing.data.AnyArray,
) -> bool:
    """
    Test whether every category of an effect is in a single cluster.

    Parameters
    ----------
    effect : ndarray
        Integer codes of the categories of the effect, with shape (nobs,).
    cluster : ndarray
        Cluster of each observation, with shape (nobs,).

    Returns
    -------
    bool
        True if each category of the effect has observations in one cluster only.
    """
    _, effect_codes = unique(effect, return_inverse=True)
    _, cluster_codes = unique(cluster, return_inverse=True)
    pairs = unique(effect_codes * (cluster_codes.max() + 1) + cluster_codes)
    return bool(pairs.shape[0] == effect_codes.max() + 1)


def _check_absorb_options(options: dict[str, Any], use_hdfe: bool, method: str) -> None:
    """
    Raise a ValueError if options are those of the algorithm that is not used.

    Parameters
    ----------
    options : dict
        The options of the algorithm that absorbs the variables.
    use_hdfe : bool
        Flag indicating that pyhdfe absorbs the variables and not LSMR.
    method : str
        The method that was passed to fit.

    Notes
    -----
    The algorithm depends on the method and the model, and "auto" uses HDFE if
    it can. The options of LSMR that are passed to pyhdfe.create, and the options
    of pyhdfe.create that are passed to lsmr, are a TypeError that does not
    say why. The options of the two algorithms have different names.
    """
    from pyhdfe import create  # noqa: PLC0415

    lsmr_names = set(signature(lsmr).parameters) - {"A", "b"}
    hdfe_names = set(signature(create).parameters) - {"ids"}
    wrong = sorted(set(options).intersection(lsmr_names if use_hdfe else hdfe_names))
    if not wrong:
        return
    names = ", ".join(f"'{name}'" for name in wrong)
    why = f'method is "{method}"'
    if use_hdfe:
        if method == "auto":
            why += " and the model can be absorbed using HDFE"
        raise ValueError(
            f"absorb_options contains options of scipy.sparse.linalg.lsmr ({names}) "
            "that pyhdfe.create does not have. The variables are absorbed using "
            f'HDFE because {why}. Use method="lsmr" to absorb them using LSMR.'
        )
    if method == "auto":
        why += (
            " and the model cannot be absorbed using HDFE, which requires that it "
            "is not weighted and that the absorbed variables are all categorical"
        )
    raise ValueError(
        f"absorb_options contains options of pyhdfe.create ({names}) that "
        "scipy.sparse.linalg.lsmr does not have. The variables are absorbed "
        f"using LSMR because {why}. Use options of LSMR or remove them."
    )


class _LSMROptions(TypedDict, total=False):
    """The keyword arguments of scipy.sparse.linalg.lsmr that can be passed."""

    damp: float
    atol: float
    btol: float
    conlim: float
    maxiter: int | None
    show: bool
    x0: linearmodels.typing.data.Float64Array | None


def lsmr_annihilate(
    x: csc_array,
    y: linearmodels.typing.data.Float64Array,
    use_cache: bool = True,
    x_hash: Hashable | None = None,
    **lsmr_options: (
        bool | float | str | linearmodels.typing.data.ArrayLike | dict[str, Any] | None
    ),
) -> linearmodels.typing.data.Float64Array:
    r"""
    Remove the projection of y on x from y.

    Parameters
    ----------
    x : csc_array
        Sparse array of regressors.
    y : ndarray
        Array with shape (nobs, nvar).
    use_cache : bool, optional
        Flag indicating whether results should be stored in the cache,
        and retrieved if available. The default is True.
    x_hash : object, optional
        Hashable object representing the values in x. The cache is not used if
        it is not provided.
    **lsmr_options : dict
        Options to pass to scipy.sparse.linalg.lsmr. The defaults are
        ``atol=1e-8``, ``btol=1e-8`` and ``show=False``.

    Returns
    -------
    ndarray
        The residuals from regressing y on x, with shape (nobs, nvar).

    Notes
    -----
    Residuals are estimated column-by-column as

    .. math::

        \hat{\epsilon}_{j} = y_{j} - x^\prime \hat{\beta}

    where :math:`\hat{\beta}` is computed using lsmr.

    The cache identifies residuals using `x_hash`, the values of the column
    of `y` and the options passed to lsmr, since residuals computed using
    other options are not the same.
    """
    if y.shape[1] == 0:
        return empty_like(y)
    use_cache = use_cache and x_hash is not None
    default_opts: _LSMROptions = {"atol": 1e-8, "btol": 1e-8, "show": False}
    assert lsmr_options is not None
    # The options are keyword arguments of lsmr. AbsorbingLS.fit rejects the options
    # of pyhdfe with _check_absorb_options before they reach this function
    default_opts.update(cast("_LSMROptions", lsmr_options))
    cache_key = (x_hash, _options_key(default_opts))
    resids = []
    for i in range(y.shape[1]):
        _y = y[:, i : i + 1]

        variable_digest = ""
        if use_cache:
            hasher = Hasher()
            hasher.update(memoryview(ascontiguousarray(_y.data)))
            variable_digest = hasher.hexdigest()

        if use_cache and variable_digest in _VARIABLE_CACHE[cache_key]:
            resid = _VARIABLE_CACHE[cache_key][variable_digest]
        else:
            beta = lsmr(x, _y, **default_opts)[0]
            resid = y[:, i : i + 1] - x.dot(csc_array(beta[:, None])).toarray()
            if use_cache:
                _VARIABLE_CACHE[cache_key][variable_digest] = resid
        resids.append(resid)
    return column_stack(resids)


def category_product(cats: linearmodels.typing.AnyPandas) -> Series:
    """
    Construct category from all combination of input categories.

    Parameters
    ----------
    cats : Series or DataFrame
        DataFrame containing categorical variables.  If cats is a Series, cats
        is returned unmodified.

    Returns
    -------
    Series
        Categorical series containing the cartesian product of the categories
        in cats.

    Raises
    ------
    TypeError
        If a column of cats is not categorical.
    ValueError
        If the categories have too many states to be combined.
    """
    if isinstance(cats, Series):
        return cats
    assert isinstance(cats, DataFrame)
    sizes = []
    for c in cats:
        # TODO: Bug in pandas-stubs
        #  https://github.com/pandas-dev/pandas-stubs/issues/97
        if not isinstance(cats[c].dtype, CategoricalDtype):
            raise TypeError("cats must contain only categorical variables")
        # TODO: Bug in pandas-stubs
        #  https://github.com/pandas-dev/pandas-stubs/issues/97
        col = cats[c]
        max_code = col.cat.codes.max()
        size = 1
        while max_code >= 2**size:
            size += 1
        sizes.append(size)
    nobs = cats.shape[0]
    total_size = sum(sizes)
    if total_size >= 63:
        raise ValueError(
            "There are too many cats with too many states to use this method."
        )
    dtype_size = min(filter(lambda v: total_size < (v - 1), (8, 16, 32, 64)))
    dtype_str = f"int{dtype_size:d}"
    dtype_val = dtype(dtype_str)
    codes = zeros(nobs, dtype=dtype_val)
    cum_size = 0
    for i, col_name in enumerate(cats):
        if dtype_str == "int8":
            shift: int8 | int16 | int32 | int64 = int8(cum_size)
        elif dtype_str == "int16":
            shift = int16(cum_size)
        elif dtype_str == "int32":
            shift = int32(cum_size)
        else:  # elif dtype_str == "int64":
            shift = int64(cum_size)
        cat_codes = asarray(cats[col_name].cat.codes)
        codes += cat_codes.astype(dtype_val) << shift
        cum_size += sizes[i]

    return Series(Categorical(codes), index=cats.index)


def category_interaction(cat: Series, precondition: bool = True) -> csc_array:
    """
    Convert a categorical series to a sparse matrix of dummy variables.

    Parameters
    ----------
    cat : Series
        Categorical series to convert to dummy variables.
    precondition : bool, optional
        Flag whether dummies should be preconditioned. The default is True.

    Returns
    -------
    csc_array
        Sparse matrix of dummies. The columns have unit norm if the dummies are
        preconditioned.
    """
    codes = asarray(category_product(cat).cat.codes)[:, None]
    mat = dummy_matrix(codes, precondition=precondition)[0]
    assert isinstance(mat, csc_array)
    return mat


def category_continuous_interaction(
    cat: linearmodels.typing.AnyPandas,
    cont: linearmodels.typing.AnyPandas,
    precondition: bool = True,
) -> csc_array:
    """
    Interact a categorical series with continuous variables.

    Parameters
    ----------
    cat : Series or DataFrame
        Categorical series to convert to dummy variables. The cartesian product
        of the categories of a DataFrame is used.
    cont : Series or DataFrame
        Continuous variable values to use in the dummy interaction.
    precondition : bool, optional
        Flag whether dummies should be preconditioned. The default is True.

    Returns
    -------
    csc_array
        Sparse matrix of dummy interactions. The columns have unit norm if the
        dummies are preconditioned.
    """
    codes = category_product(cat).cat.codes
    interact = csc_array((cont.to_numpy().flat, (arange(codes.shape[0]), codes)))
    if not precondition:
        return interact
    else:
        contioned = preconditioner(interact)[0]
        assert isinstance(contioned, csc_array)
        return contioned


class Interaction:
    """
    Class that simplifies specifying interactions.

    Parameters
    ----------
    cat : array_like, optional
        Variables to treat as categoricals. Best format is a Categorical
        Series or DataFrame containing Categorical Series. Other formats
        are converted to Categorical Series, column-by-column. cat has
        shape (nobs, ncat).
    cont : array_like, optional
        Variables to treat as continuous, (nobs, ncont). A ``TypeError`` is
        raised if the values are complex or cannot be converted to float64.
    nobs : int, optional
        Number of observations. It is required if both ``cat`` and ``cont`` are
        not provided, and it is otherwise the number of rows of the inputs.

    Raises
    ------
    TypeError
        If ``cat`` or ``cont`` is complex or cannot be converted to float64.
    ValueError
        If ``nobs`` is not provided when ``cat`` and ``cont`` are not provided,
        if both are empty, if they have different numbers of rows, or if they
        carry labels that differ.

    Notes
    -----
    For each variable in `cont`, computes the interaction of the variable
    and the cartesian product of the categories.

    If both ``cat`` and ``cont`` carry labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates), they must use the same index, in the same
    order.

    Examples
    --------
    >>> import numpy as np
    >>> from linearmodels.iv.absorbing import Interaction
    >>> rs = np.random.RandomState(0)
    >>> n = 100000
    >>> cats = rs.randint(2, size=n)  # binary dummy
    >>> cont = rs.standard_normal((n, 3))
    >>> interact = Interaction(cats, cont)
    >>> interact.sparse.shape  # Get the shape of the dummy matrix
    (100000, 6)

    >>> rs = np.random.RandomState(0)
    >>> import pandas as pd
    >>> cats_df = pd.concat([pd.Series(pd.Categorical(rs.randint(5,size=n)))
    ...                     for _ in range(4)], axis=1)
    >>> cats_df.describe()
                 0       1       2       3
    count   100000  100000  100000  100000
    unique       5       5       5       5
    top          3       3       0       4
    freq     20251   20195   20331   20158

    >>> interact = Interaction(cats_df, cont)
    >>> interact.sparse.shape # Cart product of all cats, 5**4, times ncont, 3
    (100000, 1875)
    """

    _iv_data = IVData(None, "none", 1)

    def __init__(
        self,
        cat: linearmodels.typing.data.ArrayLike | None = None,
        cont: linearmodels.typing.data.ArrayLike | None = None,
        nobs: int | None = None,
    ) -> None:
        # Lists and tuples are arrays, which have the shape that is used below
        self._cat = None if cat is None else as_array_like(cat, "cat")
        self._cont = None if cont is None else as_array_like(cont, "cont")
        self._cat_data = self._iv_data
        self._cont_data = self._iv_data
        self._nobs = nobs
        self._check_data()

    @property
    def nobs(self) -> int:
        """Number of observations, which drop reduces."""
        assert self._nobs is not None
        return self._nobs

    def _check_data(self) -> None:
        cat, cont = self._cat, self._cont
        cat_nobs = getattr(cat, "shape", (0,))[0]
        cont_nobs = getattr(cont, "shape", (0,))[0]
        nobs = max(cat_nobs, cont_nobs)
        if cat is None and cont is None:
            if self._nobs is not None:
                # Separate objects, since drop modifies them in place
                self._cat_data = IVData(None, "none", nobs=self._nobs)
                self._cont_data = IVData(None, "none", nobs=self._nobs)
            else:
                raise ValueError("nobs must be provided when cat and cont are None")
            return
        self._nobs = nobs

        self._cat_data = IVData(cat, "cat", nobs=nobs, convert_dummies=False)
        self._cont_data = IVData(cont, "cont", nobs=nobs, convert_dummies=False)
        if self._cat_data.shape[1] == self._cont_data.shape[1] == 0:
            raise ValueError("Both cat and cont are empty arrays")
        check_alignment(
            row_labels(self._cat_data), {"cont": self._cont_data}, reference="cat"
        )
        cat_data = self._cat_data.pandas
        convert = [
            col for col in cat_data if not (isinstance(cat_data[col], CategoricalDtype))
        ]
        if convert:
            cat_data = DataFrame(
                {col: cat_data[col].astype("category") for col in cat_data}
            )
            self._cat_data = IVData(cat_data, "cat", convert_dummies=False)

    @property
    def cat(self) -> DataFrame:
        """Categorical variables."""
        return self._cat_data.pandas

    @property
    def cont(self) -> DataFrame:
        """Continuous variables."""
        return self._cont_data.pandas

    @property
    def isnull(self) -> Series:
        """Return a Series that is True for the rows that contain a missing value."""
        cat_null = self.cat.isnull().any(axis=1)
        cont_null = self.cont.isnull().any(axis=1)
        # Rows are paired by position since cat and cont need not have labels
        index = cont_null.index if is_default_index(cat_null.index) else cat_null.index
        return Series(cat_null.to_numpy() | cont_null.to_numpy(), index=index)

    def drop(self, locs: linearmodels.typing.data.BoolArray) -> None:
        """
        Drop observations.

        Parameters
        ----------
        locs : ndarray
            Boolean array with one element for each observation, which is True
            for the observations to drop.

        Notes
        -----
        The instance is modified in place and nothing is returned. Afterwards
        ``nobs``, ``cat``, ``cont``, ``sparse`` and ``hash`` use only the
        remaining observations.
        """
        self._cat_data.drop(locs)
        self._cont_data.drop(locs)
        self._nobs = self._cat_data.shape[0]

    @property
    def sparse(self) -> csc_array:
        r"""
        Construct a sparse interaction matrix.

        Returns
        -------
        csc_array
            Dummy interaction constructed from the cartesian product of
            the categories and each of the continuous variables.

        Notes
        -----
        The number of columns is

        .. math::

            ncont \times \prod_{i=1}^{ncat} |c_i|

        where :math:`|c_i|` is the number distinct categories in column i.
        """
        if self.cat.shape[1] and self.cont.shape[1]:
            out = [
                category_continuous_interaction(
                    self.cat, self.cont[col], precondition=False
                )
                for col in self.cont
            ]
            return sp.hstack(out, format="csc")
        elif self.cat.shape[1]:
            return category_interaction(category_product(self.cat), precondition=False)
        elif self.cont.shape[1]:
            return csc_array(self._cont_data.ndarray)
        else:  # empty interaction
            return csc_array(empty((self._cat_data.shape[0], 0)))

    @property
    def hash(self) -> list[tuple[str, ...]]:
        """
        Construct a hash that is invariant to the order of the inputs.

        Returns
        -------
        list of tuple of str
            The sorted hashes of the interactions of the categorical variables
            with each continuous variable. Inputs that produce the same fit when
            they are used as regressors have the same hash.
        """
        # Sorted hashes of any categoricals
        hasher = Hasher()
        cat_hashes = []
        cat = self.cat
        for col in cat:
            hasher.update(
                memoryview(ascontiguousarray(self.cat[col].cat.codes.to_numpy().data))
            )
            cat_hashes.append(hasher.hexdigest())
            hasher.reset()
        sorted_hashes = tuple(sorted(cat_hashes))

        hashes = []
        cont = self.cont
        for col in cont:
            hasher.update(memoryview(ascontiguousarray(cont[col].to_numpy()).data))
            hashes.append((*sorted_hashes, hasher.hexdigest()))
            hasher.reset()

        return sorted(hashes)

    @staticmethod
    def from_frame(frame: DataFrame) -> Interaction:
        """
        Create an Interaction from the columns of a DataFrame.

        Parameters
        ----------
        frame : DataFrame
            Frame containing categorical and continuous variables. All
            categorical variables are passed to `cat` and all other
            variables are passed as `cont`.

        Returns
        -------
        Interaction
            Instance using the columns of frame.

        Raises
        ------
        TypeError
            If ``frame`` is not a DataFrame, or a column that is not categorical
            is not numeric.

        Examples
        --------
        >>> import numpy as np
        >>> from linearmodels.iv.absorbing import Interaction
        >>> import pandas as pd
        >>> rs = np.random.RandomState(0)
        >>> n = 100000
        >>> cats = pd.concat([pd.Series(pd.Categorical(rs.randint(i+2,size=n)))
        ...                  for i in range(4)], axis=1)
        >>> cats.columns = ["cat{0}".format(i) for i in range(4)]
        >>> columns = ["cont{0}".format(i) for i in range(6)]
        >>> cont = pd.DataFrame(rs.standard_normal((n, 6)), columns=columns)
        >>> frame = pd.concat([cats, cont], axis=1)
        >>> interact = Interaction.from_frame(frame)
        >>> interact.sparse.shape # Cart product of all cats, 5!, times ncont, 6
        (100000, 720)
        """
        return Interaction._from_frame(frame, "frame")

    @staticmethod
    def _from_frame(frame: DataFrame, name: str) -> Interaction:
        """from_frame, where name is used to refer to frame in error messages."""
        if not isinstance(frame, DataFrame):
            raise TypeError(f"{name} must be a DataFrame. Got {type(frame).__name__}.")
        cat_cols = [
            col for col in frame if isinstance(frame[col].dtype, CategoricalDtype)
        ]
        cont_cols = [col for col in frame if col not in cat_cols]
        for col in cont_cols:
            dtype = frame[col].dtype
            if not is_numeric_dtype(dtype):
                raise TypeError(
                    f"The column {col!r} of {name} has dtype {dtype}. The columns "
                    "that do not have a categorical dtype are treated as continuous "
                    "variables and must be numeric. Convert the columns that "
                    f"contain categories to categorical, for example using "
                    f"{name}[{col!r}].astype('category')."
                )
        # TODO: Bug in pandas-stubs
        #   https://github.com/pandas-dev/pandas-stubs/issues/97
        frame_cats = frame[cat_cols]
        frame_conts = frame[cont_cols]
        return Interaction(frame_cats, frame_conts, nobs=frame.shape[0])


InteractionVar = Union[DataFrame, Interaction]


class AbsorbingRegressor:
    """
    Construct a sparse matrix of regressors from components.

    Parameters
    ----------
    cat : DataFrame, optional
        Categorical variables (factors) to absorb.
    cont : DataFrame, optional
        Continuous variables to absorb.
    interactions : list of Interaction, optional
        Included interactions.
    weights : ndarray, optional
        Weights, if any. The regressors are multiplied by the square root of the
        weights.
    """

    def __init__(
        self,
        *,
        cat: DataFrame | None = None,
        cont: DataFrame | None = None,
        interactions: list[Interaction] | None = None,
        weights: linearmodels.typing.data.Float64Array | None = None,
    ):
        self._cat = cat
        self._cont = cont
        self._interactions = interactions
        self._weights = weights
        self._approx_rank: int | None = None

    @property
    def has_constant(self) -> bool:
        """Flag indicating whether the regressors have a constant equivalent."""
        return self._cat is not None and self._cat.shape[1] > 0

    @property
    def approx_rank(self) -> int:
        """Return the number of columns of the regressors, an upper bound of their rank."""
        if self._approx_rank is None:
            self._regressors()
        assert self._approx_rank is not None
        return self._approx_rank

    @property
    def hash(self) -> tuple[tuple[str, ...], ...]:
        """Return a hash of the regressors, which does not depend on their order."""
        hashes: list[tuple[str, ...]] = []
        hasher = Hasher()
        if self._cat is not None:
            for col in self._cat:
                hasher.update(
                    memoryview(
                        ascontiguousarray(self._cat[col].cat.codes.to_numpy()).data
                    )
                )
                hashes.append((hasher.hexdigest(),))
                hasher.reset()
        if self._cont is not None:
            for col in self._cont:
                hasher.update(
                    memoryview(ascontiguousarray(self._cont[col].to_numpy()).data)
                )
                hashes.append((hasher.hexdigest(),))
                hasher.reset()
        if self._interactions is not None:
            for interact in self._interactions:
                hashes.extend(interact.hash)
        # Add weight hash if provided
        if self._weights is not None:
            hasher = Hasher()
            hasher.update(memoryview(ascontiguousarray(self._weights.data)))
            hashes.append((hasher.hexdigest(),))
        return tuple(sorted(hashes))

    @property
    def regressors(self) -> csc_array:
        """Return the sparse matrix of regressors."""
        return self._regressors()

    def _regressors(self) -> csc_array:
        regressors = []

        if self._cat is not None and self._cat.shape[1] > 0:
            regressors.append(dummy_matrix(self._cat, precondition=False)[0])
        if self._cont is not None and self._cont.shape[1] > 0:
            regressors.append(csc_array(self._cont.astype(float).to_numpy()))
        if self._interactions is not None:
            regressors.extend([interact.sparse for interact in self._interactions])

        if regressors:
            regressor_mat = sp.hstack(regressors, format="csc")
            approx_rank = regressor_mat.shape[1]
            self._approx_rank = approx_rank
            if self._weights is not None:
                return (
                    diags_array(sqrt(self._weights.squeeze())).dot(regressor_mat)
                ).asformat("csc")
            return regressor_mat
        else:
            self._approx_rank = 0
            return csc_array(empty((0, 0)))


class AbsorbingLS:
    r"""
    Linear regression with high-dimensional effects.

    Parameters
    ----------
    dependent : array_like
        Dependent variable (nobs by 1).
    exog : array_like, optional
        Exogenous regressors (nobs by nexog).
    absorb : DataFrame or Interaction, optional
        The effects or continuous variables to absorb. When using a DataFrame,
        effects must be categorical variables. Other variable types are treated
        as continuous variables that should be absorbed, so they must be
        numeric, and a ``TypeError`` is raised for columns of strings, which
        have to be converted to categorical. When using an Interaction,
        variables in the `cat` argument are treated as effects and variables in
        the `cont` argument are treated as continuous.
    interactions : DataFrame, Interaction or list of DataFrame or Interaction, optional
        Interactions containing both categorical and continuous variables.  Each
        interaction is constructed using the Cartesian product of the categorical
        variables to produce the dummy, which are then separately interacted with
        each continuous variable. The columns of a DataFrame follow the rules of
        ``absorb``.
    weights : array_like, optional
        Observation weights used in estimation. The weights must be strictly positive
        and finite. They are rescaled to have a mean of 1. A NaN weight is missing, and
        its observation is dropped. The default is to weight all observations equally.
    drop_absorbed : bool, optional
        Flag indicating whether to drop absorbed variables. The default is False.

    Raises
    ------
    TypeError
        If ``absorb`` or ``interactions`` is not a DataFrame or an Interaction
        (a list of them for ``interactions``), or if a column that is not
        categorical is not numeric.
    ValueError
        If ``dependent``, ``exog``, ``absorb``, ``interactions`` and ``weights``
        have different numbers of observations or labels that do not match (see
        Notes), if all observations have a missing value, or if ``weights`` are
        not strictly positive and finite.

    See Also
    --------
    Interaction : Specify interactions between categorical and continuous variables.
    linearmodels.iv.model.IVLIML : Limited information ML and k-class estimation of
        IV models.
    linearmodels.iv.model.IV2SLS : Estimation of IV models using two-stage least
        squares.
    scipy.sparse.linalg.lsmr : Iterative solver for sparse least squares problems.

    Notes
    -----
    Capable of estimating models with millions of effects. The model is not a
    panel estimator. See :ref:`iv-absorbing-ls` for when to use it and when to
    use :class:`~linearmodels.panel.model.PanelOLS`.

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
    combined with Frish-Waugh-Lovell to orthogonalize x and y from z.

    z can contain factors that are perfectly linearly dependent. LSMR
    estimates a particular restricted set of parameters that captures the
    effect of non-redundant components in z.

    A variable in ``exog`` is absorbed if the effects explain all of it. The
    algorithms that absorb the effects stop with a relative error of about
    1e-8, so a variable is treated as absorbed if less than 1e-5 of its norm
    is left after the effects are removed, whatever its units. It is treated
    in the same way if it is collinear with the variables before it once the
    effects are removed. An ``AbsorbingEffectError`` is raised for such a
    variable. If ``drop_absorbed`` is True, the variable is dropped and an
    ``AbsorbingEffectWarning`` is raised, and the constant is never dropped.
    The norm includes the mean of the variable, so a variable with a large mean
    and little other variation, less than 1e-5 of its norm, is also treated as
    absorbed. Subtract its mean to avoid this.

    Rows are paired by position. An input that carries labels (a pandas
    Series or DataFrame with an index other than the default 0, 1, ..., n-1,
    or an xarray DataArray with coordinates) must use the same index, in the
    same order, as ``dependent``, so that a value cannot be attached to the
    wrong observation. A ``ValueError`` is raised if the index of ``exog``,
    ``absorb``, ``interactions`` or ``weights`` differs from the index of
    ``dependent``. Either reindex the input to the index of ``dependent``, or
    call ``to_numpy()`` to pair its values by position. NumPy arrays and
    pandas objects with a default index do not carry labels and are always
    paired by position. If ``dependent`` does not carry labels, the inputs
    that do must use the same index as the first of ``exog``, ``absorb``,
    ``interactions`` and ``weights`` that does. The same rule applies to
    ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.

    Examples
    --------
    Estimate a model by absorbing 2 categoricals and 2 continuous variables

    >>> import numpy as np
    >>> import pandas as pd
    >>> from linearmodels.iv import AbsorbingLS, Interaction
    >>> dep = np.random.standard_normal((20000,1))
    >>> exog = np.random.standard_normal((20000,2))
    >>> cats = pd.DataFrame({i: pd.Categorical(np.random.randint(1000, size=20000))
    ...                      for i in range(2)})
    >>> cont = pd.DataFrame({i+2: np.random.standard_normal(20000) for i in range(2)})
    >>> absorb = pd.concat([cats, cont], axis=1)
    >>> mod = AbsorbingLS(dep, exog, absorb=absorb)
    >>> res = mod.fit()

    Add interactions between the cartesian product of the categorical and
    each continuous variables

    >>> iaction = Interaction(cat=cats, cont=cont)
    >>> absorb = Interaction(cat=cats) # Other encoding of categoricals
    >>> mod = AbsorbingLS(dep, exog, absorb=absorb, interactions=iaction)
    """

    def __init__(
        self,
        dependent: linearmodels.typing.data.ArrayLike,
        exog: linearmodels.typing.data.ArrayLike | None = None,
        *,
        absorb: InteractionVar | None = None,
        interactions: InteractionVar | Iterable[InteractionVar] | None = None,
        weights: linearmodels.typing.data.ArrayLike | None = None,
        drop_absorbed: bool = False,
    ) -> None:
        self._dependent = IVData(dependent, "dependent")
        self._nobs = nobs = self._dependent.shape[0]
        self._exog = IVData(exog, "exog", nobs=self._nobs)
        self._absorb = absorb
        if isinstance(absorb, DataFrame):
            self._absorb_inter = Interaction._from_frame(absorb, "absorb")
        elif absorb is None:
            self._absorb_inter = Interaction(None, None, nobs)
        elif isinstance(absorb, Interaction):
            self._absorb_inter = absorb
        else:
            raise TypeError("absorb must ba a DataFrame or an Interaction")
        self._weights = weights
        self._is_weighted = False
        self._drop_absorbed = drop_absorbed
        self._weight_data: IVData
        self._check_weights()

        self._interactions = interactions
        self._interaction_list: list[Interaction] = []
        self._prepare_interactions()
        self._absorbed_dependent: DataFrame | None = None
        self._absorbed_exog: DataFrame | None = None
        # The arguments of fit that were used to absorb the variables
        self._absorb_args: tuple[Any, ...] | None = None

        self._check_shape()
        self._check_alignment()
        self._original_index = self._dependent.pandas.index
        self._drop_locs = self._drop_missing()
        self._columns = self._exog.cols
        self._index = self._dependent.rows
        self._method = "Absorbing LS"

        self._const_col: int | None = 0
        self._has_constant = False
        self._has_constant_exog = self._check_constant()
        # Location of the constant among the regressors, which changes if some are dropped
        self._const_loc: int | None = self._const_col
        self._constant_absorbed = False
        self._num_params = 0
        self._regressors: csc_array | None = None
        self._regressors_hash: tuple[tuple[str, ...], ...] | None = None

    def _drop_missing(self) -> linearmodels.typing.data.BoolArray:
        missing = require(self.dependent.isnull.to_numpy(), requirements="W")
        missing |= self.exog.isnull.to_numpy()
        missing |= self._weight_data.isnull.to_numpy()
        missing |= self._absorb_inter.cat.isnull().any(axis=1).to_numpy()
        missing |= self._absorb_inter.cont.isnull().any(axis=1).to_numpy()
        for interact in self._interaction_list:
            missing |= interact.isnull.to_numpy()
        if npany(missing):
            if missing.all():
                raise ValueError(
                    "All observations contain missing data. Model cannot be estimated."
                )
            # drop modifies an Interaction, which the caller may use again
            self._absorb_inter = deepcopy(self._absorb_inter)
            self._interaction_list = deepcopy(self._interaction_list)
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self._absorb_inter.drop(missing)
            for interact in self._interaction_list:
                interact.drop(missing)
            # Weights have mean 1 in the sample used to estimate the model
            weights = self._weight_data.ndarray[~missing]
            self._weight_data = IVData(normalize_weights(weights), "weights")
            self._nobs = self.dependent.shape[0]
        missing_warning(missing, stacklevel=4)
        return missing

    def _check_constant(self) -> bool:
        col_delta = ptp(self.exog.ndarray, 0)
        has_constant = npany(col_delta == 0)
        self._const_col = int(where(col_delta == 0)[0][0]) if has_constant else None
        return bool(has_constant)

    def _check_weights(self) -> None:
        if self._weights is None:
            nobs = self._dependent.shape[0]
            self._is_weighted = False
            self._weight_data = IVData(ones(nobs), "weights")
        else:
            self._is_weighted = True
            weights = IVData(self._weights, var_name="weights").ndarray
            check_weights(weights)
            weights = normalize_weights(weights)
            self._weight_data = IVData(weights, var_name="weights", nobs=self._nobs)

    def _check_shape(self) -> None:
        nobs = self._nobs
        if self._absorb is not None:
            if self._absorb_inter.nobs != nobs:
                raise ValueError(
                    "absorb and dependent have different number of observations"
                )
        for interact in self._interaction_list:
            if interact.nobs != nobs:
                raise ValueError(
                    "interactions ({}) and dependent have different number of "
                    "observations".format(str(interact))
                )

    def _check_alignment(self) -> None:
        """Check that inputs with labels use the labels of dependent, or the first."""
        inputs: list[tuple[str, object]] = [("exog", self._exog)]
        interactions: list[tuple[str, Interaction]] = [("absorb", self._absorb_inter)]
        single = isinstance(self._interactions, (DataFrame, Interaction))
        for i, interact in enumerate(self._interaction_list):
            name = "interactions" if single else f"interactions[{i}]"
            interactions.append((name, interact))
        for name, interact in interactions:
            inputs.extend([(name, interact.cat), (name, interact.cont)])
        inputs.append(("weights", self._weights))
        # If dependent has no labels, the first input with labels sets the labels
        self._labels, self._labels_name = reference_labels(
            row_labels(self._dependent), inputs
        )
        check_alignment(self._labels, inputs, reference=self._labels_name)

    @property
    def absorbed_dependent(self) -> DataFrame:
        """
        Dependent variable with effects absorbed.

        Returns
        -------
        DataFrame
            Dependent after effects have been absorbed.

        Raises
        ------
        RuntimeError
            If called before `fit` has been used once
        """
        if self._absorbed_dependent is not None:
            return self._absorbed_dependent
        raise RuntimeError(
            "fit must be called once before absorbed_dependent is available"
        )

    @property
    def absorbed_exog(self) -> DataFrame:
        """
        Exogenous variables with effects absorbed.

        Returns
        -------
        DataFrame
            Exogenous after effects have been absorbed.

        Raises
        ------
        RuntimeError
            If called before `fit` has been used once
        """
        if self._absorbed_exog is not None:
            return self._absorbed_exog
        raise RuntimeError("fit must be called once before absorbed_exog is available")

    @property
    def weights(self) -> IVData:
        """Return the weights, which have a mean of 1."""
        return self._weight_data

    @property
    def dependent(self) -> IVData:
        """Return the dependent variable."""
        return self._dependent

    @property
    def exog(self) -> IVData:
        """Return the exogenous regressors."""
        return self._exog

    @property
    def has_constant(self) -> bool:
        """Flag indicating whether the model has a constant or equivalent."""
        return self._has_constant

    @property
    def instruments(self) -> IVData:
        """Return the instruments, of which there are none."""
        return IVData(None, "instrument", nobs=self._dependent.shape[0])

    def _prepare_interactions(self) -> None:
        if self._interactions is None:
            return
        elif isinstance(self._interactions, DataFrame):
            self._interaction_list = [
                Interaction._from_frame(self._interactions, "interactions")
            ]
        elif isinstance(self._interactions, Interaction):
            self._interaction_list = [self._interactions]
        else:
            for interact in self._interactions:
                if isinstance(interact, DataFrame):
                    self._interaction_list.append(
                        Interaction._from_frame(interact, "interactions")
                    )
                elif isinstance(interact, Interaction):
                    self._interaction_list.append(interact)
                else:
                    raise TypeError(
                        "interactions must contain DataFrames or Interactions"
                    )

    def _first_time_fit(
        self,
        use_cache: bool,
        absorb_options: (
            dict[
                str,
                bool
                | float
                | str
                | linearmodels.typing.data.ArrayLike
                | dict[str, Any]
                | None,
            ]
            | None
        ),
        method: str,
    ) -> None:
        # Absorbing is repeated when fit uses other options, so start from the
        # state of a model that has not been fit
        absorb_args = _absorb_args(method, use_cache, absorb_options)
        self._num_params = 0
        self._columns = self._exog.cols
        weights = (
            cast("linearmodels.typing.data.Float64Array", self.weights.ndarray)
            if self._is_weighted
            else None
        )

        use_hdfe = weights is None and method in ("auto", "hdfe")
        use_hdfe = use_hdfe and not self._absorb_inter.cont.shape[1]
        use_hdfe = use_hdfe and not self._interaction_list

        if not use_hdfe and method == "hdfe":
            raise RuntimeError(
                "HDFE has been set as the method but the model cannot be estimated "
                "using HDFE. HDFE requires that the model is unweighted and that the "
                "absorbed regressors include only fixed effects (dummy variables)."
            )
        areg = AbsorbingRegressor(
            cat=self._absorb_inter.cat,
            cont=self._absorb_inter.cont,
            interactions=self._interaction_list,
            weights=weights,
        )
        areg_constant = areg.has_constant
        self._regressors = areg.regressors
        self._num_params += areg.approx_rank
        # Do not double count intercept-like terms
        self._has_constant = self._has_constant_exog or areg_constant
        self._num_params -= min(self._has_constant_exog, areg_constant)
        self._regressors_hash = areg.hash
        self._constant_absorbed = self._has_constant_exog and areg_constant

        dep = self._dependent.ndarray.astype(float, copy=False)
        exog = cast("linearmodels.typing.data.Float64Array", self._exog.ndarray)

        root_w = sqrt(self._weight_data.ndarray.astype(float, copy=False))
        dep = root_w * dep
        exog = root_w * exog
        denom = root_w.T @ root_w
        mu_dep = (root_w.T @ dep) / denom
        mu_exog = (root_w.T @ exog) / denom

        absorb_options = {} if absorb_options is None else dict(absorb_options)
        assert isinstance(self._regressors, csc_array)
        if self._regressors.shape[1] > 0:
            _check_absorb_options(absorb_options, use_hdfe, method)
            if use_hdfe:
                from pyhdfe import create  # noqa: PLC0415

                absorb_options["drop_singletons"] = False
                algo = create(self._absorb_inter.cat, **absorb_options)
                dep_exog = column_stack((dep, exog)).astype(float, copy=False)
                resids = algo.residualize(dep_exog)
                dep_resid = resids[:, :1]
                exog_resid = resids[:, 1:]
            else:
                self._regressors = preconditioner(self._regressors)[0]
                dep_exog = column_stack((dep, exog)).astype(float, copy=False)
                resid = lsmr_annihilate(
                    self._regressors,
                    dep_exog,
                    use_cache,
                    self._regressors_hash,
                    **absorb_options,
                )
                dep_resid = resid[:, :1]
                exog_resid = resid[:, 1:]
        else:
            dep_resid = dep
            exog_resid = exog

        if self._constant_absorbed:
            dep_resid += root_w * mu_dep
            exog_resid += root_w * mu_exog

        ncol = exog_resid.shape[1]
        retain = list(range(ncol))
        if not self._drop_absorbed:
            check_absorbed(exog_resid, self.exog.cols, exog)
        else:
            retain = not_absorbed(
                exog_resid, self._constant_absorbed, self._const_col, exog
            )
            if not retain:
                raise ValueError(
                    "All columns in exog have been fully absorbed by the "
                    "included effects. This model cannot be estimated."
                )
            elif len(retain) < ncol:
                drop = set(range(ncol)).difference(retain)
                dropped = ", ".join([str(self.exog.cols[i]) for i in drop])
                warnings.warn(
                    absorbing_warn_msg.format(absorbed_variables=dropped),
                    AbsorbingEffectWarning,
                    stacklevel=3,
                )

            exog_resid = exog_resid[:, retain]
            self._columns = [self._columns[i] for i in retain]

        # The location of the constant among the retained regressors, which is not
        # the location in exog if a regressor before it was dropped
        const_col = self._const_col
        self._const_loc = retain.index(const_col) if const_col in retain else None
        self._absorbed_dependent = DataFrame(
            dep_resid,
            index=self._dependent.pandas.index,
            columns=self._dependent.pandas.columns,
        )
        self._absorbed_exog = DataFrame(
            exog_resid, index=self._exog.pandas.index, columns=self._columns
        )
        self._num_params += exog_resid.shape[1]
        self._absorb_args = absorb_args

    def _discard_stale_absorption(
        self,
        method: str,
        use_cache: bool,
        absorb_options: dict[str, Any] | None,
        lsmr_options: dict[str, float | bool] | None,
    ) -> None:
        """Discard the absorbed variables if fit uses other options to absorb."""
        if absorb_options is not None and lsmr_options is not None:
            return  # fit raises an error
        options = lsmr_options if absorb_options is None else absorb_options
        if _absorb_args(method, use_cache, options) != self._absorb_args:
            self._absorbed_dependent = None
            self._absorbed_exog = None

    def fit(
        self,
        *,
        cov_type: str = "robust",
        debiased: bool = False,
        method: str = "auto",
        absorb_options: (
            dict[
                str,
                bool
                | float
                | str
                | linearmodels.typing.data.ArrayLike
                | dict[str, Any]
                | None,
            ]
            | None
        ) = None,
        use_cache: bool = True,
        lsmr_options: dict[str, float | bool] | None = None,
        **cov_config: Any,
    ) -> AbsorbingLSResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic", "homo", "homoskedasticcovariance" -
              Classic homoskedastic inference
            * "robust", "heteroskedastic", "hccm", "heteroskedasticcovariance" -
              Heteroskedasticity robust inference
            * "kernel", "kernelcovariance" - Heteroskedasticity and autocorrelation
              robust inference
            * "clustered", "one-way", "onewayclusteredcovariance" - Cluster
              dependent inference, which is heteroskedasticity robust. It is
              one-way unless ``clusters`` has two columns.

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is False.
        method : {"auto", "hdfe", "lsmr"}, optional
            The algorithm that absorbs the variables. The names are not case
            sensitive, and a ValueError is raised for any other value. One of:

            * "auto" - (Default). Use HDFE when applicable and fallback to LSMR.
            * "lsmr" - Force LSMR.
            * "hdfe" - Force HDFE. Raises RuntimeError if the model contains
              continuous variables or continuous-binary interactions to absorb or
              if the model is weighted.

        absorb_options : dict, optional
            Dictionary of options to pass to the absorber. Passed to either
            scipy.sparse.linalg.lsmr or pyhdfe.create depending on the method used
            to absorb the absorbed regressors. The options have to be those of
            the algorithm that is used, which is HDFE if method is "auto" and the
            model can be absorbed using it. A ValueError is raised for options
            of LSMR if the variables are absorbed using HDFE, and for options
            of pyhdfe.create if they are absorbed using LSMR.
        use_cache : bool, optional
            Flag indicating whether the variables, once purged from the
            absorbed variables and interactions, should be stored in the cache,
            and retrieved if available. Cache can dramatically speed up
            re-fitting large models when the set of absorbed variables and
            interactions are identical. The default is True.
        lsmr_options : dict, optional
            Options to pass to scipy.sparse.linalg.lsmr. If they are not empty,
            method "auto" absorbs the variables using LSMR, and method "hdfe"
            raises a ValueError.

            .. deprecated:: 4.17

               Use absorb_options with method="lsmr" to pass options.
               ``lsmr_options`` will be removed in linearmodels 8.0.

        **cov_config : dict
            Additional parameters to pass to the covariance estimator. The keys that
            are used depend on ``cov_type``. "unadjusted" and "robust" do not use any.
            "kernel" uses ``kernel`` (str), the name of the kernel, which is "bartlett"
            if it is not provided, and ``bandwidth`` (int), which is selected from the
            data if it is not provided. The names of the kernels are not case
            sensitive. "clustered" uses ``clusters`` (array_like), the cluster
            assignment of each observation, with one or two columns. The keys
            ``center``, ``kappa`` and ``extra_df`` are accepted and ignored: the
            absorbed effects set the degrees of freedom. See
            :class:`~linearmodels.iv.covariance.HomoskedasticCovariance`,
            :class:`~linearmodels.iv.covariance.HeteroskedasticCovariance`,
            :class:`~linearmodels.iv.covariance.KernelCovariance` and
            :class:`~linearmodels.iv.covariance.ClusteredCovariance` for the details.

        Returns
        -------
        AbsorbingLSResults
            Results container.

        Raises
        ------
        ValueError
            If ``method`` is not supported, if ``absorb_options`` contains options
            of the algorithm that is not used, if ``lsmr_options`` is used with
            ``absorb_options`` or with ``method="hdfe"``, if all columns in ``exog``
            are absorbed and ``drop_absorbed`` is True, if ``debiased`` is True and
            the model has no degrees of freedom left, or if the labels of
            ``clusters`` do not match the labels of the other inputs.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.
        RuntimeError
            If ``method`` is "hdfe" and the model cannot be absorbed using HDFE.
        Exception
            An ``AbsorbingEffectError``, which is defined in
            ``linearmodels.panel.utility``, if a variable in ``exog`` is absorbed by
            the effects and ``drop_absorbed`` is False.
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.

        Warns
        -----
        AbsorbingEffectWarning
            If a variable in ``exog`` is absorbed by the effects and
            ``drop_absorbed`` is True. The variable is dropped.
        FutureWarning
            If ``lsmr_options`` is used.

        See Also
        --------
        linearmodels.iv.covariance.HomoskedasticCovariance : Homoskedastic covariance.
        linearmodels.iv.covariance.HeteroskedasticCovariance : Heteroskedasticity
            robust covariance.
        linearmodels.iv.covariance.KernelCovariance : Kernel (HAC) covariance.
        linearmodels.iv.covariance.ClusteredCovariance : Clustered covariance.

        Notes
        -----
        Additional covariance parameters depend on specific covariance used.
        See the docstring of the specific covariance estimator for a list of
        supported options. Defaults are used if no covariance configuration
        is provided.

        Observations with a missing value in ``dependent``, ``exog``,
        ``weights``, ``absorb`` or ``interactions`` are dropped when the model
        is created. When ``cov_type`` is "clustered", ``clusters`` must have
        one row for each of the remaining observations, so the rows of
        ``clusters`` that belong to dropped observations must be dropped too.

        If use_cache is True, then variables are hashed based on their
        contents using either a 64-bit value (if xxhash is installed) or
        a 256-bit value. This allows variables to be reused in different
        models if the set of absorbing variables and interactions is held
        constant. The options used to absorb the variables are part of
        what identifies them.

        If ``debiased`` is True, the covariance estimator is adjusted for the
        degrees of freedom that the model uses: the k regressors in ``exog`` that
        are estimated and the d degrees of freedom of the absorbed terms. They are
        the dummy variables of the effects, with one less if ``exog`` has a
        constant and one less for each effect after the first, the absorbed
        continuous variables and the interactions, which is ``df_absorbed``. The
        unadjusted, robust and kernel covariances are multiplied by n / (n - k - d),
        and the clustered covariance by g (n - 1) / ((g - 1) (n - k - d)) where g
        is the number of clusters. The result is that of the regression of the
        dependent variable on ``exog`` and the dummy variables of the effects. It
        is also that of ``PanelOLS(debiased=True)`` with the same effects, using
        ``group_debias=True`` for the clustered covariance. The exception is a
        clustered covariance when there is a single categorical effect and each of
        its categories is in one cluster of every cluster variable: its degrees of
        freedom are not counted, since the effect is nested in the clusters, as in
        ``PanelOLS`` and Stata. Unlike ``PanelOLS``, nothing is adjusted if
        ``debiased`` is False. The t and F distributions of the results use
        ``df_resid`` whatever the covariance estimator. A ``ValueError`` is raised if
        ``debiased`` is True and the model has no degrees of freedom left.

        The variables are absorbed when fit is called for the first time. A
        later call uses the same absorbed variables if ``method``,
        ``absorb_options`` and ``use_cache`` have the same values, so that
        changing only the covariance estimator is fast. If any of them differs,
        the variables are absorbed again using the new values.
        """
        check_cov_config(cov_config, cov_type, _LS_COV_KEYS, KERNEL_LOOKUP)
        method = _check_method(method)
        if lsmr_options and absorb_options is None:
            # The options are options of LSMR, so they can not be passed to HDFE
            if method == "hdfe":
                raise ValueError(
                    "lsmr_options are options of scipy.sparse.linalg.lsmr and "
                    'cannot be used with method="hdfe". Use method="lsmr" or '
                    'method="auto", which uses LSMR if lsmr_options are given.'
                )
            method = "lsmr" if method == "auto" else method
        self._discard_stale_absorption(method, use_cache, absorb_options, lsmr_options)
        if lsmr_options is not None:
            if absorb_options is not None:
                raise ValueError("absorb_options cannot be used with lsmr_options")
            warnings.warn(
                "lsmr_options is deprecated and will be removed in linearmodels "
                '8.0. Use absorb_options with method="lsmr", which passes the options '
                "to scipy.sparse.linalg.lsmr.",
                FutureWarning,
                stacklevel=2,
            )
            absorb_options = dict(lsmr_options.items())
        # Clusters are for the observations that are used to estimate the model
        labels = None if self._labels is None else self._labels[~self._drop_locs]
        check_alignment(
            labels,
            {"clusters": cov_config.get("clusters")},
            reference=self._labels_name,
        )
        if self._absorbed_dependent is None:
            self._first_time_fit(use_cache, absorb_options, method)

        exog_resid = self.absorbed_exog.to_numpy()
        dep_resid = self.absorbed_dependent.to_numpy()
        if self._exog.shape[1] == 0:
            params = empty((0, 1))
        else:
            params = lstsq(exog_resid, dep_resid, rcond=None)[0]

        cov_estimator = lookup_option(COVARIANCE_ESTIMATORS, cov_type, "cov_type")
        cov_config["debiased"] = debiased
        cov_config["kappa"] = 0.0
        cov_config_copy = dict(cov_config.items())
        cov_config_copy.pop("center", None)
        if debiased:
            cov_config_copy["extra_df"] = self._absorbed_df(
                cov_estimator, cov_config.get("clusters")
            )
        cov_estimator_inst = cov_estimator(
            exog_resid, dep_resid, exog_resid, params, **cov_config_copy
        )

        results = {"kappa": 0.0, "liml_kappa": 0.0}
        pe = self._post_estimation(params, cov_estimator_inst, cov_type)
        results.update(pe)
        results["df_model"] = self._num_params

        return AbsorbingLSResults(results, self)

    def _absorbed_df(
        self,
        cov_estimator: type[
            HomoskedasticCovariance
            | HeteroskedasticCovariance
            | KernelCovariance
            | ClusteredCovariance
        ],
        clusters: linearmodels.typing.data.ArrayLike | None,
    ) -> int:
        """
        Degrees of freedom of the absorbed terms that a debiased covariance counts.

        Parameters
        ----------
        cov_estimator : type
            The class of the covariance estimator.
        clusters : array_like
            The clusters that are passed to fit, if any.

        Returns
        -------
        int
            The degrees of freedom of the absorbed terms: the dummy variables of the
            effects, the absorbed continuous variables and the interactions, less the
            overlap with a constant in exog. They are the same as df_absorbed, except
            that a clustered covariance does not count the degrees of freedom of a
            single categorical effect that is nested in all of the clusters, as
            PanelOLS does.
        """
        assert isinstance(self._absorbed_exog, DataFrame)
        absorbed = self._num_params - self._absorbed_exog.shape[1]
        cat = self._absorb_inter.cat
        if (
            cov_estimator is not ClusteredCovariance
            or clusters is None
            or cat.shape[1] != 1
        ):
            return absorbed
        cluster_array = asarray(clusters).squeeze()
        if cluster_array.shape[0] != self._nobs:
            return absorbed  # The covariance estimator raises an error
        columns = [cluster_array] if cluster_array.ndim == 1 else list(cluster_array.T)
        codes = asarray(cat.iloc[:, 0].cat.codes).astype(int64)
        if not all(_is_nested(codes, column) for column in columns):
            return absorbed
        # The dummy variables of the effect, less one if the constant takes one
        effect_df = unique(codes).shape[0] - int(self._has_constant_exog)
        return absorbed - effect_df

    def resids(
        self, params: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute model residuals.

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1).

        Returns
        -------
        ndarray
            Model residuals.
        """
        resids = self.wresids(params)
        return resids / sqrt(self.weights.ndarray)

    def wresids(
        self, params: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute weighted model residuals.

        Parameters
        ----------
        params : ndarray
            Model parameters (nvar by 1).

        Returns
        -------
        ndarray
            Weighted model residuals.

        Notes
        -----
        Uses weighted versions of data instead of raw data.  Identical to
        resids if all weights are unity.
        """
        assert isinstance(self._absorbed_dependent, DataFrame)
        assert isinstance(self._absorbed_exog, DataFrame)
        return (
            self._absorbed_dependent.to_numpy()
            - self._absorbed_exog.to_numpy() @ params
        )

    def _f_statistic(
        self,
        params: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        debiased: bool,
    ) -> WaldTestStatistic | InvalidTestStatistic:
        resid_df = self._nobs - self._num_params

        return f_statistic(params, cov, debiased, resid_df, self._const_loc)

    def _retained_exog(self) -> linearmodels.typing.data.Float64Array:
        """The regressors that were estimated, in the units of exog."""
        locs = [self._exog.cols.index(col) for col in self._columns]
        return cast(
            "linearmodels.typing.data.Float64Array", self._exog.ndarray[:, locs]
        )

    def _post_estimation(
        self,
        params: linearmodels.typing.data.Float64Array,
        cov_estimator: (
            HomoskedasticCovariance
            | HeteroskedasticCovariance
            | KernelCovariance
            | ClusteredCovariance
        ),
        cov_type: str,
    ) -> dict[str, Any]:
        columns = self._columns
        index = self._index
        eps = self.resids(params)
        fitted_values = self._dependent.ndarray - eps
        fitted = DataFrameWrapper(
            fitted_values,
            index=self._dependent.rows,
            columns=["fitted_values"],
        )
        assert isinstance(self._absorbed_dependent, DataFrame)
        # The part of the fitted values that is not explained by the regressors
        absorbed_effects = DataFrameWrapper(
            fitted_values - self._retained_exog() @ params,
            columns=["absorbed_effects"],
            index=self._dependent.rows,
        )

        weps = self.wresids(params)
        cov = cov_estimator.cov
        debiased = cov_estimator.debiased

        residual_ss = (weps.T @ weps)[0, 0]

        w = self.weights.ndarray
        root_w = sqrt(w)
        e = self._dependent.ndarray * root_w
        # The size of the dependent variable, before the means and the effects are removed
        y_ss = float(squeeze(e.T @ e))
        if self.has_constant:
            e = e - root_w * average(self._dependent.ndarray, weights=w)

        total_ss = float(squeeze(e.T @ e))
        nobs = e.shape[0]
        r2 = _non_negative(r_squared(float(residual_ss), total_ss, y_ss, nobs))

        e = self._absorbed_dependent.to_numpy()  # already scaled by root_w
        # If absorbing contains a constant, but exog does not, no need to demean
        assert isinstance(self._absorbed_exog, DataFrame)
        if self._const_loc is not None:
            col = self._const_loc
            x = self._absorbed_exog.to_numpy()[:, col : col + 1]
            mu = (lstsq(x, e, rcond=None)[0]).squeeze()
            e = e - x * mu

        aborbed_total_ss = float(squeeze(e.T @ e))
        r2_absorbed = _non_negative(
            r_squared(float(residual_ss), aborbed_total_ss, y_ss, nobs)
        )

        fstat = self._f_statistic(params, cov, debiased)
        out = {
            "params": Series(params.squeeze(), columns, name="parameter"),
            "eps": SeriesWrapper(eps.squeeze(), index=index, name="residual"),
            "weps": SeriesWrapper(
                weps.squeeze(), index=index, name="weighted residual"
            ),
            "cov": DataFrame(cov, columns=columns, index=columns),
            "s2": float(squeeze(cov_estimator.s2)),
            "debiased": debiased,
            "residual_ss": float(residual_ss),
            "total_ss": float(total_ss),
            "r2": float(r2),
            "fstat": fstat,
            "vars": columns,
            "instruments": [],
            "cov_config": cov_estimator.config,
            "cov_type": cov_type,
            "method": self._method,
            "cov_estimator": cov_estimator,
            "fitted": fitted,
            "original_index": self._original_index,
            "absorbed_effects": absorbed_effects,
            "absorbed_r2": r2_absorbed,
        }

        return out
