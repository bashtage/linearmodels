"""Utilities for panel models: dummy variables, singleton removal and test data."""

from __future__ import annotations

from linearmodels.compat.pandas import ANNUAL_FREQ
from linearmodels.compat.scipy import coo_array, csc_array, csr_array

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, TypeVar, cast

import numpy as np
import numpy.random
from pandas import CategoricalDtype, DataFrame, concat, date_range
import scipy.sparse as sp

from linearmodels.shared.utility import panel_to_frame
import linearmodels.typing.data

try:
    from linearmodels.panel._utility import _drop_singletons

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


__all__ = [
    "AbsorbingEffectError",
    "AbsorbingEffectWarning",
    "_drop_singletons",
    "_py_drop_singletons",
    "_remove_node",
    "check_absorbed",
    "dummy_matrix",
    "generate_panel_data",
    "in_2core_graph",
    "in_2core_graph_slow",
    "not_absorbed",
    "preconditioner",
]


class AbsorbingEffectError(Exception):
    """Error raised when the effects absorb one or more of the variables."""


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
    """Warning issued when absorbed variables are dropped from the regression."""


absorbing_warn_msg = """
Variables have been fully absorbed and have removed from the regression:

{absorbed_variables}
"""

SparseArray = TypeVar("SparseArray", csr_array, csc_array, coo_array)
SparseOrDense = TypeVar(
    "SparseOrDense",
    linearmodels.typing.data.Float64Array,
    csc_array,
    csr_array,
    coo_array,
)


def preconditioner(
    d: SparseOrDense, *, copy: bool = False
) -> tuple[SparseOrDense, linearmodels.typing.data.Float64Array]:
    """
    Scale the columns of an array to have unit 2-norm.

    Parameters
    ----------
    d : array_like or csc_array or csr_array or coo_array
        Array to precondition. Dense input is converted with ``np.asarray``.
    copy : bool, optional
        Flag indicating whether the operation should be in-place, if possible.
        If True, a new array is always returned. The default is False.

    Returns
    -------
    d : ndarray or csc_array or csr_array or coo_array
        Array with the same type as the input array. If copy is False, and d is
        an ndarray or a csc_array, then the operation is inplace.
    cond : ndarray
        Conditioning numbers with shape (nvar,), defined as the square root of
        the 2-norm of each column.
    """
    # Dense path
    if not sp.issparse(d):
        klass: Any = None if type(d) is np.ndarray else d.__class__
        d_id = id(d)
        d = np.asarray(d)
        if id(d) == d_id or copy:
            d = d.copy()
        cond = cast("linearmodels.typing.data.Float64Array", np.sqrt((d**2).sum(0)))
        d /= cond
        if klass is not None:
            d = d.view(klass)
        return d, cond

    klass = None
    if not isinstance(d, csc_array):
        klass = d.__class__
        d_csc = csc_array(d)
    else:
        assert isinstance(d, csc_array)
        d_csc = d
        if copy:
            d_csc = d.copy()

    cond = cast(
        "linearmodels.typing.data.Float64Array",
        np.asarray(np.sqrt(d_csc.multiply(d_csc).sum(0))).ravel(),
    )
    locs: np.ndarray = np.zeros_like(d_csc.indices)
    locs[d_csc.indptr[1:-1]] = 1
    locs = np.cumsum(locs)
    d_csc.data /= np.take(cond, locs)
    if klass is not None:
        d_out = klass(d_csc)
    else:
        d_out = d_csc

    return d_out, cond


def dummy_matrix(
    cats: linearmodels.typing.data.ArrayLike,
    *,
    output_format: Literal["csc", "csr", "coo"] = "csc",
    drop: Literal["first", "last"] = "first",
    drop_all: bool = False,
    precondition: bool = True,
) -> tuple[
    csc_array | csr_array | coo_array,
    linearmodels.typing.data.Float64Array,
]:
    """
    Construct a sparse matrix of dummy variables from category codes.

    Parameters
    ----------
    cats : DataFrame or array_like
        Array containing the category codes of pandas categoricals with shape
        (nobs, ncats). A DataFrame must only contain categoricals. An array
        is converted with ``np.asarray`` and must have 2 dimensions.
    output_format : {"csc", "csr", "coo"}, optional
        Format of the sparse output. The default is "csc". The formats are:

        * "csc" - sparse array in compressed column form (csc_array)
        * "csr" - sparse array in compressed row form (csr_array)
        * "coo" - sparse array in coordinate form (coo_array)

    drop : {"first", "last"}, optional
        Exclude either the first or last category. This only applies when
        cats contains more than one column, unless `drop_all` is True. The
        default is "first".
    drop_all : bool, optional
        Flag indicating whether all sets of dummies should exclude one
        category. The default is False.
    precondition : bool, optional
        Flag indicating whether the columns of the dummy matrix should be
        preconditioned to have unit 2-norm. The default is True.

    Returns
    -------
    dummies : csc_array or csr_array or coo_array
        Sparse array, with the format set by `output_format`, of size
        nobs by ndummies containing the dummy variable values.
    cond : ndarray
        Conditioning number of each column.

    Raises
    ------
    TypeError
        If cats is a DataFrame with a column that is not categorical.
    ValueError
        If cats is an array that does not have 2 dimensions, or if
        `output_format` is not one of the supported formats.
    """
    if isinstance(cats, DataFrame):
        for col in cats:
            if not isinstance(cats[col].dtype, CategoricalDtype):
                raise TypeError(
                    "cats must only contain categorical columns, but the column "
                    f"{col!r} has dtype {cats[col].dtype}. Pass the category codes "
                    "as an array to use other columns."
                )
        codes = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    else:
        codes = np.asarray(cats)
        if codes.ndim != 2:
            raise ValueError(
                "cats must be a DataFrame or a 2-d array with a column for each "
                f"set of categories, but it has {codes.ndim} dimension(s)."
            )

    data: dict[str, list[np.ndarray]] = defaultdict(list)
    total_dummies = 0
    nobs, ncats = codes.shape
    for i in range(ncats):
        rows = np.arange(nobs)
        ucats, inverse = np.unique(codes[:, i], return_inverse=True)
        ncategories = len(ucats)
        bits = min(
            i for i in (8, 16, 32, 64) if i - 1 > np.log2(ncategories + total_dummies)
        )
        replacements = np.arange(ncategories, dtype=f"int{bits:d}")
        cols = replacements[inverse]
        if i == 0 and not drop_all:
            retain = np.arange(nobs)
        elif drop.lower() == "first":
            # remove first
            retain = cols != 0
        elif drop.lower() == "last":
            # remove last
            retain = cols != (ncategories - 1)
        else:
            raise ValueError(f"Unknown drop option: {drop}")
        rows = rows[retain]
        col_adj = -1 if (drop.lower() == "first" and i > 0) else 0
        cols = cols[retain] + total_dummies + col_adj
        values = np.ones(rows.shape)
        data["values"].append(values)
        data["rows"].append(rows)
        data["cols"].append(cols)
        total_dummies += ncategories - (i > 0)

    if output_format == "csc":
        fmt: Callable[..., Any] = csc_array
    elif output_format == "csr":
        fmt = csr_array
    elif output_format == "coo":
        fmt = coo_array
    else:
        raise ValueError(f"Unknown format: {output_format}")
    out = fmt(
        (
            np.concatenate(data["values"]),
            (np.concatenate(data["rows"]), np.concatenate(data["cols"])),
        )
    )
    if precondition:
        out, cond = preconditioner(out, copy=False)
    else:
        cond = np.ones(out.shape[1])

    return out, cond


def _remove_node(
    node: int,
    meta: linearmodels.typing.data.IntArray,
    orig_dest: linearmodels.typing.data.IntArray,
) -> tuple[int, int]:
    """
    Remove a node that has one link and step to its neighbor.

    Parameters
    ----------
    node : int
        ID of the node to remove.
    meta : ndarray
        Array with rows containing node, count, and address where
        address is used to find the first occurrence in orig_dest.
    orig_dest : ndarray
        Array with rows containing origin and destination nodes.

    Returns
    -------
    next_node : int
        ID of the next node in the branch.
    next_count : int
        Count of the next node in the branch.

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


def _py_drop_singletons(
    meta: linearmodels.typing.data.IntArray,
    orig_dest: linearmodels.typing.data.IntArray,
) -> None:
    """
    Loop through the nodes and recursively drop singleton chains.

    Parameters
    ----------
    meta : ndarray
        Array with rows containing node, count, and address where
        address is used to find the first occurrence in orig_dest.
    orig_dest : ndarray
        Array with rows containing origin and destination nodes.
    """
    for i in range(meta.shape[0]):
        if meta[i, 1] == 1:
            next_node = i
            next_count = 1
            while next_count == 1:
                # Follow singleton chains
                next_node, next_count = _remove_node(next_node, meta, orig_dest)


if not HAS_CYTHON:
    _drop_singletons = _py_drop_singletons


def in_2core_graph(
    cats: DataFrame | np.ndarray,
) -> linearmodels.typing.data.BoolArray:
    """
    Find the observations that are in the 2-core of the graph.

    Parameters
    ----------
    cats : DataFrame or ndarray
        Array containing the category codes of pandas categoricals
        with shape (nobs, ncats). A DataFrame must only contain categoricals.

    Returns
    -------
    ndarray
        Boolean array that marks non-singleton entries as True.
    """
    if isinstance(cats, DataFrame):
        cats = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    if cats.shape[1] == 1:
        # Fast, simple path
        ucats, counts = np.unique(cats, return_counts=True)
        retain = ucats[counts > 1]
        return np.isin(cats, retain).ravel()

    nobs, ncats = cats.shape
    zero_cats_lst = []
    # Switch to 0 based indexing
    for col in range(ncats):
        u, inv = np.unique(cats[:, col], return_inverse=True)
        zero_cats_lst.append(np.arange(u.shape[0])[inv])
    zero_cats = np.column_stack(zero_cats_lst)
    # 2 tables
    # a.
    #    origin_id, dest_id
    max_cat = zero_cats.max(0)
    shift = np.r_[0, max_cat[:-1] + 1]
    zero_cats += shift
    orig_dest_lst = []
    inverter = np.empty_like(zero_cats[:, 0])
    for i in range(ncats):
        col_order = list(range(ncats))
        col_order.remove(i)
        col_order = [i, *col_order]
        temp = zero_cats[:, col_order]
        idx = np.argsort(temp[:, 0])
        orig_dest_lst.append(temp[idx])
        if i == 0:
            inverter[idx] = np.arange(nobs)
    orig_dest = np.concatenate(orig_dest_lst, 0)
    # b.
    #    node_id, count, offset
    node_id, count = np.unique(orig_dest[:, 0], return_counts=True)
    offset = np.r_[0, np.where(np.diff(orig_dest[:, 0]) != 0)[0] + 1]

    def min_dtype(*args: linearmodels.typing.data.IntArray) -> str:
        bits = np.amax([np.log2(max(float(arg.max()), 1.0)) for arg in args])
        return f"int{min(j for j in (8, 16, 32, 64) if bits < (j - 1))}"

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


def in_2core_graph_slow(
    cats: DataFrame | np.ndarray,
) -> linearmodels.typing.data.BoolArray:
    """
    Find the observations in the 2-core using a reference method.

    Parameters
    ----------
    cats : DataFrame or ndarray
        Array containing the category codes of pandas categoricals
        with shape (nobs, ncats). A DataFrame must only contain categoricals.

    Returns
    -------
    ndarray
        Boolean array that marks non-singleton entries as True.

    Notes
    -----
    This is a reference implementation that can be very slow to remove
    all singleton nodes in some graphs.
    """
    if isinstance(cats, DataFrame):
        cats = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    if cats.shape[1] == 1:
        return in_2core_graph(cats)
    nobs, ncats = cats.shape
    retain_idx = np.arange(cats.shape[0])
    num_singleton = 1
    while num_singleton > 0 and cats.shape[0] > 0:
        singleton = np.zeros(cats.shape[0], dtype=bool)
        for i in range(ncats):
            ucats, counts = np.unique(cats[:, i], return_counts=True)
            singleton |= np.isin(cats[:, i], ucats[counts == 1])
        num_singleton = int(singleton.sum())
        if num_singleton:
            cats = cats[~singleton]
            retain_idx = retain_idx[~singleton]
    retain = np.zeros(nobs, dtype=bool)
    retain[retain_idx] = True
    return retain


# A column is absorbed if less than this share of its norm, measured before the effects
# were absorbed, is left. Both iterative algorithms that absorb effects, LSMR and the
# method of alternating projections (HDFE), stop with an error of about 1e-8 of the
# norm of the data, and the error is up to 100 times larger when the effects are only
# weakly connected. The tolerance is well above that, and far above machine precision.
_ABSORBED_TOL = 1e-5


def _absorbed_columns(
    x: linearmodels.typing.data.Float64Array,
    x_orig: linearmodels.typing.data.Float64Array | None = None,
    constant: int | None = None,
) -> list[tuple[int, list[int]]]:
    """
    Find the columns that are absorbed or collinear with the columns before them

    Parameters
    ----------
    x : ndarray
        Regressors after the effects were absorbed, with shape (nobs, ncol).
    x_orig : ndarray, optional
        Regressors before the effects were absorbed, with the same shape as x.
        Only the norm of each column is used. If not provided, x is assumed to be
        exact, and only columns that are collinear with the columns before them
        are found.
    constant : int, optional
        The location of a constant column. It is retained, and the other columns
        are compared with it first, so that a column that was restored as its
        grand mean is found.

    Returns
    -------
    list of tuple of (int, list of int)
        The columns to drop, in order. Each is paired with the sorted indices
        of the columns in the relation that makes it redundant: the column
        itself and the retained columns that it depends on.

    Notes
    -----
    Every column is divided by the norm that it had before the effects were
    absorbed, so that a column of size 1 is not touched by the effects and a
    column of size 0 was absorbed entirely. The columns are processed in order,
    and a column is dropped if less than ``_ABSORBED_TOL`` of it is left after it
    is orthogonalized to the retained columns before it. The tolerance does not
    depend on the units of the columns, and is above the error of the algorithms
    that absorb the effects. Without x_orig, the columns are divided by their own
    norm and the tolerance is the one of ``numpy.linalg.matrix_rank``.
    """
    tol = _ABSORBED_TOL if x_orig is not None else max(x.shape) * np.finfo(float).eps
    nobs, ncol = x.shape
    order = list(range(ncol))
    if constant is not None:
        order.remove(constant)
        order.insert(0, constant)
    norms = np.linalg.norm(x if x_orig is None else x_orig, axis=0)

    basis = np.empty((nobs, ncol))  # Orthonormal basis of the retained columns
    nbasis = 0
    unit_constant = None
    retained: list[int] = []
    scale: dict[int, float] = {}
    absorbed: list[tuple[int, list[int]]] = []
    for j in order:
        col = x[:, j]
        norm = float(norms[j])
        if x_orig is None and unit_constant is not None:
            # Without the original data, measure the part that is not constant
            norm = float(np.linalg.norm(col - unit_constant * (unit_constant @ col)))
        scale[j] = norm if norm > 0.0 else 1.0
        scaled = col / scale[j]
        resid = scaled
        for _ in range(2):  # Orthogonalizing twice removes the rounding error
            resid = resid - basis[:, :nbasis] @ (basis[:, :nbasis].T @ resid)
        size = float(np.linalg.norm(resid))
        if j == constant or size >= tol:
            retained.append(j)
            if size > 0.0:
                basis[:, nbasis] = resid / size
                if j == constant:
                    unit_constant = basis[:, nbasis]
                nbasis += 1
            continue

        # The relation is the column and the retained columns that make it redundant
        relation = [j]
        if retained:
            others = x[:, retained] / np.array([scale[i] for i in retained])
            coefs = np.linalg.lstsq(others, scaled, rcond=None)[0]
            share = np.abs(coefs) * np.linalg.norm(others, axis=0)
            relation.extend(i for i, s in zip(retained, share, strict=True) if s >= tol)
        absorbed.append((j, sorted(relation)))
    return absorbed


def check_absorbed(
    x: linearmodels.typing.data.Float64Array,
    variables: list[str] | tuple[str, ...],
    x_orig: linearmodels.typing.data.Float64Array | None = None,
) -> None:
    """
    Check a regressor matrix for variables absorbed by the effects.

    Parameters
    ----------
    x : ndarray
        Regressor matrix to check, after the effects were absorbed.
    variables : list of str or tuple of str
        Variable names, in the order of the columns of x.
    x_orig : ndarray, optional
        Regressors before the effects were absorbed, with the same shape as x. A
        column of x is absorbed if less than 1e-5 of the norm of the same
        column of x_orig is left, so that the check does not depend on the units
        of the variables and detects the columns that the algorithm that
        absorbed the effects left with an error of its tolerance. If not
        provided, only columns that are collinear with other columns are
        found.

    Raises
    ------
    AbsorbingEffectError
        If a column is absorbed by the effects or is collinear with other
        columns after the effects are removed.
    """
    if x.size == 0:
        return

    absorbed = _absorbed_columns(x, x_orig)
    if not absorbed:
        return
    if x_orig is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            size = np.linalg.norm(x, axis=0) / np.linalg.norm(x_orig, axis=0)
        if np.all(~(size >= _ABSORBED_TOL)):
            raise AbsorbingEffectError(
                "All exog variables have been absorbed. The model cannot be estimated."
            )
    rows = [
        " " * 10 + ", ".join(str(variables[i]) for i in relation)
        for _, relation in absorbed
    ]
    msg = absorbing_error_msg.format(absorbed_variables="\n".join(rows))
    raise AbsorbingEffectError(msg)


def not_absorbed(
    x: linearmodels.typing.data.Float64Array,
    has_constant: bool = False,
    loc: int | None = None,
    x_orig: linearmodels.typing.data.Float64Array | None = None,
) -> list[int]:
    """
    Construct a list of the indices of regressors that are not absorbed.

    Parameters
    ----------
    x : ndarray
        Regressor matrix to check, after the effects were absorbed.
    has_constant : bool, optional
        Flag indicating that x has a constant column. The default is False.
    loc : int, optional
        The location of the constant column. It is always retained, and the
        other columns are compared with it first.
    x_orig : ndarray, optional
        Regressors before the effects were absorbed, with the same shape as x. A
        column of x is absorbed if less than 1e-5 of the norm of the same
        column of x_orig is left, so that the check does not depend on the units
        of the variables. A column is also dropped if it is collinear with the
        retained columns before it, using the same tolerance. If not provided,
        only columns that are collinear with other columns are dropped.

    Returns
    -------
    list of int
        Columns to retain.
    """
    constant = None
    if has_constant:
        assert isinstance(loc, int)
        constant = loc
    dropped = {i for i, _ in _absorbed_columns(x, x_orig, constant)}
    return [i for i in range(x.shape[1]) if i not in dropped]


class PanelModelData(NamedTuple):  # numpydoc ignore=PR02
    """
    Typed namedtuple to hold simulated panel data.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the panel model's data.
    weights : DataFrame
        DataFrame containing the panel model's weights.
    other_effects : DataFrame
        DataFrame containing the panel model's effects,
        excluding time and entity effects.
    clusters : DataFrame
        DataFrame containing cluster ids.
    """

    data: DataFrame
    weights: DataFrame
    other_effects: DataFrame
    clusters: DataFrame


def _integers(
    rng: numpy.random.RandomState | numpy.random.Generator,
    high: int,
    size: tuple[int, ...],
) -> linearmodels.typing.data.IntArray:
    """Random integers in [0, high) from a RandomState or a Generator."""
    if isinstance(rng, numpy.random.Generator):
        return rng.integers(0, high, size)
    return rng.randint(0, high, size)


def generate_panel_data(
    nentity: int = 971,
    ntime: int = 7,
    nexog: int = 5,
    const: bool = False,
    missing: float = 0,
    other_effects: int = 2,
    ncats: int | list[int] = 4,
    rng: numpy.random.RandomState | numpy.random.Generator | None = None,
) -> PanelModelData:
    """
    Simulate panel data for testing.

    Parameters
    ----------
    nentity : int, optional
        The number of entities in the panel. The default is 971.
    ntime : int, optional
        The number of time periods in the panel. The default is 7.
    nexog : int, optional
        The number of explanatory variables in the dataset. The default
        is 5.
    const : bool, optional
        Flag indicating that the model should include a constant. The
        default is False.
    missing : float, optional
        The fraction of values that are missing, between 0 and 1. The
        default is 0, so that no values are missing.
    other_effects : int, optional
        The number of other effects generated. The default is 2.
    ncats : int or list of int, optional
        The number of categories to use in other_effects and variance
        clusters. If a list, then it must have as many elements as
        other_effects. The default is 4.
    rng : numpy.random.RandomState or numpy.random.Generator, optional
        A NumPy RandomState or Generator instance. If not provided, a
        RandomState is initialized using a fixed seed.

    Returns
    -------
    PanelModelData
        A namedtuple derived class containing 4 DataFrames:

        * `data` - A simulated data with variables y and x# for # in 0,...,4.
          If const is True, then also contains a column named const.
        * `weights` - Simulated non-negative weights.
        * `other_effects` - Simulated effects.
        * `clusters` - Simulated data to use in clustered covariance estimation.
    """
    if rng is None:
        rng = np.random.RandomState(
            [
                0xA14E2429,
                0x448D2E51,
                0x91B558E7,
                0x6A3F5CD2,
                0x22B43ABB,
                0xE746C92D,
                0xCE691A7D,
                0x66746EE7,
            ]
        )

    n, t, k = nentity, ntime, nexog
    k += int(const)
    x = rng.standard_normal((k, t, n))
    beta = np.arange(1, k + 1)[:, None, None] / k
    y: linearmodels.typing.data.Float64Array = (
        (x * beta).sum(0)
        + rng.standard_normal((t, n))
        + 2 * rng.standard_normal((1, n))
    )

    w = rng.chisquare(5, (t, n)) / 5
    c: linearmodels.typing.data.IntArray | None = None
    cats = [f"cat.{i}" for i in range(other_effects)]
    if other_effects:
        if not isinstance(ncats, list):
            ncats = [ncats] * other_effects
        _c = []
        for i in range(other_effects):
            nc = ncats[i]
            _c.append(_integers(rng, nc, (1, t, n)))
        c = np.concatenate(_c, 0)

    vcats = [f"varcat.{i}" for i in range(2)]
    vc2 = np.ones((2, t, 1)) @ _integers(rng, n // 2, (2, 1, n))

    if const:
        x[0] = 1.0

    if missing > 0:
        locs = rng.choice(n * t, int(n * t * missing))
        y.flat[locs] = np.nan
        locs = rng.choice(n * t * k, int(n * t * k * missing))
        x.flat[locs] = np.nan

    entities = [f"firm{i}" for i in range(n)]
    time = list(date_range("1-1-1900", periods=t, freq=ANNUAL_FREQ))
    var_names = [f"x{i}" for i in range(k)]
    if const:
        var_names[1:] = var_names[:-1]
        var_names[0] = "const"
    # y = DataFrame(y, index=time, columns=entities)
    y_df = panel_to_frame(
        y[None], items=["y"], major_axis=time, minor_axis=entities, swap=True
    )
    index = y_df.index
    w_df = panel_to_frame(
        w[None], items=["w"], major_axis=time, minor_axis=entities, swap=True
    )
    w_df = w_df.reindex(index)
    x_df = panel_to_frame(
        x, items=var_names, major_axis=time, minor_axis=entities, swap=True
    )
    x_df = x_df.reindex(index)
    c_df = panel_to_frame(
        c, items=cats, major_axis=time, minor_axis=entities, swap=True
    )
    other_eff = c_df.reindex(index)
    clusters = panel_to_frame(
        vc2, items=vcats, major_axis=time, minor_axis=entities, swap=True
    )
    clusters = clusters.reindex(index)
    data = concat([y_df, x_df], axis=1, sort=False)
    return PanelModelData(data, w_df, other_eff, clusters)
