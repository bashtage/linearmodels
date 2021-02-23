from collections import defaultdict
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from pandas import DataFrame, concat, date_range
import scipy.sparse as sp

from linearmodels.shared.utility import panel_to_frame
from linearmodels.typing import NDArray
from linearmodels.typing.data import ArrayLike

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

SparseArray = TypeVar("SparseArray", sp.csc_matrix, sp.csr_matrix, sp.coo_matrix)
SparseOrDense = TypeVar(
    "SparseOrDense", NDArray, sp.csc_matrix, sp.csr_matrix, sp.coo_matrix
)


def preconditioner(
    d: SparseOrDense, *, copy: bool = False
) -> Tuple[SparseOrDense, NDArray]:
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
        cond = cast(NDArray, np.sqrt((d ** 2).sum(0)))
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

    cond = cast(NDArray, np.sqrt(d.multiply(d).sum(0)).A1)
    locs = np.zeros_like(d.indices)
    locs[d.indptr[1:-1]] = 1
    locs = np.cumsum(locs)
    d.data /= np.take(cond, locs)
    if klass is not None:
        d = klass(d)

    return d, cond


def dummy_matrix(
    cats: ArrayLike,
    *,
    output_format: str = "csc",
    drop: str = "first",
    drop_all: bool = False,
    precondition: bool = True,
) -> Tuple[Union[sp.csc_matrix, sp.csr_matrix, sp.coo_matrix, NDArray], NDArray]:
    """
    Parameters
    ----------
    cats: {DataFrame, ndarray}
        Array containing the category codes of pandas categoricals
        (nobs, ncats)
    output_format: {'csc', 'csr', 'coo', 'array'}
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
    if isinstance(cats, DataFrame):
        codes = np.column_stack([np.asarray(cats[c].cat.codes) for c in cats])
    else:
        codes = cats

    data: Dict[str, List[np.ndarray]] = defaultdict(list)
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

    if output_format in ("csc", "array"):
        fmt = sp.csc_matrix
    elif output_format == "csr":
        fmt = sp.csr_matrix
    elif output_format == "coo":
        fmt = sp.coo_matrix
    else:
        raise ValueError("Unknown format: {0}".format(output_format))
    out = fmt(
        (
            np.concatenate(data["values"]),
            (np.concatenate(data["rows"]), np.concatenate(data["cols"])),
        )
    )
    if output_format == "array":
        out = out.toarray()

    if precondition:
        out, cond = preconditioner(out, copy=False)
    else:
        cond = np.ones(out.shape[1])

    return out, cond


def _remove_node(node: int, meta: NDArray, orig_dest: NDArray) -> Tuple[int, int]:
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


def _py_drop_singletons(meta: NDArray, orig_dest: NDArray) -> None:
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


def in_2core_graph(cats: ArrayLike) -> NDArray:
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
        col_order = [i] + col_order
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

    def min_dtype(*args: NDArray) -> str:
        bits = np.amax([np.log2(max(float(arg.max()), 1.0)) for arg in args])
        return "int{0}".format(min([j for j in (8, 16, 32, 64) if bits < (j - 1)]))

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


def in_2core_graph_slow(cats: ArrayLike) -> NDArray:
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


def check_absorbed(
    x: NDArray, variables: Sequence[str], x_orig: Optional[NDArray] = None
) -> None:
    """
    Check a regressor matrix for variables absorbed

    Parameters
    ----------
    x : ndarray
        Regressor matrix to check
    variables : List[str]
        List of variable names
    x_orig : ndarray, optional
        Original data. If provided uses a norm check to ascertain if all
        variables have been absorbed.
    """
    if x.size == 0:
        return

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
            rows.append(" " * 10 + ", ".join((str(variables[vi]) for vi in vars_idx)))
        absorbed_variables = "\n".join(rows)
        msg = absorbing_error_msg.format(absorbed_variables=absorbed_variables)
        raise AbsorbingEffectError(msg)
    if x_orig is None:
        return

    new_norm = np.linalg.norm(x, axis=0)
    orig_norm = np.linalg.norm(x_orig, axis=0)
    if np.all(((new_norm / orig_norm) ** 2) < np.finfo(float).eps):
        raise AbsorbingEffectError(
            "All exog variables have been absorbed. The model cannot be estimated."
        )


def not_absorbed(
    x: NDArray, has_constant: bool = False, loc: Optional[int] = None
) -> List[int]:
    """
    Construct a list of the indices of regressors that are not absorbed

    Parameters
    ----------
    x : ndarray
        Regressor matrix to check
    has_constant : bool
        Flag indicating that x has a constant column
    loc : int
        The location of the constant column

    Returns
    -------
    retain : list[int]
        List of columns to retain
    """
    if np.linalg.matrix_rank(x) == x.shape[1]:
        return list(range(x.shape[1]))
    if has_constant:
        assert isinstance(loc, int)
        check = [i for i in range(x.shape[1]) if i != loc]
        const = x[:, [loc]]
        sub = x[:, check]
        x = sub - const @ np.linalg.lstsq(const, sub, rcond=None)[0]
    xpx = x.T @ x
    vals, vecs = np.linalg.eigh(xpx)
    if vals.max() == 0.0:
        if has_constant:
            assert isinstance(loc, int)
            return [loc]
        return []

    tol = vals.max() * x.shape[1] * np.finfo(np.float64).eps
    absorbed = vals < tol
    nabsorbed = absorbed.sum()
    q, r = np.linalg.qr(x)
    threshold = np.sort(np.abs(np.diag(r)))[nabsorbed]
    drop = np.where(np.abs(np.diag(r)) < threshold)[0]
    retain: Set[int] = set(range(x.shape[1])).difference(drop)
    if has_constant:
        assert isinstance(loc, int)
        retain.update({loc})
    return sorted(retain)


class PanelModelData(NamedTuple):
    """
    Typed namedtuple to hold simulated panel data
    """

    data: DataFrame
    weights: DataFrame
    other_effects: DataFrame
    clusters: DataFrame


def generate_panel_data(
    nentity: int = 971,
    ntime: int = 7,
    nexog: int = 5,
    const: bool = False,
    missing: float = 0,
    other_effects: int = 2,
    ncats: Union[int, List[int]] = 4,
    rng: Optional[np.random.RandomState] = None,
) -> PanelModelData:
    """

    Parameters
    ----------
    nentity : int, default 971
        The number of entities in the panel.
    ntime : int, default 7
        The number of time periods in the panel.
    nexog : int, default 5
        The number of explanatory variables in the dataset.
    const : bool, default False
        Flag indicating that the model should include a constant.
    missing : float, default 0
        The percentage of values that are missing. Should be between 0 and 100.
    other_effects : int, default 2
        The number of other effects generated.
    ncats : Union[int, Sequence[int]], default 4
        The number of categories to use in other_effects and variance
        clusters. If list-like, then it must have as many elements
        as other_effects.
    rng : RandomState, default None
        A NumPy RandomState instance. If not provided, one is initialized
        using a fixed seed.

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
    y: NDArray = (
        (x * beta).sum(0)
        + rng.standard_normal((t, n))
        + 2 * rng.standard_normal((1, n))
    )

    w = rng.chisquare(5, (t, n)) / 5
    c: Optional[NDArray] = None
    cats = [f"cat.{i}" for i in range(other_effects)]
    if other_effects:
        if not isinstance(ncats, list):
            ncats = [ncats] * other_effects
        _c = []
        for i in range(other_effects):
            nc = ncats[i]
            _c.append(rng.randint(0, nc, (1, t, n)))
        c = np.concatenate(_c, 0)

    vcats = [f"varcat.{i}" for i in range(2)]
    vc2 = np.ones((2, t, 1)) @ rng.randint(0, n // 2, (2, 1, n))
    vc1 = vc2[[0]]

    if const:
        x[0] = 1.0

    if missing > 0:
        locs = rng.choice(n * t, int(n * t * missing))
        # TODO:: Fix typing in later version of numpy
        y.flat[locs] = np.nan  # type: ignore
        locs = rng.choice(n * t * k, int(n * t * k * missing))
        # TODO:: Fix typing in later version of numpy
        x.flat[locs] = np.nan  # type: ignore

    entities = [f"firm{i}" for i in range(n)]
    time = date_range("1-1-1900", periods=t, freq="A-DEC")
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
    vc1_df = panel_to_frame(
        vc1, items=vcats[:1], major_axis=time, minor_axis=entities, swap=True
    )
    vc1_df = vc1_df.reindex(index)
    vc2_df = panel_to_frame(
        vc2, items=vcats, major_axis=time, minor_axis=entities, swap=True
    )
    vc2_df = vc2_df.reindex(index)
    clusters = concat([vc1_df, vc2_df], sort=False)
    data = concat([y_df, x_df], axis=1, sort=False)
    return PanelModelData(data, w_df, other_eff, clusters)
