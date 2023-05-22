# cython: boundscheck=False, wraparound=False, language_level=3


cimport numpy as np

ctypedef fused any_int:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t


cdef void _remove_node(
        any_int node,
        any_int[:, ::1] meta,
        any_int[:, ::1] orig_dest,
        any_int *next_node,
        any_int *next_count
):
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
    cdef any_int next_offset, orig, reverse_offset, reverse_node, _next_orig
    # 3. Decrement
    meta[node, 1] -= 1
    # 1. Remove forewrd link
    next_offset = meta[node, 2]
    orig = orig_dest[next_offset, 0]
    next_node[0] = orig_dest[next_offset, 1]
    while next_node[0] == -1:
        # Increment since this could have been previously deleted
        next_offset += 1
        _next_orig = orig_dest[next_offset, 0]
        next_node[0] = orig_dest[next_offset, 1]
    # 4. Remove next_node's link
    orig_dest[next_offset, 1] = -1

    # 2. Remove the backward link
    # Set reverse to -1
    reverse_offset = meta[next_node[0], 2]
    reverse_node = orig_dest[reverse_offset, 1]
    while reverse_node != orig:
        reverse_offset += 1
        reverse_node = orig_dest[reverse_offset, 1]
    orig_dest[reverse_offset, 1] = -1

    # Step forward
    meta[next_node[0], 1] -= 1
    next_count[0] = meta[next_node[0], 1]


def _drop_singletons(any_int[:, ::1] meta, any_int[:, ::1] orig_dest):
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
    cdef any_int next_node, next_count, i
    for i in range(meta.shape[0]):
        if meta[i, 1] == 1:
            next_node = i
            next_count = 1
            while next_count == 1:
                # Follow singleton chains
                _remove_node(next_node, meta, orig_dest, &next_node, &next_count)
