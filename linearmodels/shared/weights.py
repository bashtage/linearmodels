"""
Validation and normalization of the observation weights of the models.
"""

from __future__ import annotations

import numpy as np

from linearmodels.typing import Float64Array

__all__ = ["check_weights", "normalize_weights"]


def check_weights(weights: Float64Array, name: str = "weights") -> None:
    """
    Check that weights are strictly positive and finite.

    Parameters
    ----------
    weights : ndarray
        The weights to check. A NaN is a missing value, which the models drop with
        the observation, and it is not checked.
    name : str, optional
        The name that the error message uses for the weights. The default is
        ``"weights"``. The system models name the equation.

    Raises
    ------
    ValueError
        If a weight is zero, negative, or infinite.

    Notes
    -----
    The models call this function with the weights as they are given. The message
    is ``"{name} must be strictly positive and finite."``.
    """
    values = np.asarray(weights, dtype=np.float64)
    if np.any(values <= 0) or np.any(np.isinf(values)):
        raise ValueError(f"{name} must be strictly positive and finite.")


def normalize_weights(weights: Float64Array) -> Float64Array:
    """
    Scale weights to have a mean of 1 without overflow.

    Parameters
    ----------
    weights : ndarray
        Strictly positive and finite weights. A NaN is a missing value, which is
        ignored in the mean and remains a NaN.

    Returns
    -------
    ndarray
        The weights divided by their mean, with the shape of weights.

    Notes
    -----
    The weights are divided by the largest weight, and then by the mean of the result.
    The mean of the weights can overflow, e.g., the sum of 10 weights of 1e308 is
    infinite, so that dividing by it gives weights of 0, but the mean of weights that
    are at most 1 cannot. The result is finite for any finite weights: it is at most
    the number of weights. A weight that is smaller than the largest weight by a factor
    of more than about 1e323 is 0 after the division by the largest weight. The result
    is the weights divided by their mean, up to rounding, and weights that are all
    equal are exactly 1.
    """
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return values.copy()
    scaled = values / np.nanmax(values)
    return scaled / np.nanmean(scaled)
