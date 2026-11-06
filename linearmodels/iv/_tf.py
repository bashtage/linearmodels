"""Interface to the optional tF calibration package."""

from functools import lru_cache
from importlib import import_module
from math import isfinite
from types import ModuleType


@lru_cache(maxsize=1)
def _tf_levels(package: ModuleType) -> dict[float, float]:
    """Map confidence levels to exact table keys without rescanning package data."""
    return {1 - alpha: alpha for alpha in package.available_alphas()}


def _tf_critical_value(
    f_stat: float, level: float = 0.95, *, method: str = "table"
) -> float:
    """Evaluate tF critical values using the optional tf-inference package."""
    if not isfinite(f_stat) or f_stat < 0:
        raise ValueError("The first-stage F statistic must be finite and nonnegative.")
    if not isfinite(level) or not 0.75 <= level <= 0.9999:
        raise ValueError("level must be finite and between 0.75 and 0.9999.")
    if method not in ("table", "compute"):
        raise ValueError("method must be 'table' or 'compute'.")
    try:
        package = import_module("tf_inference")
    except ModuleNotFoundError as exc:
        if exc.name != "tf_inference":
            raise
        raise ImportError(
            "tF inference requires the optional tf-inference package "
            "(Python 3.12 or later). "
            "Install it with 'python -m pip install tf-inference'."
        ) from exc

    # Match levels before subtraction to preserve the package's exact table keys.
    alpha = _tf_levels(package).get(level)
    if alpha is None:
        if method == "table":
            raise ValueError(
                f"No tF table for level={level}. Use method='compute' "
                "for numerical calibration."
            )
        alpha = 1 - level
    return float(package.critical_value(f_stat, alpha=alpha, method=method))
