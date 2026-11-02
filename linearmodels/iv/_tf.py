"""Generated tF critical values for single-instrument IV inference."""

from functools import lru_cache
import hashlib
from importlib import import_module
from importlib.resources import files
import io
import json
from math import inf, isfinite, sqrt
from typing import Any

from numpy import load
from scipy.interpolate import PchipInterpolator

_TF_ALPHAS = (
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.06,
    0.075,
    0.1,
    0.125,
    0.15,
    0.2,
    0.25,
)
_TF_LEVELS = {1 - alpha: alpha for alpha in _TF_ALPHAS}


@lru_cache(maxsize=32)
def _tf_table(alpha: float) -> tuple[float, float, PchipInterpolator]:
    root = files("linearmodels.iv").joinpath("_tf_data")
    prefix = f"alpha-{alpha!r}"
    metadata = json.loads(root.joinpath(prefix + ".json").read_text())
    if metadata["schema_version"] != 1 or metadata["alpha"] != alpha:
        raise ValueError("Unsupported or inconsistent tF table metadata.")
    contents = root.joinpath(prefix + ".npz").read_bytes()
    if hashlib.sha256(contents).hexdigest() != metadata["npz_sha256"]:
        raise ValueError("tF table checksum does not match its metadata.")
    with load(io.BytesIO(contents), allow_pickle=False) as data:
        spline = PchipInterpolator(data["f"], data["regularized"], extrapolate=False)
    return metadata["boundary"], metadata["plateau_f"], spline


@lru_cache(maxsize=32)
def _tf_calibration(alpha: float) -> Any:
    try:
        package = import_module("tf_inference")
    except ImportError as exc:
        raise ImportError(
            "method='compute' requires the optional tf-inference package. "
            "Install it from https://github.com/finite-sample/tf-inference."
        ) from exc
    return package.calibrate(alpha)


def _tf_critical_value(
    f_stat: float, level: float = 0.95, *, method: str = "table"
) -> float:
    """Evaluate a generated tF table or explicitly request numerical calibration."""
    if not isfinite(f_stat) or f_stat < 0:
        raise ValueError("The first-stage F statistic must be finite and nonnegative.")
    if not isfinite(level) or not 0.75 <= level <= 0.9999:
        raise ValueError("level must be finite and between 0.75 and 0.9999.")
    # Match confidence levels before subtraction to retain the exact bundled alpha.
    alpha = _TF_LEVELS.get(level)
    if method == "compute":
        return float(
            _tf_calibration(alpha if alpha is not None else 1 - level).critical_value(
                f_stat
            )
        )
    if method != "table":
        raise ValueError("method must be 'table' or 'compute'.")
    if alpha is None:
        raise ValueError(
            f"No bundled tF table for level={level}. "
            "Use method='compute' with the optional tf-inference package."
        )
    boundary, plateau_f, spline = _tf_table(alpha)
    if f_stat <= boundary:
        return inf
    value = min(f_stat, plateau_f)
    return sqrt(float(spline(value)) / (value - boundary))
