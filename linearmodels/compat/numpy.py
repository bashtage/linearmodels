from distutils.version import LooseVersion
from typing import Optional, Tuple

import numpy as np

from linearmodels.typing import NDArray

NP_LT_114 = LooseVersion(np.__version__) < LooseVersion("1.14")


def lstsq(
    a: NDArray, b: NDArray, rcond: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Wrapper that allows rcond to be automatically set to avoid FutureWarning
    """
    if rcond is None and NP_LT_114:
        rcond = -1
    return np.linalg.lstsq(a, b, rcond=rcond)


__all__ = ["lstsq"]
