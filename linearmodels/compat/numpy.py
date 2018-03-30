from distutils.version import LooseVersion

import numpy as np

NP_LT_114 = LooseVersion(np.__version__) < LooseVersion('1.14')


def lstsq(a, b, rcond=None):
    """
    Wrapper that allows rcond to be automatically set to avoid FutureWarning
    """
    if rcond is None and NP_LT_114:
        rcond = -1
    return np.linalg.lstsq(a, b, rcond=rcond)


__all__ = ['lstsq']
