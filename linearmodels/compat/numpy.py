from distutils.version import LooseVersion

import numpy as np

NP_LT_113 = LooseVersion(np.__version__) < LooseVersion('1.13')
NP_LT_114 = LooseVersion(np.__version__) < LooseVersion('1.14')


def lstsq(a, b, rcond=None):
    """
    Wrapper that allows rcond to be automatically set to avoid FutureWarning
    """
    if rcond is None and NP_LT_114:
        rcond = -1
    return np.linalg.lstsq(a, b, rcond=rcond)


if NP_LT_113:
    def isin(element, test_elements, assume_unique=False, invert=False):
        """
        Compatibility version of isin that was added in 1.13
        """
        element = np.asarray(element)
        return np.in1d(element, test_elements, assume_unique=assume_unique,
                       invert=invert).reshape(element.shape)
else:
    isin = np.isin


__all__ = ['isin', 'lstsq']
