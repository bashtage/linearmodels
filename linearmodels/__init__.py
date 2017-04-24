import os

from linearmodels.iv.model import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from linearmodels.panel.model import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, \
    RandomEffects
from ._version import get_versions

WARN_ON_MISSING = os.environ.get('LINEARMODELS_WARN_ON_MISSING', True)
WARN_ON_MISSING = False if WARN_ON_MISSING in ('0', 'False') else True
DROP_MISSING = os.environ.get('LINEARMODELS_DROP_MISSING', True)
DROP_MISSING = False if DROP_MISSING in ('0', 'False') else True


__all__ = ['PooledOLS', 'PanelOLS', 'FirstDifferenceOLS', 'BetweenOLS', 'RandomEffects',
           'IVLIML', 'IVGMM', 'IVGMMCUE', 'IV2SLS', 'WARN_ON_MISSING', 'DROP_MISSING']

__version__ = get_versions()['version']
del get_versions

# TODO: Finish documentation for IV estimators
# TODO: Finish examples for IV estimators
