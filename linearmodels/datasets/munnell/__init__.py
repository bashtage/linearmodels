DESCR = """
Munnell, A. “Why Has Productivity Declined? Productivity and Public
Investment.” New England Economic Review, 1990, pp. 3–22.

STATE   Full State Name
ST_ABB  State Code
YR      Year
P_CAP   Public Capital
HWY     Highway Capital
WATER   Water Utility Capital
UTIL    Utility Capital
PC      Private Capital
GSP     Gross state product
EMP     Employment (labor)
UNEMP   Unemployment rate
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'munnell.csv.bz2')
