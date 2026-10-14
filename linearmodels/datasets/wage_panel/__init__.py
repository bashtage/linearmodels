"""
Panel of wages and union membership of Vella and Verbeek (1998).
"""

from pandas import DataFrame

from linearmodels import datasets

DESCR = """
F. Vella and M. Verbeek (1998), "Whose Wages Do Unions Raise? A Dynamic Model
of Unionism and Wage Rate Determination for Young Men," Journal of Applied
Econometrics 13, 163-183.

nr                       person identifier
year                     1980 to 1987
black                    =1 if black
exper                    labor market experience
hisp                     =1 if Hispanic
hours                    annual hours worked
married                  =1 if married
educ                     years of schooling
union                    =1 if in union
lwage                    log(wage)
expersq                  exper^2
occupation               Occupation code
"""


def load() -> DataFrame:
    """
    Load the wages and union membership data.

    Returns
    -------
    DataFrame
        The data. ``DESCR`` describes the variables.
    """
    return datasets.load(__file__, "wage_panel.csv.bz2")
