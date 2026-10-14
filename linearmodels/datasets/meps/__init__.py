"""
Medical Expenditure Panel Survey data on prescription drug expenditure.
"""

from pandas import DataFrame

from linearmodels import datasets

DESCR = """
age               Age
age2              Age-squared
black             Black
blhisp            Black or Hispanic
drugexp           Presc-drugs expense
educyr            Years of education
fair              Fair health
female            Female
firmsz            Firm size
fph               fair or poor health
good              Good health
hi_empunion       Insured thro emp/union
hisp              Hiapanic
income            Income
ldrugexp          log(drugexp)
linc              log(income)
lowincome         Low income
marry             Married
midincome         Middle income
msa               Metropolitan stat area
multlc            Multiple locations
poor              Poor health
poverty           Poor
priolist          Priority list cond
private           Private insurance
ssiratio          SSI/Income ratio
totchr            Total chronic cond
vegood            V-good health
vgh               vg or good health
"""


def load() -> DataFrame:
    """
    Load the prescription drug expenditure data.

    Returns
    -------
    DataFrame
        The data. ``DESCR`` describes the variables.
    """
    return datasets.load(__file__, "meps.csv.bz2")
