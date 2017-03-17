DESCR = """
J. Mullahy (1997), "Instrumental-Variable Estimation of Count Data Models:
Applications to Models of Cigarette Smoking Behavior," Review of Economics
and Statistics 79, 596-593.

faminc                   1988 family income, $1000s
cigtax                   cig. tax in home state, 1988
cigprice                 cig. price in home state, 1988
bwght                    birth weight, ounces
fatheduc                 father's yrs of educ
motheduc                 mother's yrs of educ
parity                   birth order of child
male                     =1 if male child
white                    =1 if white
cigs                     cigs smked per day while preg
lbwght                   log of bwght
bwghtlbs                 birth weight, pounds
packs                    packs smked per day while preg
faminc                   log(faminc)
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'birthweight.csv.bz2')
