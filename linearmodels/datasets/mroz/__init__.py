DESCR = """
T.A. Mroz (1987), "The Sensitivity of an Empirical Model of Married Women's
Hours of Work to Economic and Statistical Assumptions," Econometrica 55,
765-799.

nlf        1 if in labor force, 1975
hours      hours worked, 1975
kidslt6    # kids < 6 years
kidsge6    # kids 6-18
age        woman's age in yrs
educ       years of schooling
wage       estimated wage from earns., hours
repwage    reported wage at interview in 1976
hushrs     hours worked by husband, 1975
husage     husband's age
huseduc    husband's years of schooling
huswage    husband's hourly wage, 1975
faminc     family income, 1975
mtr        fed. marginal tax rate facing woman
motheduc   mother's years of schooling
fatheduc   father's years of schooling
unem       unem. rate in county of resid.
city       =1 if live in SMSA
exper      actual labor mkt exper
nwifeinc   (faminc - wage*hours)/1000
lwage      log(wage)
expersq    exper^2
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'mroz.csv.bz2')
