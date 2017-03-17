DESCR = """
M. Blackburn and D. Neumark (1992), "Unobserved Ability, Efficiency Wages, and
Interindustry Wage Differentials," Quarterly Journal of Economics 107, 1421-1436.

wage                     monthly earnings
hours                    average weekly hours
IQ                       IQ score
KWW                      knowledge of world work score
educ                     years of education
exper                    years of work experience
tenure                   years with current employer
age                      age in years
married                  =1 if married
black                    =1 if black
south                    =1 if live in south
urban                    =1 if live in SMSA
sibs                     number of siblings
brthord                  birth order
meduc                    mother's education
feduc                    father's education
lwage                    natural log of wage
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'wage.csv.bz2')
