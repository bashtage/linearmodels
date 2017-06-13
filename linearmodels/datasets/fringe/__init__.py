DESCR = """
F. Vella (1993), "A Simple Estimator for Simultaneous Models with Censored
Endogenous Regressors," International Economic Review 34, 441-457.

annearn                  annual earnings, $
hrearn                   hourly earnings, $
exper                    years work experience
age                      age in years
depends                  number of dependents
married                  =1 if married
tenure                   years with current employer
educ                     years schooling
nrtheast                 =1 if live in northeast
nrthcen                  =1 if live in north central
south                    =1 if live in south
male                     =1 if male
white                    =1 if white
union                    =1 if union member
office                   =1 if office worker
annhrs                   annual hours worked
ind1                     =1 if industry == 1
ind2                     =1 if industry == 2
ind3                     =1 if industry == 3
ind4                     =1 if industry == 4
ind5                     =1 if industry == 5
ind6                     =1 if industry == 6
ind7                     =1 if industry == 7
ind8                     =1 if industry == 8
ind9                     =1 if industry == 9
vacdays                  $ value of vac days
sicklve                  $ value of sick leave
insur                    $ value of employee insur
pension                  $ value of employee pension
annbens                  vacdays+sicklve+insur+pension
hrbens                   hourly benefits, $
annhrssq                 annhrs^2
beratio                  annbens/annearn
lannhrs                  log(annhrs)
tenuresq                 tenure^2
expersq                  exper^2
lannearn                 log(annearn)
peratio                  pension/annearn
vserat                   (vacdays+sicklve)/annearn
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'fringe.csv.bz2')
