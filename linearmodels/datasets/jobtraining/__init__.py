DESCR = """
H. Holzer, R. Block, M. Cheatham, and J. Knott (1993), "Are Training Subsidies
Effective? The Michigan Experience," Industrial and Labor Relations Review 46,
625-636.

year                     1987, 1988, or 1989
fcode                    firm code number
employ                   # employees at plant
sales                    annual sales, $
avgsal                   average employee salary
scrap                    scrap rate (per 100 items)
rework                   rework rate (per 100 items)
tothrs                   total hours training
union                    =1 if unionized
grant                    =1 if received grant
d89                      =1 if year = 1989
d88                      =1 if year = 1988
totrain                  total employees trained
hrsemp                   tothrs/totrain
lscrap                   log(scrap)
lemploy                  log(employ)
lsales                   log(sales)
lrework                  log(rework)
lhrsemp                  log(1 + hrsemp)
lscrap_1                 lagged lscrap; missing 1987
grant_1                  lagged grant; assumed 0 in 1987
clscrap                  lscrap - lscrap_1; year > 1987
cgrant                   grant - grant_1
clemploy                 lemploy - lemploy[t-1]
clsales                  lavgsal - lavgsal[t-1]
lavgsal                  log(avgsal)
clavgsal                 lavgsal - lavgsal[t-1]
cgrant_1                 cgrant[t-1]
chrsemp                  hrsemp - hrsemp[t-1]
lhrsemp                 lhrsemp - lhrsemp[t-1]
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'jobtraining.csv.bz2')
