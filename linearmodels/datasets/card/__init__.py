DESCR = """
D. Card (1995), "Using Geographic Variation in College Proximity to Estimate
the Return to Schooling," in Aspects of Labour Market Behavior:  Essays in
Honour of John Vanderkamp.  Ed. L.N. Christophides, E.K. Grant, and R.
Swidinsky, 201-222.  Toronto: University of Toronto Press.

id                       person identifier
nearc2                   =1 if near 2 yr college, 1966
nearc4                   =1 if near 4 yr college, 1966
educ                     years of schooling, 1976
age                      in years
fatheduc                 father's schooling
motheduc                 mother's schooling
weight                   NLS sampling weight, 1976
momdad14                 =1 if live with mom, dad at 14
sinmom14                 =1 if with single mom at 14
step14                   =1 if with step parent at 14
reg661                   =1 for region 1, 1966
reg662                   =1 for region 2, 1966
reg663                   =1 for region 3, 1966
reg664                   =1 for region 4, 1966
reg665                   =1 for region 5, 1966
reg666                   =1 for region 6, 1966
reg667                   =1 for region 7, 1966
reg668                   =1 for region 8, 1966
reg669                   =1 for region 9, 1966
south66                  =1 if in south in 1966
black                    =1 if black
smsa                     =1 in in SMSA, 1976
south                    =1 if in south, 1976
smsa66                   =1 if in SMSA, 1966
wage                     hourly wage in cents, 1976
enroll                   =1 if enrolled in school, 1976
KWW                      knowledge world of work score
IQ                       IQ score
married                  =1 if married, 1976
libcrd14                 =1 if lib. card in home at 14
exper                    age - educ - 6
lwage                    log(wage)
xpersq                   exper**2
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'card.csv.bz2')
