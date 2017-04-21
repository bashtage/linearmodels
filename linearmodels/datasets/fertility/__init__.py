DESCR = """
W. Sander, "The Effect of Women's Schooling on Fertility," Economics Letters
40, 229-233.

ear                     72 to 84, even
educ                     years of schooling
meduc                    mother's education
feduc                    father's education
age                      in years
kids                     # children ever born
black                    = 1 if black
east                     = 1 if lived in east at 16
northcen                 = 1 if lived in nc at 16
west                     = 1 if lived in west at 16
farm                     = 1 if on farm at 16
othrural                 = 1 if other rural at 16
town                     = 1 if lived in town at 16
smcity                   = 1 if in small city at 16
y74                      = 1 if year = 74
y76
y78
y80
y82
y84
agesq                    age^2
y74educ                  y74*educ
y76educ
y78educ
y80educ
y82educ
y84educ
"""


def load():
    from linearmodels import datasets
    return datasets.load(__file__, 'fertility.csv.bz2')
