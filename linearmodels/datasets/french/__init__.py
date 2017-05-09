import pandas as pd

DESCR = """
Data from Ken French's data library
http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

dates    Year and Month of Return
MktRF    Market Factor
SMB      Size Factor
HML      Value Factor
Mom      Momentum Factor
RF       Risk-free rate
NoDur    Industry: Non-durables
Durbl    Industry: Durables
Manuf    Industry: Manufacturing
Enrgy    Industry: Energy
Chems    Industry: Chemicals
BusEq    Industry: Business Equipment
Telcm    Industry: Telecoms
Utils    Industry: Utilities
Shops    Industry: Retail
Hlth     Industry: Health care
Money    Industry: Finance
Other    Industry: Other
S1V1     Small firms, low value
S1V3     Small firms, medium value
S1V5     Small firms, high value
S3V1     Size 3, value 1
S3V3     Size 3, value 3
S3V5     Size 3, value 5
S5V1     Large firms, Low value
S5V3     Large firms, medium value
S5V5     Large Firms, High value
S1M1     Small firms, losers
S1M3     Small firms, neutral
S1M5     Small firms, winners
S3M1     Size 3, momentum 1
S3M3     Size 3, momentum 3
S3M5     Size 3, momentum 5
S5M1     Large firms, losers
S5M3     Large firms, neutral
S5M5     Large firms, winners
"""


def load():
    from linearmodels import datasets
    data = datasets.load(__file__, 'french.csv.bz2')
    data['dates'] = pd.to_datetime(data.dates)
    return data
