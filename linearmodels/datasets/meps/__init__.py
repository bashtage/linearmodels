from os import path

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

def load():
    from linearmodels import datasets
    DATA_FILE = path.join(datasets.get_path(__file__), 'meps.csv.bz2')
    return datasets.load(DATA_FILE)
