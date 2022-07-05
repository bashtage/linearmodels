"""
Important cases

"""
import os
import subprocess
from typing import List

import pandas as pd

from linearmodels.tests.system._utility import generate_simultaneous_data

data = generate_simultaneous_data()
all_cols: List[str] = []
out = []
for key in data:
    eqn = data[key]
    for key in ("exog", "endog"):
        vals = eqn[key]
        for col in vals:
            if col in all_cols:
                continue
            else:
                out.append(vals[col])
                all_cols.append(col)
out_df = pd.concat(out, 1, sort=False)
if "const" in out_df:
    out_df.pop("const")
out_df.to_stata("simulated-3sls.dta", write_index=False)
SEP = """

file open myfile using {outfile}, write append
file write myfile  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {method} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" _n
file close myfile

"""

# add , 2sls to get non GLS estimator
CMD = """
reg3 (dependent_0 dependent_1 dependent_2 exog_1 exog_2 exog_3 exog_4 exog_5) ///
     (dependent_1 dependent_0 dependent_2 exog_1 exog_2 exog_3 exog_6 exog_7) ///
     (dependent_2 dependent_0 dependent_1 exog_1 exog_2 exog_3 exog_8 exog_9), {method}
"""

STATA_PATH = os.path.join("C:\\", "Program Files (x86)", "Stata15", "StataMP-64.exe")
OUTFILE = os.path.join(os.getcwd(), "stata-3sls-results.txt")

header = [
    r'use "C:\git\linearmodels\linearmodels\tests\system\results\simulated-3sls.dta", clear'
]

all_stats = "estout using {outfile}, cells(b(fmt(%13.12g)) t(fmt(%13.12g)) p(fmt(%13.12g))) stats("
stats = ["chi2_", "F_", "p_", "df_m", "mss_", "r2_", "rss_"]
for i in range(1, 4):
    all_stats += " ".join([f"{s}{i}" for s in stats]) + " "
all_stats += ") append"
output = (
    all_stats
    + "\n"
    + """

file open myfile using {outfile}, write append
file write myfile  "*********** Variance ****************" _n
file close myfile

matrix V = e(V)

estout matrix(V, fmt(%13.12g)) using {outfile}, append

file open myfile using {outfile}, write append
file write myfile  "*********** Sigma ****************" _n
file close myfile

matrix Sigma = e(Sigma)

estout matrix(Sigma, fmt(%13.12g)) using {outfile}, append
"""
)
output = output.format(outfile=OUTFILE)

methods = ("3sls", "2sls", "ols", "sur", "3sls ireg3")

with open("three-sls.do", "w") as stata_file:
    stata_file.write("\n\n".join(header))

    for method in methods:
        stata_file.write(SEP.format(method=method, outfile=OUTFILE))
        stata_file.write("\n\n".join([CMD.format(method=method), output]))

if os.path.exists(OUTFILE):
    os.unlink(OUTFILE)

do_file = os.path.join(os.getcwd(), "three-sls.do")
cmd = [STATA_PATH, "/e", "do", do_file]
print(" ".join(cmd))
subprocess.call(cmd)
