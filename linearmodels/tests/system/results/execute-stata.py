"""
Important cases

1. Base
2. Small sample adjustment
3. Constraints across equations
"""
import os
import subprocess

import pandas as pd

from linearmodels.tests.system._utility import generate_data

STATA_PATH = os.path.join("C:\\", "Program Files (x86)", "Stata13", "StataMP-64.exe")
OUTFILE = os.path.join(os.getcwd(), "stata-sur-results.txt")

header = [
    r'use "C:\git\linearmodels\linearmodels\tests\system\results\simulated-sur.dta", clear'
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

data = generate_data(n=200, k=3, p=[2, 3, 4], const=True, seed=0)
common_data = generate_data(n=200, k=3, p=3, common_exog=True, seed=1)
missing_data = generate_data(n=200, k=3, p=[2, 3, 4], const=True, seed=2)

cmds = []
for i, dataset in enumerate((data, common_data, missing_data)):
    base = f"mod_{i}"
    cmd = ""
    for j, key in enumerate(dataset):
        dep = dataset[key]["dependent"]
        dep = pd.DataFrame(dep, columns=[base + f"_y_{j}"])
        exog = dataset[key]["exog"][:, 1:]
        exog_cols = [base + f"_x_{j}{k}" for k in range(exog.shape[1])]
        exog = pd.DataFrame(exog, columns=exog_cols)
        if i != 1 or j == 0:
            cmd += " ( " + " ".join(list(dep.columns) + list(exog.columns)) + " ) "
        else:
            new_cmd = cmd[: cmd.find(")") + 1]
            new_cmd = new_cmd.replace("mod_1_y_0", f"mod_1_y_{j}")
            cmd += new_cmd
    cmds.append(cmd)

outcmds = {}
key_bases = ["basic", "common", "missing"]
for key_base, cmd in zip(key_bases, cmds):
    base = "sureg " + cmd
    ss = base + ", small dfk"
    comp = cmd.replace("(", "").strip().split(")")[:-1]
    comp = [c.strip() for c in comp]
    deps = [c.split(" ")[0] for c in comp]
    first = [c.split(" ")[1] for c in comp]
    vals = {}
    i = 0
    for d, f in zip(deps, first):
        vals["y" + str(i)] = d
        vals["x" + str(i)] = f
        i += 1

    constraint = """
constraint 1 [{y0}]{x0} = [{y1}]{x1}
constraint 2 [{y0}]{x0} = [{y2}]{x2}
"""
    cons = constraint.format(**vals) + base + ", const (1 2)"
    outcmds[key_base + "-base"] = base
    outcmds[key_base + "-ss"] = ss
    outcmds[key_base + "-constrained"] = cons

sep = """
file open myfile using {outfile}, write append \n
file write myfile  "#################!{key}!####################" _n \n
file close myfile\n
"""
with open("sur.do", "w") as stata_file:
    stata_file.write("\n".join(header) + "\n")
    for outcmd in outcmds:
        stata_file.write(sep.format(outfile=OUTFILE, key=outcmd))
        stata_file.write(outcmds[outcmd] + "\n")
        stata_file.write(f"\n{output}\n\n")
        stata_file.write("\n" * 5)

if os.path.exists(OUTFILE):
    os.unlink(OUTFILE)
do_file = os.path.join(os.getcwd(), "sur.do")
stata_cmd = [STATA_PATH, "/e", "do", do_file]
print(" ".join(stata_cmd))
subprocess.call(stata_cmd)
