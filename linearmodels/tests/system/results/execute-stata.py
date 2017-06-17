"""
Important cases

1. Base
2. Small sample adjustement
3. Constraints across equations
"""
import os
import subprocess

import pandas as pd

from linearmodels.tests.system._utility import generate_data

STATA_PATH = os.path.join('C:\\', 'Program Files (x86)', 'Stata13', 'StataMP-64.exe')
OUTFILE = os.path.join(os.getcwd(), 'stata-sur-results.txt')

header = [r'use "C:\git\linearmodels\linearmodels\tests\system\results\simulated-sur.dta", clear']
outcmds = []

all_stats = 'estout using {outfile}, cells(b(fmt(%13.12g)) t(fmt(%13.12g)) p(fmt(%13.12g))) stats('
stats = ['chi2_{0}', 'F_{0}', 'p_{0}', 'df_m{0}', 'mss_{0}', 'r2_{0}', 'rss_{0}']
for i in range(1, 4):
    all_stats += ' '.join(map(lambda s: s.format(i), stats)) + ' '
all_stats += ') append'
output = all_stats + '\n' + """

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
output = output.format(outfile=OUTFILE)

data = generate_data(n=200, k=3, p=[2, 3, 4], const=True, seed=0)
common_data = generate_data(n=200, k=3, p=3, common_exog=True, seed=1)
missing_data = generate_data(n=200, k=3, p=[2, 3, 4], const=True, seed=2)

cmds = []
for i, dataset in enumerate((data, common_data, missing_data)):
    base = 'mod_{0}'.format(i)
    cmd = ''
    for j, key in enumerate(dataset):
        dep = dataset[key]['dependent']
        dep = pd.DataFrame(dep, columns=[base + '_y_{0}'.format(j)])
        exog = dataset[key]['exog'][:, 1:]
        exog_cols = [base + '_x_{0}{1}'.format(j, k) for k in range(exog.shape[1])]
        exog = pd.DataFrame(exog, columns=exog_cols)
        if i != 1 or j == 0:
            cmd += ' ( ' + ' '.join(list(dep.columns) + list(exog.columns)) + ' ) '
        else:
            new_cmd = cmd[:cmd.find(')') + 1]
            new_cmd = new_cmd.replace('mod_1_y_0', 'mod_1_y_{0}'.format(j))
            cmd += new_cmd
    cmds.append(cmd)

outcmds = {}
key_bases = ['basic', 'common', 'missing']
for key_base, cmd in zip(key_bases,cmds):
    base = 'sureg ' + cmd
    ss = base + ', small dfk'
    comp = cmd.replace('(', '').strip().split(')')[:-1]
    comp = list(map(lambda s: s.strip(), comp))
    deps = [c.split(' ')[0] for c in comp]
    first = [c.split(' ')[1] for c in comp]
    vals = {}
    i = 0
    for d, f in zip(deps, first):
        vals['y' + str(i)] = d
        vals['x' + str(i)] = f
        i += 1

    constraint = """
constraint 1 [{y0}]{x0} = [{y1}]{x1}
constraint 2 [{y0}]{x0} = [{y2}]{x2}
"""
    cons = constraint.format(**vals) + base + ', const (1 2)'
    outcmds[key_base + '-base'] = base
    outcmds[key_base + '-ss'] = ss
    outcmds[key_base + '-constrained'] = cons

with open('sur.do', 'w') as stata_file:
    stata_file.write('\n'.join(header) + '\n')
    for outcmd in outcmds:
        stata_file.write('file open myfile using {outfile}, write append \n'.format(outfile=OUTFILE))
        stata_file.write('file write myfile  "#################!{key}!####################" _n \n'.format(key=outcmd))
        stata_file.write('file close myfile\n')
        stata_file.write(outcmds[outcmd] + '\n')
        stata_file.write('\n{0}\n\n'.format(output))
        stata_file.write('\n'*5)

if os.path.exists(OUTFILE):
    os.unlink(OUTFILE)
do_file = os.path.join(os.getcwd(), 'sur.do')
cmd = [STATA_PATH, '/e', 'do', do_file]
print(' '.join(cmd))
subprocess.call(cmd)
