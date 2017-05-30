import os
import subprocess
from collections import OrderedDict
from os.path import join

STATA_PATH = join('C:\\', 'Program Files (x86)', 'Stata13', 'StataMP-64.exe')

dtafile = join(os.getcwd(), 'simulated-panel.dta')

# Permutations
# estimator -> be, fe, or regress to match pooled
# datasets, (nothing), _light, _heavy
# vce options -> conventional (be, fe, re), robust(re, fe, *regress*), ols(*regress*)

configs = {'xtreg {vars}, be vce(conventional)': 'between-conventional-',
           'xtreg {vars}, be wls vce(conventional)': 'between-conventional-wls',
           'xtreg {vars}, fe vce(conventional)': 'fixed_effect-conventional-',
           'xtreg {vars}, fe vce(robust)': 'fixed_effect-robust-',
           'xtreg {vars}, fe vce(cluster firm_id)': 'fixed_effect-cluster-',
           'xtreg {vars}, re vce(conventional)': 'random_effect-conventional-',
           'xtreg {vars}, re vce(robust)': 'random_effect-robust-',
           'xtreg {vars}, re vce(cluster firm_id)': 'random_effect-cluster-',
           'xtreg {vars} [aweight=w], fe vce(conventional)': 'fixed_effect-conventional-weighted',
           'xtreg {vars} [aweight=w], fe vce(robust)': 'fixed_effect-robust-weighted',
           'xtreg {vars} [aweight=w], fe vce(cluster firm_id)': 'fixed_effect-cluster-weighted',
           'regress {vars}, vce(ols)': 'pooled-conventional-',
           'regress {vars}, vce(robust)': 'pooled-robust-',
           'regress {vars}, vce(cluster firm_id)': 'pooled-cluster-',
           'regress {vars} [aweight=w], vce(ols)': 'pooled-conventional-weighted',
           'regress {vars} [aweight=w], vce(robust)': 'pooled-robust-weighted',
           'regress {vars} [aweight=w], vce(cluster firm_id)': 'pooled-cluster-weighted'}

od = OrderedDict()
for key in sorted(configs.keys()):
    od[key] = configs[key]

configs = od

start = """
use {dtafile}, clear \n
xtset firm_id time \n
""".format(dtafile=dtafile)

_sep = '#################!{config}-{ending}!####################'
endings = ['', '_light', '_heavy']
vars = ['y', 'x1', 'x2', 'x3', 'x4', 'x5']

results = """
estout using {outfile}, cells(b(fmt(%13.12g)) t(fmt(%13.12g)) p(fmt(%13.12g))) """

results += """stats(rss df_m df_r r2 rmse mss r2_a ll ll_0 tss N df_b r2_w df_a F F_f Tbar g_min rho sigma sigma_e r2_b r2_o corr sigma_u N_g g_max g_avg, fmt(%13.12g)) unstack append
file open myfile using {outfile}, write append
file write myfile "********* Variance *************" _n
file close myfile
matrix V = e(V)
estout matrix(V, fmt(%13.12g)) using {outfile}, append
"""

section_header = """
file open myfile using {outfile}, write append
file write myfile  _n _n "{separator}" _n
file close myfile
"""

outfile = os.path.join(os.getcwd(), 'stata-panel-simulated-results.txt')

if os.path.exists(outfile):
    os.unlink(outfile)

with open('simulated-results.do', 'w') as stata:
    stata.write(start)
    for config in configs:
        descr = configs[config]
        for ending in endings:
            _vars = ' '.join([v + ending for v in vars])
            command = config.format(vars=_vars)
            sep = _sep.format(config=descr, ending=ending)
            stata.write(section_header.format(outfile=outfile, separator=sep))
            stata.write(command + '\n')
            stata.write(results.format(outfile=outfile))
            stata.write('\n' * 4)

do_file = join(os.getcwd(), 'simulated-results.do')
cmd = [STATA_PATH, '/e', 'do', do_file]
print(' '.join(cmd))
subprocess.call(cmd)
