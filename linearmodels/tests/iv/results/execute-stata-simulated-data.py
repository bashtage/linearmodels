import os
import subprocess
from itertools import product
from os.path import join

STATA_PATH = join('C:\\', 'Program Files (x86)', 'Stata13', 'StataMP-64.exe')

dtafile = join(os.getcwd(), 'simulated-data.dta')
start = """
use {dtafile}, clear \n
tsset time \n
""".format(dtafile=dtafile)

model = """
ivregress {method} {depvar} {exog_var} ({endog_var} = {instr}) {weight_opt}, {variance_option} {other_option}
"""

methods = ['2sls', 'liml', 'gmm']
depvars = ['y_unadjusted', 'y_robust', 'y_clustered', 'y_kernel']
variance_options = ['vce(unadjusted)', 'vce(robust)', 'vce(cluster cluster_id)',
                    'vce(hac bartlett 12)']
depvar_with_var = list(zip(depvars, variance_options))
exog_vars = ['x3 x4 x5']
endog_vars = ['x1', 'x1 x2']
instr = ['z1', 'z1 z2']
other_options = ['', 'small', 'noconstant', 'small noconstant', 'small center',
                 'center', 'center noconstant', 'small center noconstant']
weight_options = [' ', ' [aweight=weights] ']
inputs = [methods, depvar_with_var, exog_vars, endog_vars, instr, other_options, weight_options]
configs = []
for val in product(*inputs):
    method, dvo, exog, endog, instr, other_opt, weight_opt = val
    depvar, var_opt = dvo
    if (len(endog) > len(instr)) or (other_opt.find('center') >= 0 and method != 'gmm'):
        continue
    if method == 'gmm':
        var_opt = var_opt.replace('vce', 'wmatrix')

    configs.append({'method': method,
                    'depvar': depvar,
                    'exog_var': exog,
                    'endog_var': endog,
                    'instr': instr,
                    'variance_option': var_opt,
                    'other_option': other_opt,
                    'weight_opt': weight_opt})

results = """
estout using {outfile}, cells(b(fmt(%13.12g)) t(fmt(%13.12g))) """

results += """stats(r2 r2_a mss rss rmse {extra}, fmt(%13.12g)) unstack append
file open myfile using {outfile}, write append
file write myfile  "********* Variance *************" _n
file close myfile
matrix V = e(V)
estout matrix(V, fmt(%13.12g)) using {outfile}, append
"""

gmm_extra = """
file open myfile using {outfile}, write append
file write myfile  "********* GMM Weight *************" _n
file close myfile
matrix W = e(W)
estout matrix(W, fmt(%13.12g)) using {outfile}, append
"""

m = '{method}-num_endog_{num_endog}-num_exog_{num_exog}-num_instr_{num_instr}'
m = m + '-weighted_{weighted}-{variance}-{other}'
section_header = """
file open myfile using {outfile}, write append
file write myfile  _n _n "########## !"""
section_header += m
section_header += """! ##########" _n
file close myfile
"""

outfile = os.path.join(os.getcwd(), 'stata-iv-simulated-results.txt')

if os.path.exists(outfile):
    os.unlink(outfile)


def count_vars(v):
    return sum(map(lambda s: s == ' ', v)) + 1


with open('simulated-results.do', 'w') as stata:
    stata.write(start)
    for config in configs:
        sec_header = {'method': config['method'],
                      'num_endog': count_vars(config['endog_var']),
                      'num_exog': count_vars(config['exog_var']),
                      'num_instr': count_vars(config['instr']),
                      'variance': config['variance_option'],
                      'other': config['other_option'].replace(' ', '_'),
                      'outfile': outfile,
                      'weighted': 'aweight' in config['weight_opt']}
        stata.write(section_header.format(**sec_header))
        stata.write(model.format(**config))

        small = config['other_option'].find('small') >= 0
        extra = ' J ' if config['method'] == 'gmm' else ' kappa '
        extra += ' F p ' if small else ' chi2 p '
        stata.write(results.format(outfile=outfile, extra=extra))

        if config['method'] == 'gmm':
            stata.write(gmm_extra.format(outfile=outfile))
        stata.write('\n')

do_file = join(os.getcwd(), 'simulated-results.do')
cmd = [STATA_PATH, '/e', 'do', do_file]
print(' '.join(cmd))
subprocess.call(cmd)
