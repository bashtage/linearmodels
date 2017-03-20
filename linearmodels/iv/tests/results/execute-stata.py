import os
import subprocess
from os.path import join

STATA_PATH = join('C:\\', 'Program Files (x86)', 'Stata13', 'StataMP-64.exe')

start = """
use http://www.stata-press.com/data/r13/hsng, clear \n
"""
iv_tempplate = """
ivregress {method} rent pcturban (hsngval = faminc i.region){variance_option}
estout using {outfile}, cells(b(fmt(%13.12g)) t(fmt(%13.12g))) """

iv_tempplate += """stats(r2 r2_a p mss rss rmse {extra}, fmt(%13.12g)) unstack append
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

section_header = """
file open myfile using {outfile}, write append
file write myfile  _n _n "########## !{method}-{desc}-{small}! ##########" _n
file close myfile
"""

methods = ['2sls', 'liml', 'gmm']
outfile = os.path.join(os.getcwd(), 'stata-iv-housing-results.txt')
if os.path.exists(outfile):
    os.unlink(outfile)
variance_options = [', vce(unadjusted)', ', vce(robust)', ', vce(cluster division)']
descr = ['unadjusted', 'robust', 'cluster']

with open('temp.do', 'w') as stata:
    stata.write(start)
    for small in (True, False):
        for method in methods:
            for vo, desc in zip(variance_options, descr):
                small_text = 'small' if small else 'asymptotic'
                stata.write(section_header.format(outfile=outfile, method=method, desc=desc,
                                                  small=small_text))
                desc += '-small' if small else ''
                vo += ' small' if small else ''
                of = outfile.format(method=method, descr=desc)
                extra = ' J ' if method == 'gmm' else ' kappa '
                extra += ' F p ' if small else ' chi2 p '
                cmd = iv_tempplate.format(outfile=of, variance_option=vo, method=method,
                                          extra=extra)
                if 'gmm' in method:
                    cmd = cmd.replace('vce', 'wmatrix')
                stata.write(cmd)
                if 'gmm' in method:
                    stata.write(gmm_extra.format(outfile=of))
                stata.write('\n')

do_file = join(os.getcwd(), 'temp.do')
cmd = [STATA_PATH, '/e', 'do', do_file]
print(' '.join(cmd))
subprocess.call(cmd)
