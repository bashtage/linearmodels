import numpy as np
import pandas as pd

from linearmodels import PanelOLS
from linearmodels.tests.panel._utility import generate_data

rs = np.random.RandomState()

joined = {}
for n in (250, 100, 500, 1000, 2500, 5000):
    beta = {}
    std_errs = {}
    std_errs_no = {}
    std_errs_u = {}
    std_errs_u_no = {}
    std_errs_r = {}
    std_errs_r_no = {}
    for b in range(250):
        print(n, b)
        data = generate_data(0.00, 'pandas', ntk=(n, 3, 5), other_effects=1, const=False, rng=rs)
        mod = PanelOLS(data.y, data.x, entity_effect=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        res2 = mod.fit(cov_type='clustered', cluster_entity=True, count_effects=False)
        res3 = mod.fit(cov_type='unadjusted')
        res4 = mod.fit(cov_type='unadjusted', count_effects=False)
        res5 = mod.fit(cov_type='robust')
        res6 = mod.fit(cov_type='robust', count_effects=False)
        beta[b] = res.params
        std_errs[b] = res.std_errors
        std_errs_no[b] = res2.std_errors
        std_errs_u[b] = res3.std_errors
        std_errs_u_no[b] = res4.std_errors
        std_errs_r[b] = res5.std_errors
        std_errs_r_no[b] = res6.std_errors
    beta = pd.DataFrame(beta)
    std_errs = pd.DataFrame(std_errs)
    std_errs_no = pd.DataFrame(std_errs_no)
    std_errs_u = pd.DataFrame(std_errs_u)
    std_errs_u_no = pd.DataFrame(std_errs_u_no)
    std_errs_r = pd.DataFrame(std_errs_r)
    std_errs_r_no = pd.DataFrame(std_errs_r_no)
    joined[n] = (beta, std_errs, std_errs_no,std_errs_u,std_errs_u_no,std_errs_r,std_errs_r_no)

for key in joined:
    print(key)
    temp = joined[key]
    out = [temp[0].var(1)]
    for i in range(1,len(temp)):
        out.append((temp[i] ** 2).mean(1))
    out = pd.concat(out,1)
    print(out)
