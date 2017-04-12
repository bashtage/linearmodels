import pickle

import numpy as np
import pandas as pd

from linearmodels import PanelOLS
from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data

rs = np.random.RandomState()
rs.seed(12345)

OUTPUT_FILE_NAME = 'mc-result-one-way.pkl'
NUM_REPS = 250

# 1-way effects: entity, time, random
# 1-way cluster: entity, time, random, other random
mod_entity = {'entity_effect': True}
mod_time = {'time_effect': True}
mod_entity_time = {'entity_effect': True, 'time_effect': True}
mod_random = {'other_effects': 0}
fit_entity = {'cov_type': 'clustered', 'cluster_entity': True}
fit_time = {'cov_type': 'clustered', 'cluster_time': True}
fit_entity_time = {'cov_type': 'clustered', 'cluster_entity': True, 'cluster_time': True}
fit_nested = {'cov_type': 'clustered', 'clusters': 0}
fit_nested_time = {'cov_type': 'clustered', 'cluster_time': True, 'clusters': 0}
fit_entity_direct_time = {'cov_type': 'clustered', 'cluster_time': True, 'clusters': 0}
fit_random = {'cov_type': 'clustered', 'clusters': 0}
fit_entity_nested = {'cov_type': 'clustered', 'clusters': 0}
fit_random_nested = {'cov_type': 'clustered', 'clusters': 0}

fit_other_random = {'cov_type': 'clustered', 'clusters': 0}

effects = {'entity': mod_entity, 'time': mod_time, 'random': mod_random}
clusters = {'entity': fit_entity, 'time': fit_time, 'random': fit_random, 'other-random': fit_other_random,
            'entity-nested': fit_entity_nested, 'random-nested': fit_random_nested}

effects_cluster = {}
for key in effects:
    for key2 in clusters:
        effects_cluster[key + ':' + key2] = (effects[key], clusters[key2])

options = effects_cluster

# options = {'entity:entity': (mod_entity, fit_entity),
#            'entity:nested': (mod_entity, fit_nested),
#            'entity:nonnested': (mod_entity, fit_random),
#            'entity-time:entity': (mod_entity_time, fit_entity),
#            'entity-time:nested': (mod_entity_time, fit_nested),
#            'entity-time:entity-time': (mod_entity_time, fit_entity_time),
#            'entity-time:direct-time': (mod_entity_time, fit_entity_direct_time),
#            'entity-time:nonnested': (mod_entity_time, fit_random),
#            'time:time': (mod_time, fit_time),
#            'time:entity': (mod_time, fit_entity)}
#
# options = {'entity-time:entity-time': (mod_entity_time, fit_entity_time),
#            'entity-time:direct-time': (mod_entity_time, fit_entity_direct_time)}

final = {}

for key in options:
    joined = {}
    for n in (2000,):
        beta = {}
        std_errs = {}
        std_errs_no = {}
        std_errs_u = {}
        std_errs_u_no = {}
        std_errs_r = {}
        std_errs_r_no = {}
        vals = np.zeros((NUM_REPS, 5, 7))
        for b in range(NUM_REPS):
            if b % 25 == 0:
                print(key, n, b)
            data = generate_data(0.00, 'pandas', ntk=(n, 3, 5), other_effects=1, const=False, rng=rs)
            mo, fo = options[key]

            mod_type, cluster_type = key.split(':')

            y = PanelData(data.y)
            random_effects = np.random.randint(0, n // 3, size=y.dataframe.shape)
            other_random = np.random.randint(0, n // 5, size=y.dataframe.shape)

            if mod_type == 'random':
                effects = y.copy()
                effects.dataframe.iloc[:, :] = random_effects
                mo['other_effects'] = effects

            if cluster_type in ('random', 'other-random', 'entity-nested', 'random-nested'):
                clusters = y.copy()
                if cluster_type == 'random':
                    clusters.dataframe.iloc[:, :] = random_effects
                elif cluster_type == 'other-random':
                    clusters.dataframe.iloc[:, :] = other_random
                elif cluster_type == 'entity_nested':
                    eid = y.entity_ids
                    clusters.dataframe.iloc[:, :] = eid // 3
                elif cluster_type == 'random-nested':
                    clusters.dataframe.iloc[:, :] = random_effects // 2
                fo['clusters'] = clusters

            mod = PanelOLS(data.y, data.x, **mo)
            res = mod.fit(**fo)
            res2 = mod.fit(auto_df=False, count_effects=False, **fo)
            res3 = mod.fit(auto_df=False, count_effects=True, **fo)
            res4 = mod.fit(cov_type='unadjusted')
            res5 = mod.fit(cov_type='unadjusted', auto_df=False, count_effects=False)
            res6 = mod.fit(cov_type='unadjusted', auto_df=False, count_effects=True)

            vals[b] = np.column_stack([res.params, res.std_errors, res2.std_errors,
                                       res3.std_errors, res4.std_errors, res5.std_errors,
                                       res6.std_errors])
        temp = vals.var(0)
        temp[:, 1:] = (vals[:, :, 1:] ** 2).mean(0)
        temp = temp[:, 1:] / temp[:, :1]
        columns = ['clustered:auto', 'clustered:no', 'clustered:yes', 'unadj:auto', 'unadj:no', 'unadj:yes']
        joined[n] = pd.DataFrame(temp, columns=columns)
    final[key] = joined
    with open(OUTPUT_FILE_NAME, 'wb') as result:
        pickle.dump(final, result)




# for key in joined:
#     print(key)
#     temp = joined[key]
#     out = [temp[0].var(1)]
#     for i in range(1, len(temp)):
#         out.append((temp[i] ** 2).mean(1))
#     out = pd.concat(out, 1)
#     print(out)
