import pandas as pd


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_result(name):
    with open(name, 'r') as stata:
        lines = stata.readlines()

    lines = [l.strip().split('\t') for l in lines]
    lines = lines[2:]
    for i in range(len(lines)):
        if lines[i][0] == 'r2':
            param_stop = i
        if lines[i][0].startswith('***'):
            stats_stop = i
            cov_start = i + 1
    params = lines[:param_stop]
    stats = lines[param_stop:stats_stop]
    cov = lines[cov_start:]
    cov[0].insert(0, '')
    cov = pd.DataFrame(cov)
    vars = list(cov.iloc[1:, 0])
    cov = pd.DataFrame(cov.values[1:, 1:], index=vars, columns=vars)
    cov = cov.astype('float')
    stats = pd.Series({s[0]: s[1] for s in stats}, name='stats').astype('float')

    var_name = ''
    for p in params:
        if len(p) == 2:
            var_name = p[0]
        else:
            p.insert(0, var_name + '_pval')
    params = pd.DataFrame(params).set_index(0)
    tstats = params.iloc[1::2, 0].copy().astype('float')
    params = params.iloc[::2, 0].copy().astype('float')
    params.name = 'params'
    tstats = pd.Series(tstats.values, index=list(params.index), name='tstats', copy=True)
    params = pd.concat([params, tstats], 1)
    index = list(params.index)
    for i, v in enumerate(index):
        if v == '_cons':
            index[i] = 'const'
    params.index = index
    cov.index = index
    cov.columns = index
    tstats = params.tstats
    params = params.params

    return AttrDict(params=params, tstats=tstats, stats=stats, cov=cov, rsquared=stats.r2,
                    rsquared_adj=stats.r2_a, model_ss=stats.mss, resid_ss=stats.rss,
                    f_statistic=stats.chi2, f_statistic_pval=stats.p)


if __name__ == '__main__':
    out = read_result('stata-iv2sls-unadjusted.txt')
