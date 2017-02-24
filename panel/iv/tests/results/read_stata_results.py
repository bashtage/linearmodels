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
            vcv_start = i + 1
    params = lines[:param_stop]
    stats = lines[param_stop:stats_stop]
    vcv = lines[vcv_start:]
    vcv[0].insert(0, '')
    vcv = pd.DataFrame(vcv)
    vars = list(vcv.iloc[1:,0])
    vcv = pd.DataFrame(vcv.values[1:,1:],index=vars,columns=vars)
    vcv = vcv.astype('float')
    stats = pd.Series({s[0]: s[1] for s in stats}, name='stats').astype('float')

    var_name = ''
    for p in params:
        if len(p) == 2:
            var_name = p[0]
        else:
            p.insert(0, var_name + '_pval')
    params = pd.DataFrame(params).set_index(0)
    tstats = params.iloc[1::2,0].copy().astype('float')
    params = params.iloc[::2,0].copy().astype('float')
    params.name = 'params'
    tstats = pd.Series(tstats.values, index=list(params.index), name='tstats', copy=True)
    params = pd.concat([params, tstats], 1)

    return AttrDict(params=params, stats=stats, vcv=vcv)

if __name__ == '__main__':
    out = read_result('stata-iv2sls-unadjusted.txt')
    print(out)
