from collections import defaultdict

import pandas as pd

from linearmodels.utility import AttrDict


def repl_const(df):
    index = list(df.index)
    replace_cols = list(df.columns) == index
    for i, v in enumerate(index):
        if v == '_cons':
            index[i] = 'const'
    df.index = index
    if replace_cols:
        df.columns = index
    for c in df:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def parse_file(name):
    blocks = defaultdict(list)
    current_key = ''
    with open(name, 'r') as stata:
        for line in stata:
            if line.strip() == '':
                continue
            if line.startswith('###'):
                current_key = line.split('!')[1]
                continue
            blocks[current_key].append(line)
    return blocks


def parse_block(block):
    block = [l.strip().split('\t') for l in block]
    params = []
    cov = []
    weight_mat = []
    last = 0
    for i, line in enumerate(block):
        last = i
        if len(line) == 2:
            params.append(line)
        elif len(line) == 1:
            if line[0].startswith('***'):
                break
            try:
                float(line[0])
                params[-1].append(line[0])
            except:
                pass
    params = pd.DataFrame(params, columns=['variable', 'params', 'tstats'])
    params = repl_const(params.set_index('variable'))
    stats = params.loc[params.tstats.isnull(), 'params']
    params = params.loc[params.tstats.notnull()]

    for i, line in enumerate(block[last + 2:]):
        if len(line) == 1 and line[0].startswith('***'):
            break
        cov.append(line)
    cov[0].insert(0, 'variable')
    last += i + 2

    cov = pd.DataFrame(cov[1:], columns=cov[0])
    cov = repl_const(cov.set_index('variable'))

    if len(block) > (last + 1):
        weight_mat = block[last + 2:]
        weight_mat[0].insert(0, 'variable')
        weight_mat = pd.DataFrame(weight_mat[1:], columns=weight_mat[0])
        weight_mat = repl_const(weight_mat.set_index('variable'))

    return AttrDict(params=params, cov=cov, weight_mat=weight_mat, stats=stats)


def finalize(params, stats, cov, weight_mat):
    tstats = params.tstats
    params = params.params
    out = AttrDict(params=params, tstats=tstats, stats=stats, cov=cov, weight_mat=weight_mat)
    for key in stats.index:
        out[key] = stats[key]
    fixes = {'model_ss': 'mss', 'resid_ss': 'rss', 'rsquared': 'r2', 'rsquared_adj': 'r2_a'}
    for key in fixes:
        if fixes[key] in out:
            out[key] = out[fixes[key]]
        else:
            out[key] = None
    if 'chi2' in out:
        out['f_statistic'] = out['chi2']
    elif 'F' in out:
        out['f_statistic'] = out['F']
    else:
        out['f_statistic'] = None

    return out


def process_results(filename):
    blocks = parse_file(filename)
    for key in blocks:
        out = parse_block(blocks[key])
        blocks[key] = finalize(out.params, out.stats, out.cov, out.weight_mat)
    return blocks


if __name__ == '__main__':
    import os

    blocks = parse_file(os.path.join(os.getcwd(), 'stata-iv-simulated-results.txt'))
    for key in blocks:
        out = parse_block(blocks[key])
        finalize(out['params'], out['stats'], out['cov'], out['weight_mat']).keys()
