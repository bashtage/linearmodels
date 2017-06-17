import os
from io import StringIO

import pandas as pd

from linearmodels.utility import AttrDict

filename = 'stata-sur-results.txt'

cwd = os.path.split(os.path.abspath(__file__))[0]
results = open(os.path.join(cwd, filename))

with open(os.path.join(cwd, filename), 'r') as results_file:
    results = results_file.readlines()

blocks = {}
block = []
key = ''
for line in results:
    if '###!' in line:
        if block:
            blocks[key] = block
            block = []
        key = line.strip().split('!')[1]
        block = []
    block.append(line)
blocks[key] = block

block = blocks[list(blocks.keys())[0]]


def split_block(block):
    block = block[:]
    for i, line in enumerate(block):
        if '** Sigma **' in line:
            sigma = block[i + 2:]
            block = block[:i]
    for i, line in enumerate(block):
        if '** Variance **' in line:
            variance = block[i + 2:]
            block = block[:i]
    for i, line in enumerate(block):
        if 'chi2_' in line or 'F_' in line:
            stats = block[i:]
            params = block[:i]
            break
    return AttrDict(sigma=process_sigma(sigma),
                    variance=process_variance(variance),
                    stats=process_stats(stats),
                    params=process_params(params))


def process_stats(stats):
    sio = StringIO(''.join(stats))
    values = pd.read_csv(sio, sep='\t', header=None, index_col=0, engine='c')
    values.columns = ['value']
    values.index.name = 'stat'
    values = values.astype('float64')
    return values


def process_sigma(sigma):
    sio = StringIO(''.join(sigma))
    values = pd.read_csv(sio, sep='\t', index_col=0)
    return values


def process_variance(variance):
    key = ''
    new = [variance[0]]
    for i, line in enumerate(variance[1:]):
        if '\t\t' in line:
            key = line.split('\t')[0]
            continue
        new.append(key + '_' + line)
    sio = StringIO(''.join(new))
    values = pd.read_csv(sio, sep='\t', index_col=0)
    values.index = [i.replace('__','_') for i in values.index]
    values.columns = [c.replace(':','_').replace('__', '_') for c in values.columns]
    return values


def process_params(params):
    reformatted = []
    values = []
    key = var_name = ''
    for line in params[3:]:
        if '\t\n' in line:
            if values:
                new_line = key + '_' + var_name + '\t' + '\t'.join(values)
                reformatted.append(new_line)
                values = []
            key = line.split('\t')[0]
            continue
        if line.split('\t')[0].strip():
            if values:
                new_line = key + '_' + var_name + '\t' + '\t'.join(values)
                reformatted.append(new_line)
                values = []
            var_name = line.split('\t')[0].strip()
        values.append(line.split('\t')[1].strip())
    new_line = key + '_' + var_name + '\t' + '\t'.join(values)

    reformatted.append(new_line)
    sio = StringIO('\n'.join(reformatted))
    values = pd.read_csv(sio, sep='\t', index_col=0, header=None)
    new_index = []
    for idx in list(values.index):
        new_index.append(idx.replace('__', '_'))
    values.index = new_index
    values.index.name = 'param'
    values.columns = ['param', 'tstat', 'pval']
    return values


stata_results = {block: split_block(blocks[block]) for block in blocks}
