from io import StringIO
import os

import pandas as pd

from linearmodels.shared.utility import AttrDict

filename = "stata-panel-simulated-results.txt"

cwd = os.path.split(os.path.abspath(__file__))[0]
blocks = {}
block: list[str] = []
key = ""
with open(os.path.join(cwd, filename)) as results:
    for line in results.readlines():
        _line = line.strip()
        if not _line:
            continue
        if "###!" in _line:
            if key:
                blocks[key] = block
            block = []
            key = _line.split("!")[1]
        block.append(_line)
    if block:
        blocks[key] = block


def parse_block(block):
    params = {}
    stats = {}
    for i, line in enumerate(block):
        if "b/t" in line:
            params_start = i + 1
        if "rss" in line:
            stats_start = i
        if "** Variance **" in line:
            variance_start = i + 1

    for i in range(params_start, stats_start, 3):
        name, value = block[i].split("\t")
        value = float(value)
        tstat = float(block[i + 1])
        pvalue = float(block[i + 1])
        params[name] = pd.Series({"param": value, "tstat": tstat, "pvalue": pvalue})
    params = pd.DataFrame(params).sort_index()
    for i in range(stats_start, variance_start - 1):
        if "\t" in block[i]:
            name, value = block[i].split("\t")
            stats[name] = float(value)
        else:
            stats[block[i]] = None
    stats = pd.Series(stats)
    var = "\n".join(block[variance_start + 1 :])
    variance = pd.read_csv(StringIO("," + var.replace("\t", ",")))
    index = variance.pop(variance.columns[0])
    index.name = None
    variance.index = index
    out = AttrDict(variance=variance, params=params.T)
    for key in stats.index:
        out[key] = stats.loc[key]

    return out


def data():
    data_blocks = {}
    for key, block_val in blocks.items():
        data_blocks[key] = parse_block(block_val)
    return data_blocks


if __name__ == "__main__":
    print(data())
