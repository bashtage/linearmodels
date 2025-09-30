from io import StringIO
import os

import numpy as np
import pandas as pd

from linearmodels.shared.utility import AttrDict

base = os.path.split(os.path.abspath(__file__))[0]


def process_block(results):
    for i, line in enumerate(results):
        if line.startswith("chi2_1"):
            stat_start = i
        elif "* Variance" in line:
            variance_start = i + 2
        elif "* Sigma" in line:
            sigma_start = i + 2
    param_results = results[:stat_start]
    stats = results[stat_start : variance_start - 2]
    variance = results[variance_start : sigma_start - 2]
    sigma = results[sigma_start:]

    def parse_block(block):
        values = pd.read_csv(StringIO("\n".join(block)), header=None)
        nums = np.asarray(values.iloc[:, -1])
        nums = np.reshape(nums, (len(nums) // 3, 3))
        values = pd.DataFrame(
            nums, index=values.iloc[::3, 0], columns=["param", "tstat", "pval"]
        )
        values.index.name = ""
        return values

    params = {}
    block = []
    key = None
    for line in param_results[2:]:
        contents = [s.strip() for s in line.split("\t")]
        if contents[0] != "" and contents[1] == "":
            if key is not None:
                params[key] = parse_block(block)
            key = contents[0]
            block = []
        else:
            block.append(",".join(contents))
    params[key] = parse_block(block)

    stat_values = AttrDict()
    for line in stats:
        contents = line.strip().split("\t")
        if len(contents) > 1 and contents[0] and contents[1]:
            stat_values[contents[0]] = float(contents[1])
    stats = stat_values

    variance = [s.replace("\t", ",") for s in variance]
    header = variance[0]
    block = []
    for line in variance[1:]:
        if ",,," in line:
            continue
        else:
            block.append(line)
    out = pd.read_csv(StringIO("".join([header, *block])))
    out = out.iloc[:, 1:]
    out.index = header.strip().split(",")[1:]
    vcv = out

    sigma = [s.replace("\t", ",") for s in sigma]
    sigma = pd.read_csv(StringIO("".join(sigma)), index_col=0)
    return AttrDict(sigma=sigma, params=params, variance=vcv, stats=stats)


with open(os.path.join(base, "stata-3sls-results.txt")) as stata_results:
    stata_result_contents = stata_results.readlines()
block: list[str] = []
results = {}
key = ""
for line in stata_result_contents:
    if "!!!!" in line:
        if key:
            results[key] = process_block(block)

        key = line.replace("!", "").strip()
        block = []
    else:
        block.append(line)
results[key] = process_block(block)
