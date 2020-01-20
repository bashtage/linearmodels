from os.path import abspath, join, split

import pandas as pd
from pandas import DataFrame


def get_path(f: str) -> str:
    return split(abspath(f))[0]


def load(module: str, file_name: str) -> DataFrame:
    return pd.read_csv(join(get_path(module), file_name), compression="bz2")
