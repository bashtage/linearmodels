from os.path import abspath, join, split

from pandas import DataFrame, read_csv


def get_path(f: str) -> str:
    return split(abspath(f))[0]


def load(module: str, file_name: str) -> DataFrame:
    return read_csv(join(get_path(module), file_name), compression="bz2")
