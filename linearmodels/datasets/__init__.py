from os.path import abspath, join, split
import pandas as pd

def get_path(f):
    return split(abspath(f))[0]


def load(file_name):
    return pd.read_csv(file_name, compression='bz2')
