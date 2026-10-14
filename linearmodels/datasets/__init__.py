"""
Data sets used in the documentation and the tests.
"""

from os.path import abspath, join, split

from pandas import DataFrame, read_csv


def get_path(f: str) -> str:
    """
    Return the directory that contains a file.

    Parameters
    ----------
    f : str
        Path of the file.

    Returns
    -------
    str
        Path of the directory that contains ``f``.
    """
    return split(abspath(f))[0]


def load(module: str, file_name: str) -> DataFrame:
    """
    Read a bz2 compressed CSV file that is stored with a data set.

    Parameters
    ----------
    module : str
        Path of the module of the data set, usually ``__file__``.
    file_name : str
        Name of the file in the directory of the module.

    Returns
    -------
    DataFrame
        The data in the file.
    """
    return read_csv(join(get_path(module), file_name), compression="bz2")
