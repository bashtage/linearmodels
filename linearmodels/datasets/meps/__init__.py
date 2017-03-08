from os import path

def load():
    from linearmodels import datasets
    DATA_FILE = path.join(datasets.get_path(__file__), 'meps.csv.bz2')
    return datasets.load(DATA_FILE)
