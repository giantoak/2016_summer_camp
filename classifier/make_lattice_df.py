"""
Read in Lattice Data
Other parsing needed for lattice data?
 will eventually only read the specified train data
... which will be chosen by Kyle Hundeman
We'll need to wrangle a bit with cleanup here
"""
from sys import argv


def gz_jsonlines_to_df(fpath):
    import gzip
    import ujson as json
    from pandas import DataFrame
    jsns = []
    for line in gzip.open(fpath):
        jsn = json.loads(line)
        # Checks/Transforms can occur here to see if we want to keep the data
        jsns.append(jsn)

    return DataFrame.from_records(jsns)


def main(path_to_cdr_ids, path_to_lattice):
    """

    :param str path_to_cdr_ids:
    :param str path_to_lattice:
    :returns: `None` -- saves pickled `pandas.DataFrame` to disk
    """
    from glob import glob
    import multiprocessing as mp
    import pandas as pd

    cdr_ids_to_get = set(open(path_to_cdr_ids).readlines())

    ls = glob('{}/*.json.gz'.format(path_to_lattice))
    # pool = mp.Pool(10)
    # dfs = p.map(gz_jsonlines_to_df, ls)
    # dfs = pool.map(gz_jsonlines_to_df, ls[0])
    # pool.close()
    # pool.join()
    # df = pd.concat(dfs)
    df = gz_jsonlines_to_df(ls[0])

    df.to_pickle('data/generated/lattice_df.pkl')

if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python make_lattice_df.py <path_to_cdr_ids> <path_to_lattice>")
    main(argv[1], argv[2])
