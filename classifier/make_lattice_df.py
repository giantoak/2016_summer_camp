"""
Read in Lattice Data
Other parsing needed for lattice data?
 will eventually only read the specified train data
... which will be chosen by Kyle Hundeman
We'll need to wrangle a bit with cleanup here
"""

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

def main():
    """

    :returns: `pandas.DataFrame` ---
    """
    from glob import glob
    import multiprocessing as mp
    import pandas as pd

    ls = glob('/Users/pmlandwehr/wkbnch/memex/memex_ad_features/lattice_data/stripped_before_201605/*.json.gz')

    # pool = mp.Pool(10)
    # dfs = p.map(gz_jsonlines_to_df, ls)
    # dfs = pool.map(gz_jsonlines_to_df, ls[0])
    # pool.close()
    # pool.join()
    # df = pd.concat(dfs)

    df = gz_jsonlines_to_df(ls[0])

    df.to_pickle('lattice_df.pkl')

if __name__ == "__main__":
    main()
