"""
This is an adaptation of
201604_memex_qpr/notebooks/go_notebooks/Make Merged Feature DF For Test Data.ipynb
... for the summer hackathon.
That is, it should take local data clean it, and get it ready for use.

This should be pretty similar to the original version *except*:

# More input data
* Currently KH is drawing from the CDR, but we should be using Lattice data for this (in process; can for the moment use the old data)
* Jeff is going to get us price imputations for missing economic data. That's his model
* Integrate any calculations made by Gabriel on the raw data. Once I have the earlier numbers, he should be able to run on it
* Once we have examples, train against them. Sanitization is the goal right now, along with training against the whole of the data.

Issues
---
* Need a good way to deal with JSON lines, esp. Gzipped json lines.
  * Related: when *does* pd.read_json work?
  * Is blaze the best way? approached it before. Best answer is probably to just buy AWS time, but I would like to figure out how to make this laptop hum...

To do
---
* Make sure Gabriel can handle the new lattice data - these are the columns, who needs to deal with what?
* Some of these are problems for Steve Bach's data. Which?
* Melt homology data to bool features

"""


def main():
    """

    :returns: `None` -- saves pickled `pandas.DataFrame` to disk
    """
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_pickle('data/generated/lattice_df.pkl')

    pkls_to_merge = ['giantoak_df.pkl'
                     'homology_df.pkl',
                     'image_df.pkl']

    pbar = tqdm(pkls_to_merge)
    for pkl_df in pbar:
        pbar.set_description(pkl_df)
        tmp_df = pd.read_pickle('data/generated/{}'.format(pkl_df))
        df = pd.merge(df,
                      left_on='_id',
                      right_on='cdr_id',
                      how='left')
        del tmp_df

    df.to_pickle('data/generated/merged_df.pkl')


if __name__ == "__main__":
    main()
