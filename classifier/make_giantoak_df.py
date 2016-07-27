"""
Reads Giant Oak data produced by Jeff / Gabriel
"""

def main(path_to_cdr_ids):
    """

    :param str path_to_cdr_ids:
    :returns: `None` -- saves pickled `pandas.DataFrame` to disk
    """
    # This should replace our use of the lattice_df.
    import pandas as pd

    cdr_ids_to_get = set(open(path_to_cdr_ids).readlines())

    df = pd.DataFrame({})
    df.to_pickle('data/generated/giantoak_df.pkl')

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python make_giantoak_df.py <path_to_cdr_ids>")
        return
    main()
