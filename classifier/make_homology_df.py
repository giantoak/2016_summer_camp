from sys import argv

def main(path_to_db):
    """
    Take the homology DB, pivot the homologies into bools, and return
    :returns `pandas.DataFrame`
    """
    from sqlalchemy import create_engine
    import pandas as pd

    ids_to_get = pd.read_pickle('lattice_df.pkl')._id.values.tolist()

    cdr_ids_str = ','.join(['"{}"'.format(x) for x in ids_to_get])
    query_str = 'select * from cdr_id_to_homology where cdr_id in ({})'.format(cdr_ids_str)

    sql_con = create_engine('sqlite:///{}'.format(path_to_db))

    df = pd.read_sql(query_str, sql_con)

    df = df.pivot(columns='homology').fillna(False)

    df.to_pickle('data/generated/homology_df.pkl')


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python make_homology_df <path_to_db>")
    main(argv[1])
