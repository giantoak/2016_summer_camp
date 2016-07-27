from sys import argv


def main(path_to_cdr_ids, path_to_db):
    """
    Take the homology DB, pivot the homologies into bools, and return
    :param sr path_to_cdr_ids:
    :param str path_to_db:
    :returns `pandas.DataFrame`
    """
    from sqlalchemy import create_engine
    import pandas as pd

    cdr_ids_to_get = set(open(path_to_cdr_ids).readlines())

    cdr_ids_str = ','.join(['"{}"'.format(x) for x in cdr_ids_to_get])
    query_fmt = 'select * from cdr_id_to_homology where cdr_id in ({})'.format

    sql_con = create_engine('sqlite:///{}'.format(path_to_db))

    df = pd.read_sql(query_fmt(cdr_ids_str), sql_con)

    df = df.pivot(columns='homology').fillna(False)

    df.to_pickle('data/generated/homology_df.pkl')


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python make_homology_df <path_to_db> <path_to_cdr_ids")
    main(argv[1], argv[2])
