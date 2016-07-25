def main():
    """
    Take the homology DB, pivot the homologies into bools, and return
    :returns `pandas.DataFrame`
    """
    from sqlalchemy import create_engine
    import pandas as pd

    ids_to_get = pd.read_pickle('lattice_data.pkl')._id.values().tolist()

    cdr_ids_str = ','.join(['"{}"'.format(x) for x in ids_to_get])
    query_str = 'select * from cdr_id_to_homology where cdr_id in ({})'.format(cdr_ids_str)

    sql_con = create_engine('sqlite:///homology_160722.db')

    df = pd.read_sql(query_str, sql_con)

    df = df.pivot(columns='homology').fillna(False)

    df.to_pickle('homology_data.pkl')


if __name__ == "__main__":
    main()
