from sys import argv

"""
* Get CDR image ids for images used in ads
* Get similar CDR IDs in common set
* Generate cluster DF
* return
"""

HBASE_ADDR = 'memex-hbase-master:8080'


def _hbase_row_value(table, row_id, key_id):
    """
    :param str table: The name of the MEMEX HBase table
    :param str row_id: The row to get from the table
    :param str key_id: The key to get from the row
    :returns: `str` -- The value in the desired key, or `None`
    """
    import requests
    try:
        hbase_url = 'http://{}/{}/{}/{}'.format(HBASE_ADDR, table, row_id, key_id)
        r = requests.get(hbase_url)
        if r.status_code == 200:
            return r.text
    except:
        pass

    return None


def main():
    import pandas as pd

    df = pd.DataFrame({})
    df.to_pickle('data/generated/image_df.pkl')

if __name__ == "__main__":
    main()
