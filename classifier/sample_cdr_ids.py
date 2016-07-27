import pandas
import numpy as np
import sqlalchemy
db_file = 'data/dd_dump_v2.db'
sqlite_connection=sqlalchemy.create_engine('sqlite:///{}'.format(db_file))
df=pandas.read_sql('select distinct(phone) from dd_id_to_phone;', sqlite_connection)
seed = 0
size=5000
np.random.seed(0)
phone_sample = df.loc[np.random.choice(df.index, size)]
phone_list = "('%s')" % "','".join(phone_sample['phone'].tolist())
cdr_ids = pandas.read_sql('''select dd_id_to_cdr_id.cdr_id, dd_id_to_cdr_id.dd_id, dd_id_to_phone.phone from dd_id_to_phone inner join dd_id_to_cdr_id on dd_id_to_cdr_id.dd_id=dd_id_to_phone.dd_id where phone in %s''' % phone_list, sqlite_connection)
cdr_ids.to_csv('data/generated/negative_sample.csv', index=False)
