import ujson
import pandas
import ipdb
import numpy as np
out_list = []
with open('cdr_id_and_phones_from_cdr.txt','r') as f:
    with open('phone_1_list.txt','w') as outfile:
        for line in f:
            doc = ujson.loads(line)
            outfile.write(doc['phones'][0] + '\n')

phones = pandas.read_csv('phone_1_list.txt', header=None)
unique_phones = phones[0].unique()

num_phones = 12000
np.random.seed(2)
sampled_phones = np.random.choice(unique_phones, num_phones)

sampled_phone_set = set(sampled_phones.astype('str').tolist())
out_list = []
with open('cdr_id_and_phones_from_cdr.txt','r') as f:
    for line in f:
        doc = ujson.loads(line)
        if doc['phones'][0] in sampled_phone_set:
            out_list.append({'cdr_id':doc['cdr_id'], 'phone_1':doc['phones'][0]})
        else:
            continue
out_df = pandas.DataFrame(out_list)
out_df.to_csv('negative_sample_cdr_id_and_phone_1.csv', index=False)
#sampled_cdr_ids = b[b['phone_1'].isin(sampled_phones)]['cdr_id']
#sampled_cdr_ids.to_csv('negative_sample_cdr_ids.txt', index=False)
