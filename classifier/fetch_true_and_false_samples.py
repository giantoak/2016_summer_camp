'''
This file should fetch true and false sets from CDR and from HBase
'''
import json
import requests
import pandas
import ipdb
import numpy as np
out_list = []
false_sample = pandas.read_csv('negative_sample_cdr_id_and_phone_1.csv')
false_cdr_ids = false_sample['cdr_id'].tolist()
q={
    "fields" : ["_id"],
    "query" : {
        "match_all" : {}
        }
    }
#with open('negative_sample_full_docs.jl','w') as f:
    #for cdr_id in false_cdr_ids:
        #try:
            #q={
            #"query": {
                #"match": {
                    #"_id": cdr_id.strip()
                    #}
                #}
            #}
            #m=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/_search',data=json.dumps(q),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
            #f.write(json.dumps(m.json()) + '\n')
        #except:
            #continue

#m=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/_search', data=json.dumps(q),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
domains = ["memex-domains_2015.11", "memex-domains_2015.10", "memex-domains_2015.12", "memex-domains_2014.09", "memex-domains_2014.08", "memex-domains_2014.06", "memex-domains_2016.09", "memex-domains_2016.08", "memex-domains_2016.03", "memex-domains_2016.02", "memex-domains_2016.01", "memex-domains_2016.07", "memex-domains_2016.06", "memex-domains_2016.05", "memex-domains_2016.04", "memex-domains_2015.08", "memex-domains_2015.09", "memex-domains_2015.02", "memex-domains_2015.03", "memex-domains_2015.01", "memex-domains_2015.06", "memex-domains_2016.dev", "memex-domains_2015.04", "memex-domains_2015.07", "memex-domains_2016.09.02", "memex-domains_2015.05", "memex-domains_2014.12", "memex-domains_2014.10", "memex-domains_2014.11", "memex-domains_2016.10", "memex-domains_2016.11"]
#m = requests.get('https://cdr-es.istresearch.com:9200/%s/escorts/%s' % (domains[0],false_cdr_ids),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
with open('negative_sample_full_docs_v2.jl','w') as f:
    for cdr_id in false_cdr_ids:
        try:
            q={
            "query": {
                "match": {
                    "_id": cdr_id.strip()
                    }
                }
            }
            m=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/_search',data=json.dumps(q),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
            f.write(json.dumps(m.json()) + '\n')
            #for d in domains:
                #m = requests.head('https://cdr-es.istresearch.com:9200/%s/escorts/%s' % (d,cdr_id.strip()),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
                #if m.status_code == 200:
                    #r = requests.get('https://cdr-es.istresearch.com:9200/%s/escorts/%s' % (d,cdr_id.strip()),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
                    #f.write(json.dumps(r.json()) + '\n')
                    #continue
        except:
            continue
