#import happybase
#hb_con = happybase.Connection('memex-hbase-master')
#ad_table = hb_con.table('ad_characteristics')  
#with open('cdr_ids_from_hbase.txt','w') as f:
    #for line in ad_table.scan():
        #f.write(line[0].strip() + '\n')


#Get cdr`
import requests
from  elasticsearch import Elasticsearch
#con = elasticsearch.Connection(
es = Elasticsearch(
        'cdr-es.istresearch.com',
        http_auth=('cdr-memex', '5OaYUNBhjO68O7Pn'),
        port=9200,
        use_ssl=True
        )

q={
    "fields" : ["_id"],
    "query" : {
        "match_all" : {}
        }
}
from elasticsearch.helpers import scan
#m=scan(es, q)
m=scan(es, q, index='memex-domains', doc_type='escorts')
with open('cdr_ids_from_cdr.txt','w') as f:
    for doc in m:
        f.write(doc['_id'] + '\n')
#r=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/_mapping',auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
