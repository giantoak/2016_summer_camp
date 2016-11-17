#import happybase
#hb_con = happybase.Connection('memex-hbase-master')
#ad_table = hb_con.table('ad_characteristics')  
#hbase_tables = requests.get('http://memex-hbase-master:8080/').content.split('\n')
##with open('cdr_ids_from_hbase.txt','w') as f:
    ##for line in ad_table.scan():
        ##f.write(line[0].strip() + '\n')
import ujson
import datetime


#Get cdr`
import requests
import ipdb
from  elasticsearch import Elasticsearch
import time
#con = elasticsearch.Connection(
es = Elasticsearch(
        'cdr-es.istresearch.com',
        http_auth=('cdr-memex', '5OaYUNBhjO68O7Pn'),
        port=9200,
        use_ssl=True
        )

#q={
    #'aggs' : {
        #'states': {
            #'terms':{
                #'field':'extractions.phonenumber.results',
                #'size': 100000
                #}
            #}
        #}
    #}
q={
    "fields" : ["_id"],
    "query" : {
        "match_all" : {}
        }
    }
#m=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/_search', data=json.dumps(q),auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
q={
    "fields" : ["_id", "_source"],
    "query" : {
        "match_all" : {}
        }
    }
from elasticsearch.helpers import scan
#m=scan(es, q)
count = 0
start = datetime.datetime.now()
m=scan(es, q, index='memex-domains', doc_type='escorts', raise_on_error=False)
with open('cdr_id_and_phones_from_cdr.txt','w') as f:
    doc = m.next()
    while doc:
        #data = requests.get('https://cdr-es.istresearch.com:9200/%s/%s/%s' % (doc['_index'], doc['_type'],doc['_id']),auth=('cdr-memex','5OaYUNBhjO68O7Pn')).json()
        #if len(data['_source'].keys()) > 12:
        doc_id=doc['_id']
        count += 1
        try:
            phones = doc['_source']['extractions']['phonenumber']['results']
        except KeyError:
            try:
                doc = m.next()
            except:
                print('Sleeping 60s after miss')
                time.sleep(60)
                try:
                    doc = m.next()
                except:
                    print('Sleeping 60s after miss')
                    time.sleep(60)
                    doc = m.next()
            continue
        if count%10000 ==0:
            print('Completed %s in %s' % (count, datetime.datetime.now()-start))
        f.write(ujson.dumps({'phones':phones,'cdr_id':doc_id}) + '\n')
        try:
            doc = m.next()
        except:
            print('Sleeping 60s after miss')
            time.sleep(60)
            try:
                doc = m.next()
            except:
                print('Sleeping 60s after miss')
                time.sleep(60)
                doc = m.next()
#r=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/_mapping',auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
#r=requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/%s' % doc['_id'],auth=('cdr-memex','5OaYUNBhjO68O7Pn'))
