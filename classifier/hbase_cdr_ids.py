import happybase
import requests
import json
import ipdb
hb_con = happybase.Connection('memex-hbase-master')
ad_table = hb_con.table('ad_characteristics')  
phone_table = hb_con.table('phone_characteristics')  
lattice_table = hb_con.table('lattice_hdfs')  
hbase_tables = requests.get('http://memex-hbase-master:8080/').content.split('\n')
keep_ids = [json.loads(i)['_id'] for i  in open('data/fall_2016/CP1_train_ads_labelled_fall2016.jsonl').readlines()]
out_list=[]
for i in keep_ids:
    response = requests.get('http://memex-hbase-master:8080/lattice_hbase/%s/' % (i))
    if response.content.strip() == 'Not found':
        out_list.append({'_id':i, 'content':''})
    else:
        ipdb.set_trace()
        out_list.append({'_id':i, 'content':response.content})

with open('temp_lattice.txt','w') as f:
    for line in out_list:
        f.write(json.dumps(line) + '\n')
ipdb.set_trace()

with open('temp_lattice.txt','w') as f:
    for line in lattice_table.scan():
        out_dict = {}
        out_dict['_id'] = line[0]
        if out_dict['_id'] in keep_ids:
            try:
                out_dict['content'] = line[1]['content:results']
            except:
                pass
            try:
                out_dict['title'] = line[1]['title:results']
            except:
                pass
            if out_dict.has_key('title') or out_dict.has_key('content'):
                f.write(json.dumps(out_dict) + '\n')
with open('temp_ad_characteristiscs.txt','w') as f:
    for line in ad_table.scan():
        #ipdb.set_trace()
        f.write(line[0].strip() + '\n')
with open('temp_phone_characteristiscs.txt','w') as f:
    for line in phone_table.scan():
        #ipdb.set_trace()
        f.write(line[0].strip() + '\n')

val='values:state'
doc='000019ACA83B7467112B61D2F616D54534D405DE984A13FA54A21E1516EE181F'
