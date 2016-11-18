import happybase
import requests
import ipdb
hb_con = happybase.Connection('memex-hbase-master')
ad_table = hb_con.table('ad_characteristics')  
phone_table = hb_con.table('phone_characteristics')  
hbase_tables = requests.get('http://memex-hbase-master:8080/').content.split('\n')
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
m=requests.get('http://memex-hbase-master:8080/ad_characteristics/%s/%s' % (doc, val))
