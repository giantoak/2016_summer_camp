import pandas
import sqlalchemy
import ujson
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
import cPickle

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor 

m = pandas.read_csv('/home/ubuntu/memex_ad_features/true_positives_price.csv')
eng=sqlalchemy.create_engine('sqlite:////home/ubuntu/2016_summer_camp/classifier/data/dd_dump_v2.db')
#df=pandas.read_sql('select * from dd_id_to_cdr_id;', eng)
#df.to_csv('dr_id_to_cdr_id.csv', index=False)
#true_positives = pandas.read_csv('data/initial/true_positives_text.psv', sep='|')
true_positives = [ujson.loads(i) for i in open('/home/ubuntu/memex_ad_features/true_positives_text.json').readlines()]
true_cdr_ids = set([i['doc_id'] for i in true_positives])
true_negatives = pandas.read_csv('/home/ubuntu/memex_ad_features/negative_sample.csv') 
false_cdr_ids = set(true_negatives['cdr_id'].tolist())
all_cdr_ids = true_cdr_ids.union(false_cdr_ids)
df = pandas.read_csv('dd_id_to_cdr_id.csv')
true_cdr_mapping=df[df['cdr_id'].isin(all_cdr_ids)]
true_cdr_mapping['true'] = true_cdr_mapping['cdr_id'].isin(true_cdr_ids)
true_cdr_mapping.to_csv('classifier_all_ids.csv', index=False)
all_dd_ids = set(true_cdr_mapping['dd_id'].tolist())
print('Filtering dd content')
study_content= []
with open('/home/ubuntu/memexHack1/data/escort_cdr_2/content.tsv') as f:
    for line in f:
        dd_id_str, site, typ, url, text, content = line.split('\t')
        if int(dd_id_str) in all_dd_ids:
            study_content.append(line.strip())

with open('study_content.tsv','w') as f:
    for i in study_content:
        f.write(i + '\n')
content = pandas.read_csv('study_content.tsv','\t', header=None, names=['dd_id','site','type','url','text','content'])
study_data = true_cdr_mapping.merge(content, how='left')
study_data.loc[study_data['content'].isnull(),'content']=''

print('imputing prices')
new_cv = cPickle.load(open('/home/ubuntu/memex_ad_features/price_imputation_text_extractor.pkl','rb'))
rf_new = cPickle.load(open('/home/ubuntu/memex_ad_features/price_imputation_model.pkl','rb'))

X=new_cv.transform(study_data['text'])
price = rf_new.predict(X)
study_data['price_imputed'] = price

def try_json(x):
    try:
        return(ujson.loads(x))
    except:
        return({})

study_extractions = pandas.DataFrame(study_data['content'].apply(try_json).tolist())
study_extractions['age'] = study_extractions['age'].convert_objects(convert_numeric=True)
def split_or_na(x):
    try:
        return(x.split('|'))
    except:
        return([])
ethnicities=set({})
for i in study_extractions['ethnicity']:
    for j in split_or_na(i):
        ethnicities.add(j.strip())
def in_or_na(ethnicity, x):
    try:
        return (ethnicity in x)
    except:
        return(False)

for ethnicity in list(ethnicities):
    study_extractions['ethnicity__' + ethnicity] = study_extractions['ethnicity'].apply(lambda x: in_or_na(ethnicity,x))


study_data=pandas.concat([study_data, study_extractions], axis=1) 
phones = pandas.read_csv('/home/ubuntu/memexHack1/data/escort_cdr_2/phones-combined.tsv', sep='\t', names=['dd_id','phone'])
out= study_data.merge(phones, how='left')
out = out[out['phone'].notnull()]
out['phone_1'] = out['phone'].apply(lambda x: x.split("|")[0])
out = out[out['text'] != '\N']
out.to_csv('classifier_in.csv', sep='\t', encoding='utf-8', index=False)


all_text = out.groupby('phone_1')['text'].apply(lambda x: ' '.join(x.tolist()))
price_data = out.groupby('phone_1')['price_imputed'].describe().unstack()
del price_data['count']
price_data = price_data.rename(columns={i:'price__' + i for i in price_data.columns})
age_data = out.groupby('phone_1')['age'].describe().unstack()
age_data = age_data.rename(columns={i:'age__' + i for i in age_data.columns})
ethnicities = out.groupby('phone_1')[[i for i in out.columns if 'ethnicity__' in i]].mean()
group = pandas.DataFrame(out.groupby('phone_1')['true'].max())


m = pandas.concat([price_data, age_data, ethnicities, group], axis=1)
m.to_csv('phone_level_classifier_in.csv', index=True)
