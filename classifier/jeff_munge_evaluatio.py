import pandas
import sqlalchemy
import ujson
import pickle
import ipdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
import cPickle

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor 

eng=sqlalchemy.create_engine('sqlite:////home/ubuntu/2016_summer_camp/classifier/data/dd_dump_v2.db')
#df=pandas.read_sql('select * from dd_id_to_cdr_id;', eng)
#df.to_csv('dr_id_to_cdr_id.csv', index=False)
#true_positives = pandas.read_csv('data/initial/true_positives_text.psv', sep='|')
true_positives = [ujson.loads(i) for i in open('/home/ubuntu/memex_ad_features/true_positives_text_evaluation.json').readlines()]
cdr_ids_to_cluster_ids = pandas.DataFrame([{'cluster_id':i['cluster_id'], 'doc_id':i['doc_id']} for i in true_positives])
#ipdb.set_trace()
true_cdr_ids = set([i['doc_id'] for i in true_positives])
all_cdr_ids = true_cdr_ids
df = pandas.read_csv('dd_id_to_cdr_id.csv')
true_cdr_mapping=df[df['cdr_id'].isin(all_cdr_ids)]
true_cdr_mapping['true'] = true_cdr_mapping['cdr_id'].isin(true_cdr_ids)
true_cdr_mapping.to_csv('evaluation_all_ids.csv', index=False)
all_dd_ids = set(true_cdr_mapping['dd_id'].tolist())
if False:
    print('Filtering dd content')
    study_content= []
    with open('/home/ubuntu/memexHack1/data/escort_cdr_2/content.tsv') as f:
        for line in f:
            dd_id_str, site, typ, url, text, content = line.split('\t')
            if int(dd_id_str) in all_dd_ids:
                study_content.append(line.strip())

    with open('evaluation_content.tsv','w') as f:
        for i in study_content:
            f.write(i + '\n')
content = pandas.read_csv('evaluation_content.tsv','\t', header=None, names=['dd_id','site','type','url','text','content'])
study_data = true_cdr_mapping.merge(content, how='left')
study_data.loc[study_data['content'].isnull(),'content']=''
prices = pandas.read_csv('/home/ubuntu/memexHack1/ad_price_ad_level.csv')
study_data=study_data.merge(prices[['ad_id','price_per_hour']], how='left', left_on='dd_id', right_on='ad_id')

print('imputing prices')
new_cv = cPickle.load(open('/home/ubuntu/memex_ad_features/price_imputation_text_extractor.pkl','rb'))
rf_new = cPickle.load(open('/home/ubuntu/memex_ad_features/price_imputation_model.pkl','rb'))

X=new_cv.transform(study_data['text'])
price = rf_new.predict(X)
study_data['price_imputed'] = price

print('imputing ages')
cv = cPickle.load(open('/home/ubuntu/memex_ad_features/age_imputation_text_extractor.pkl','rb'))
rf_age = cPickle.load(open('/home/ubuntu/memex_ad_features/age_imputation_model.pkl','rb'))
X=cv.transform(study_data['text'])
age = rf_age.predict(X)
study_data['age_imputed'] = age

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
out.index = range(len(out))
out = out.reindex()
out.to_csv('evaluation_in.csv', sep='\t', encoding='utf-8', index=False)

cluster_column = 'phone_1'

# Begin work featurizing text:w
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
tfidf = TfidfVectorizer(ngram_range=(1,2))
ad_text = tfidf.fit_transform(out['text'])
tsvd=TruncatedSVD(n_components=150)
X_text_sub=tsvd.fit_transform(ad_text)
X_text_sub = pandas.DataFrame(X_text_sub, columns = ['text_feature__' + str(i) for i in range(X_text_sub.shape[1])])
cPickle.dump(X_text_sub,open('svd_text.pkl','wb'))
phone_text=pandas.concat([out[cluster_column], X_text_sub], axis=1)
if phone_text.shape[0] != out.shape[0]:
    raise('Error concatentating phone text')
phone_text_averages=phone_text.groupby(cluster_column).mean()
phone_text_averages.to_csv('phone_text_averages_evaluation.csv')


out = out.rename(columns={'price_per_hour':'price'})
all_text = out.groupby(cluster_column)['text'].apply(lambda x: ' '.join(x.tolist()))
price_data_imputed = out.groupby(cluster_column)['price_imputed'].describe().unstack()
del price_data_imputed['count']
price_data_imputed = price_data_imputed.rename(columns={i:'price_imputed__' + i for i in price_data_imputed.columns})
price_data = out.groupby(cluster_column)['price'].describe().unstack()
price_data = price_data.rename(columns={i:'price__' + i for i in price_data.columns})
age_data = out.groupby(cluster_column)['age'].describe().unstack()
age_data = age_data.rename(columns={i:'age__' + i for i in age_data.columns})
imputed_age_data = out.groupby(cluster_column)['age_imputed'].describe().unstack()
imputed_age_data = imputed_age_data.rename(columns={i:'age_imputed__' + i for i in imputed_age_data.columns})
ethnicities = out.groupby(cluster_column)[[i for i in out.columns if 'ethnicity__' in i]].mean()
group = pandas.DataFrame(out.groupby(cluster_column)['true'].max())


m = pandas.concat([price_data, age_data, imputed_age_data, ethnicities, group, price_data_imputed], axis=1)
m.to_csv('phone_level_classifier_in_evaluation.csv', index=True)

d = pandas.concat([m, phone_text_averages], axis=1)
d.to_csv('evaluation_features.csv')

model = pickle.load(open('rf_all_features.pkl','rb'))
needed_data=pandas.read_csv('X_model_data.csv', index_col=[cluster_column], nrows=2)

for col in needed_data.columns:
    if col not in d.columns:
        d[col]=0
d=d[needed_data.columns]
scores = pandas.DataFrame({'score':model.predict_proba(d.fillna(0))[:,1] }, index=d.index)

score_output = out[['cdr_id','phone_1']].merge(scores, left_on='phone_1',right_index=True)
print(out.shape)
print(score_output.shape)
score_output=score_output.merge(cdr_ids_to_cluster_ids, left_on='cdr_id', right_on='doc_id')
print(score_output.shape)
score_output[['cluster_id','score']].tolist()
with open('evaluation_scores.json','w') as f:
    for line in json.loads(score_output[['cluster_id','score']].T.to_json()).values():
        f.write(ujson.dumps(line) + '\n')
