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

true_positives = [ujson.loads(i) for i in open('/home/ubuntu/memex_ad_features/true_positives_text.json').readlines()]
true_cdr_ids = set([i['doc_id'] for i in true_positives])
#eng=sqlalchemy.create_engine('sqlite:////home/ubuntu/2016_summer_camp/classifier/data/dd_dump_v2.db')
##df=pandas.read_sql('select * from dd_id_to_cdr_id;', eng)
##df.to_csv('dr_id_to_cdr_id.csv', index=False)
##true_positives = pandas.read_csv('data/initial/true_positives_text.psv', sep='|')

#true_negatives = pandas.read_csv('/home/ubuntu/memex_ad_features/negative_sample.csv') 
#false_cdr_ids = set(true_negatives['cdr_id'].tolist())
#all_cdr_ids = true_cdr_ids.union(false_cdr_ids)
#df = pandas.read_csv('dd_id_to_cdr_id.csv')
#true_cdr_mapping=df[df['cdr_id'].isin(all_cdr_ids)]
#true_cdr_mapping['true'] = true_cdr_mapping['cdr_id'].isin(true_cdr_ids)
#true_cdr_mapping.to_csv('classifier_all_ids.csv', index=False)
#all_dd_ids = set(true_cdr_mapping['dd_id'].tolist())
#true_dd_ids = set(true_cdr_mapping[true_cdr_mapping['true']]['dd_id'])

#if False:
    #print('Filtering dd content')
    #study_content= []
    #with open('/home/ubuntu/memexHack1/data/escort_cdr_2/content.tsv') as f:
        #for line in f:
            #dd_id_str, site, typ, url, text, content = line.split('\t')
            #if int(dd_id_str) in all_dd_ids:
                #study_content.append(line.strip())

    #with open('study_content_text_only.tsv','w') as f:
        #for i in study_content:
            #f.write(i + '\n')

training_content = pandas.read_csv('study_content_text_only.tsv','\t', header=None, names=['dd_id','site','type','url','text','content'])
phones = pandas.read_csv('/home/ubuntu/memexHack1/data/escort_cdr_2/phones-combined.tsv', sep='\t', names=['dd_id','phone'])
training_content= training_content.merge(phones, how='left')
del phones
training_content = training_content[training_content['phone'].notnull()]
training_content['phone_1'] = training_content['phone'].apply(lambda x: x.split("|")[0])
training_content = training_content[training_content['text'] != '\N']
training_content.index = range(len(training_content))
training_content = training_content.reindex()
training_content['group'] = 'train_false'
training_content.loc[training_content['dd_id'].isin(true_cdr_ids),'group'] = 'train_true'
training_content['cluster_id'] = training_content['phone_1'] # Explicitly call a phone a cluster
training_content['doc_id'] = training_content['dd_id']
training_content = training_content[['doc_id','cluster_id','group','text']]
ipdb.set_trace()

evaluation = [ujson.loads(i) for i in open('/home/ubuntu/memex_ad_features/true_positives_text_evaluation.json').readlines()]
evaluation_content=pandas.DataFrame(evaluation)
evaluation_content['group']='test'
evaluation_content = evaluation_content.rename(columns={'content':'text'})
evaluation_content = evaluation_content[['doc_id','cluster_id','group','text']]
print('Distribution of cluster id sizes in evaluation:')
print(evaluation_content.groupby('cluster_id').size().describe())
print('Distribution of cluster id sizes in false set:')
print(training_content[training_content['group']=='train_false'].groupby('cluster_id').size().describe())
print('Distribution of cluster id sizes in false set:')
print(training_content[training_content['group']=='train_true'].groupby('cluster_id').size().describe())

study_data = pandas.concat([training_content, evaluation_content])
study_data.index = range(len(study_data))
study_data=study_data.reindex()

study_data.loc[study_data['content'].isnull(),'content']=''
# Merge truth positive, truth negative, and withheld test here

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
def homogenize_clusters(df):
    if len(df['group'].unique())> 1:
        if False:
            print('___')
            print(df['group'].value_counts())
        df['group']=df['group'].value_counts().index[0] # return most popular group
        return(df)
    else:
        return(df)
out = study_data.copy()

out.to_csv('evaluation_text_in.csv', sep='\t', encoding='utf-8', index=False)
cluster_column = 'cluster_id'

# Begin work featurizing text:w
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
a=(('tfidf', TfidfVectorizer(ngram_range=(1,2))), ('svd',TruncatedSVD(n_components=150)))
pipeline=Pipeline(a)
pipeline.fit(out.loc[out['group'] != 'test', 'text'])
X_text_sub = pipeline.transform(out['text'])
#tfidf = TfidfVectorizer(ngram_range=(1,2))
#ad_text = tfidf.fit_transform(out.loc[out['group'] != 'test', 'text'])
#tsvd=TruncatedSVD(n_components=150)
#X_text_sub=tsvd.transform(out['text'])
X_text_sub = pandas.DataFrame(X_text_sub, columns = ['text_feature__' + str(i) for i in range(X_text_sub.shape[1])])
cPickle.dump(pipeline,open('svd_text_pipeline.pkl','wb'))
cPickle.dump(X_text_sub,open('svd_text_features.pkl','wb'))
phone_text=pandas.concat([out[cluster_column], X_text_sub], axis=1)
if phone_text.shape[0] != out.shape[0]:
    raise('Error concatentating phone text')
phone_text_averages=phone_text.groupby(cluster_column).mean()
#phone_text_averages.to_csv('phone_text_averages_evaluation.csv')


price_data_imputed = out.groupby(cluster_column)['price_imputed'].describe().unstack()
del price_data_imputed['count']
price_data_imputed = price_data_imputed.rename(columns={i:'price_imputed__' + i for i in price_data_imputed.columns})
age_data_imputed = out.groupby(cluster_column)['age_imputed'].describe().unstack()
del age_data_imputed['count']
age_data_imputed = age_data_imputed.rename(columns={i:'age_imputed__' + i for i in age_data_imputed.columns})
#ethnicities = out.groupby(cluster_column)[[i for i in out.columns if 'ethnicity__' in i]].mean()
group = pandas.DataFrame(out.groupby(cluster_column)['true'].apply(lambda x: x.iloc[0]))


m = pandas.concat([price_data_imputed, age_data_imputed, group ], axis=1)
m.to_csv('phone_level_classifier_in_text_evaluation.csv', index=True)

d = pandas.concat([m, phone_text_averages], axis=1)
d.to_csv('evaluation_text_features.csv')

# Begin work on model fitting
text_cols = {x for x in df.columns if x.find('text_feature__') > -1}
price_imputed_cols = {x for x in df.columns if x.find('price_imputed__') > -1}
age_imputed_cols = {x for x in df.columns if x.find('age_imputed__') > -1}
model = pickle.load(open('rf_text_all_features.pkl','rb'))
rf = RandomForestClassifier(oob_score=True,
                            random_state=2,
                            n_estimators=100,
                            n_jobs=-1,
                            class_weight="balanced")
needed_data=pandas.read_csv('X_text_model_data.csv', index_col=[cluster_column], nrows=2)

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
with open('evaluation_scores.jl','w') as f:
    for line in json.loads(score_output[['cluster_id','score']].T.to_json()).values():
        f.write(ujson.dumps(line) + '\n')
