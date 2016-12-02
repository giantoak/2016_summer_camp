import ipdb
import ujson
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

cluster_id = 'cluster_id'
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, keylist):
        self.keylist = keylist

    def fit(self, x, y=None):
        self._feature_names = x.columns
        return self
    def get_feature_names(self):
        return(self._feature_names)

    def transform(self, data_dict):
        return data_dict[self.keylist]

class GroupbyMax(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_column=None, max_column=None):
        self.grouping_column=grouping_column
        self.max_column=max_column

    def fit(self, x, y=None):
        #self._feature_names = x.columns
        return self
    def transform(self, data):
        maxes=data.groupby(self.grouping_column)[self.max_column].max()
        return(maxes)

class Uniquifier(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_column=None):
        self.grouping_column=grouping_column

    def fit(self, x, y=None):
        #self._feature_names = x.columns
        return self
    def transform(self, data):
        out = data.groupby(self.grouping_column[0]).size()
        #ipdb.set_trace()
        return(pandas.DataFrame(out.index))

class Summarizer(BaseEstimator, TransformerMixin):
    '''
    initialized with a grouping column where we do a groupb_by max of matches

    Calls to .transform grab only the named column and transform them to dummies
    '''
    def __init__(self, grouping_column=None):
        self.grouping_column=grouping_column

    def fit(self, x, y=None):
        #self._feature_names = x.columns
        return self
    #def get_feature_names(self):
        #return(self._feature_names)

    def transform(self, data):
        main_column = [i for i in data.columns if i != self.grouping_column][0]
        summary_stats=data.groupby(self.grouping_column)[main_column].describe().unstack()
        del summary_stats['count']
        summary_stats = summary_stats.rename(columns = {i:'%s__%s' % (main_column, i) for i in summary_stats.columns})
        summary_stats = summary_stats.fillna(-1)
        return summary_stats

pipeline = Pipeline([

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            #('phone_getter', Uniquifier([cluster_id])),

            #('bad_identifier', GroupbyMax([cluster_id],['bad'])),

            # Pipeline for computing price stats
            ('price', Pipeline([
                ('price_getter', ItemSelector([cluster_id,'price_imputed'])),
                ('averager', Summarizer(grouping_column=cluster_id)),
            ])),

            # Pipeline for computing age stats
            ('age', Pipeline([
                ('age_getter', ItemSelector([cluster_id,'age_imputed'])),
                ('averager', Summarizer(grouping_column=cluster_id)),
            ])),


        ],
    )),

    ('rf', RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True))
    # Fit a logistic regression with crossvalidation
])

# Training data
out_list = []
count = 0
with open('data/fall_2016/CP1_train_ads_labelled_fall2016.jsonl') as f:
    for line in f:
        data = ujson.loads(line)
        out={}
        out['doc_id']=data['_id']
        out['content'] = data['extracted_text']
        out['class'] = data['class']
        out['cluster_id'] = data['cluster_id']
        if data.has_key('extractions'):
            if data['extractions'].has_key('phonenumber'):
                #out['rate_info'] = data['extractions']['rate']
                out[cluster_id] = data['extractions']['phonenumber']['results'][0]
                
        out_list.append(out)
        print(count)
        count += 1

import pandas
true_df = pandas.DataFrame(out_list)
true_df['group'] = 'train'
true_df.to_csv('true_positives_with_phones_text.csv', index=False, sep='\t', encoding='utf-8')
del out_list

#out_list = []
#count = 0
#with open('data/initial/cp1_evaluation_data.json') as f:
    #for line in f:
        #data = ujson.loads(line)
        #out={}
        #out['doc_id']=data['doc_id']
        #out['content'] = data['extracted_text']
        #if data.has_key('extractions'):
            #if data['extractions'].has_key('phonenumber'):
                ##out['rate_info'] = data['extractions']['rate']
                #out[cluster_id] = data['extractions']['phonenumber']['results'][0]
        #out_list.append(out)
        #print(count)
        #count += 1

#eval_df = pandas.DataFrame(out_list)
#eval_df['group'] = 'test'
df = true_df
df = df[df['content'].notnull()]
del true_df

import cPickle
new_cv = cPickle.load(open('models/price_imputation_text_extractor.pkl','rb'))
rf_new = cPickle.load(open('models/price_imputation_model.pkl','rb'))
X=new_cv.transform(df['content'])
df['price_imputed'] = rf_new.predict(X)

age_cv = cPickle.load(open('models/age_imputation_text_extractor.pkl','rb'))
age_rf = cPickle.load(open('models/age_imputation_model.pkl','rb'))
X_age = age_cv.transform(df['content'])
df['age_imputed'] = age_rf.predict(X_age)
#df.to_csv('df_with_content.csv', sep='\t', encoding='utf-8',index=False)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
a=(
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))), 
        ('svd',TruncatedSVD(n_components=150)),
        #('rf', RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True))
        )
text_pipeline=Pipeline(a)
text_svd=text_pipeline.fit_transform(df['content'])
text_df=pandas.DataFrame(text_svd)
text_df=text_df.rename(columns={i:'text__%s' % i for i in text_df.columns})
text_df=pandas.concat([df[['doc_id','cluster_id','class']], text_df], axis=1)
X=text_df.groupby('cluster_id')[[i for i in text_df.columns if 'text__' in i]].mean()
y=text_df.groupby('cluster_id')['class'].max()
text_rf = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=2, oob_score=True)
text_rf.fit(X,y)
text_df.to_csv('svd_features.csv', index=False)
cPickle.dump(text_pipeline,open('models/svd_transform.pkl','wb'))
cPickle.dump(text_rf,open('models/svd_text_model.pkl','wb'))
ipdb.set_trace()

del df['content']
# Do imputation group-bys
price_data_imputed = df.groupby(cluster_id)['price_imputed'].describe().unstack()
del price_data_imputed['count']
price_data_imputed = price_data_imputed.rename(columns={i:'price_imputed__' + i for i in price_data_imputed.columns})
age_data_imputed = df.groupby(cluster_id)['age_imputed'].describe().unstack()
del age_data_imputed['count']
age_data_imputed = age_data_imputed.rename(columns={i:'age_imputed__' + i for i in age_data_imputed.columns})

model_df = pandas.concat([price_data_imputed, age_data_imputed], axis=1)
model_df=model_df.fillna(-1) # Fill missing at -1

train_indices = df['class'].apply(lambda x: True)
train_phones = df.loc[train_indices, cluster_id].tolist()
true_train_phones = df.loc[df['class']==1,cluster_id].tolist()
#test_phones = df.loc[df['group']=='test',cluster_id].tolist()
true_train_index = pandas.Series(model_df.index.isin(true_train_phones), index=model_df.index) 
train_index = pandas.Series(model_df.index.isin(train_phones), index=model_df.index) 
#test_index = pandas.Series(model_df.index.isin(test_phones), index=model_df.index) 

test_indices = df['group'] == 'test'

aggregated_group = df.groupby(cluster_id)['class'].apply(lambda x: x.max())
print(aggregated_group.describe())

save_df = model_df.copy()
#save_df.loc[true_train_index,'group'] = 'train_true'
#save_df.loc[test_index,'group'] = 'test'
#save_df.loc[(~test_index) & (~true_train_index),'group'] = 'train_false'
save_df.to_csv('qpr_model_fit.csv')
df.to_csv('qpr_imputation.csv', index=False)  


df['bad'] = df['class']
#X = out[:,2:]
#y = pandas.Series(out[:,1]).astype('int')
cluster_id='cluster_id'
y=df.groupby(cluster_id).apply(lambda x: (x['class']).max()).astype('int')    
pipeline.fit(df[df['group'] != 'test'], y)
#rf = RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True)
#rf.fit(out, y_)
probs = pipeline.predict_proba(df[df['group'] != 'test'])
cPickle.dump(pipeline,open('ht_giantoak_ht_model.pkl','wb'))
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
#from sklearn.pipeline import Pipeline
#a=(('tfidf', TfidfVectorizer(ngram_range=(1,2))), ('svd',TruncatedSVD(n_components=150)))
#pipeline=Pipeline(a)
#pipeline.fit(df['content'])
#X_text_sub = pipeline.transform(df['content'])
#X_text_sub = pandas.DataFrame(X_text_sub, columns = ['text_feature__' + str(i) for i in range(X_text_sub.shape[1])])
#model_df = pandas.concat([price_data_imputed, age_data_imputed, X_text_sub], axis=1)

#text_cols = {x for x in df.columns if x.find('text_feature__') > -1}
#price_imputed_cols = {x for x in df.columns if x.find('price_imputed__') > -1}
#age_imputed_cols = {x for x in df.columns if x.find('age_imputed__') > -1}
#from sklearn.ensemble import RandomForestClassifier
#pipeline.fit(model_df[model_df['group']!='test'],y) 
##rf = RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True)
#rf.fit(model_df.loc[train_index,[i for i  in model_df.columns if i not in text_cols]], true_train_index[train_index])
#rf.score(model_df.loc[train_index,[i for i  in model_df.columns if i not in text_cols]], true_train_index[train_index])

#predicted_test=rf.predict_proba(model_df.loc[test_index,[i for i  in model_df.columns if i not in text_cols]])[:,1]

######################################################################
# Compute output scores for test data
out_list = []
count = 0
with open('data/fall_2016/CP1_test_ads_fall2016.jsonl') as f:
    for line in f:
        data = ujson.loads(line)
        out={}
        out['doc_id']=data['_id']
        out['content'] = data['extracted_text']
        out['cluster_id'] = data['cluster_id']
        #if data.has_key('extractions'):
            #if data['extractions'].has_key('phonenumber'):
                ##out['rate_info'] = data['extractions']['rate']
                #out[cluster_id] = data['extractions']['phonenumber']['results'][0]
        out_list.append(out)
        print(count)
        count += 1
eval_df = pandas.DataFrame(out_list)
eval_df['price_imputed'] = rf_new.predict(new_cv.transform(eval_df['content']))
eval_df['age_imputed'] = age_rf.predict(age_cv.transform(eval_df['content']))
del out_list
eval_scores = []
for cluster_id in eval_df['cluster_id'].value_counts().index:
    eval_probs = pipeline.predict_proba(eval_df[eval_df['cluster_id']==cluster_id])[:,1]
    eval_scores.append({ 'cluster_id':cluster_id, 'score':eval_probs[0] })
with open('data/fall_2016/cp1_scored_imputed.jsonl','w') as f:
    for line in eval_scores:
        f.write(json.dumps(line) + '\n')

prices= eval_df.groupby('cluster_id')[['price_imputed','age_imputed']].describe().unstack()
del prices['price_imputed']['count']
del prices['age_imputed']['count']
prices.to_csv('price_imputation_features.csv')

eval_svd=text_pipeline.transform(eval_df['content']) 
svd_df=pandas.DataFrame(eval_svd)
svd_df=svd_df.rename(columns={i:'text__%s' % i for i in svd_df.columns})
svd_df=pandas.concat([eval_df[['doc_id','cluster_id']], svd_df], axis=1)
X=svd_df.groupby('cluster_id')[[i for i in svd_df.columns if 'text__' in i]].mean()
svd_eval_probas=text_rf.predict_proba(X)[:,1]
svd_output_df = pandas.DataFrame({'score':svd_eval_probas, 'cluster_id':X.index})
with open('data/fall_2016/cp1_scored_svd.jsonl','w') as f:
    for index, row in svd_output_df.iterrows():
        f.write(json.dumps({'score':row['score'], 'cluster_id':row['cluster_id']}) + '\n')



######################################################################
# Compute output scores for Summer QPR test data
out_list = []
count = 0
with open('/home/ubuntu/2016_summer_camp/classifier/data/initial/cp1_evaluation_data.json') as f:
    for line in f:
        data = ujson.loads(line)
        out={}
        out['doc_id']=data['doc_id']
        out['content'] = data['extracted_text']
        out['cluster_id'] = data['cluster_id']
        out_list.append(out)
        print(count)
        count += 1
summar_eval_df = pandas.DataFrame(out_list)
summar_eval_df['price_imputed'] = rf_new.predict(new_cv.transform(summar_eval_df['content']))
summar_eval_df['age_imputed'] = age_rf.predict(age_cv.transform(summar_eval_df['content']))
del out_list
summar_eval_scores = []
for cluster_id in summar_eval_df['cluster_id'].value_counts().index:
    summar_eval_probs = pipeline.predict_proba(summar_eval_df[summar_eval_df['cluster_id']==cluster_id])[:,1]
    summar_eval_scores.append({ 'cluster_id':cluster_id, 'score':summar_eval_probs[0] })
with open('data/fall_2016/summer_scored_imputed.jsonl','w') as f:
    for line in summar_eval_scores:
        f.write(json.dumps(line) + '\n')

prices= summar_eval_df.groupby('cluster_id')[['price_imputed','age_imputed']].describe().unstack()
del prices['price_imputed']['count']
del prices['age_imputed']['count']
prices.to_csv('summer_price_imputation_features.csv')

summar_eval_svd=text_pipeline.transform(summar_eval_df['content']) 
svd_df=pandas.DataFrame(summar_eval_svd)
svd_df=svd_df.rename(columns={i:'text__%s' % i for i in svd_df.columns})
svd_df=pandas.concat([summar_eval_df[['doc_id','cluster_id']], svd_df], axis=1)
X=svd_df.groupby('cluster_id')[[i for i in svd_df.columns if 'text__' in i]].mean()
svd_summar_eval_probas=text_rf.predict_proba(X)[:,1]
svd_output_df = pandas.DataFrame({'score':svd_summar_eval_probas, 'cluster_id':X.index})
with open('data/fall_2016/summer_cp1_scored_svd.jsonl','w') as f:
    for index, row in svd_output_df.iterrows():
        f.write(json.dumps({'score':row['score'], 'cluster_id':row['cluster_id']}) + '\n')
