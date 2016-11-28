import ipdb
import ujson
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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

            #('phone_getter', Uniquifier(['phone_1'])),

            #('bad_identifier', GroupbyMax(['phone_1'],['bad'])),

            # Pipeline for computing price stats
            ('price', Pipeline([
                ('price_getter', ItemSelector(['phone_1','price_imputed'])),
                ('averager', Summarizer(grouping_column='phone_1')),
            ])),

            # Pipeline for computing age stats
            ('age', Pipeline([
                ('age_getter', ItemSelector(['phone_1','age_imputed'])),
                ('averager', Summarizer(grouping_column='phone_1')),
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
        if data.has_key('extractions'):
            if data['extractions'].has_key('phonenumber'):
                #out['rate_info'] = data['extractions']['rate']
                out['phone_1'] = data['extractions']['phonenumber']['results'][0]
                
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
                #out['phone_1'] = data['extractions']['phonenumber']['results'][0]
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

del df['content']
# Do imputation group-bys
price_data_imputed = df.groupby('phone_1')['price_imputed'].describe().unstack()
del price_data_imputed['count']
price_data_imputed = price_data_imputed.rename(columns={i:'price_imputed__' + i for i in price_data_imputed.columns})
age_data_imputed = df.groupby('phone_1')['age_imputed'].describe().unstack()
del age_data_imputed['count']
age_data_imputed = age_data_imputed.rename(columns={i:'age_imputed__' + i for i in age_data_imputed.columns})
ipdb.set_trace()

model_df = pandas.concat([price_data_imputed, age_data_imputed], axis=1)
model_df=model_df.fillna(-1) # Fill missing at -1

train_indices = df['class'].apply(lambda x: True)
train_phones = df.loc[train_indices, 'phone_1'].tolist()
true_train_phones = df.loc[df['class']==1,'phone_1'].tolist()
#test_phones = df.loc[df['group']=='test','phone_1'].tolist()
true_train_index = pandas.Series(model_df.index.isin(true_train_phones), index=model_df.index) 
train_index = pandas.Series(model_df.index.isin(train_phones), index=model_df.index) 
#test_index = pandas.Series(model_df.index.isin(test_phones), index=model_df.index) 

test_indices = df['group'] == 'test'

aggregated_group = df.groupby('phone_1')['class'].apply(lambda x: x.max())
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
y=df.groupby('phone_1').apply(lambda x: (x['class']).max()).astype('int')    
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
