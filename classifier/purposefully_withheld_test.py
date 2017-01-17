import pandas
import datetime
import json
import ipdb
import cPickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import dateutil
#model_df = pandas.read_csv('model_fit.csv', index_col='phone_1')
result_dir='time/'
def try_parse(date_str):
    try:
        return(dateutil.parser.parse(date_str))
    except:
        return datetime.datetime(year=1800, month=1, day=1)
def force_float(age_str):
    try:
        return(float(age_str))
    except:
        return(np.nan)
model_df = pandas.read_csv('qpr_imputation.csv', dtype={'phone_1':'str'})
model_df['age'] = model_df['age'].apply(force_float)
model_df = model_df[model_df['age'].notnull()]

model_df['postdatetime'] = model_df['posttime'].apply(try_parse)
def draw_nonrandom_folds(df, variable='postdatetime', cutoff=datetime.datetime(year=2015, month=1, day=1)):
    fold_df=df.copy()
    fold_df['cutoff'] = fold_df[variable] >= cutoff
    out = []
    temp=pandas.DataFrame(fold_df.groupby('cluster_id')['cutoff'].std().fillna(0)).reset_index()
    temp=temp.rename(columns={'cutoff':'cutoff_cluster_std'})
    fold_df = fold_df.merge(temp)
    print('initial df shape: %s' % fold_df.shape[0])
    fold_df=fold_df[~(fold_df['cutoff_cluster_std'] > 0)]
    print('filtered df shape: %s' % fold_df.shape[0])
    train = fold_df.index[fold_df[variable] < cutoff]
    test = fold_df.index[fold_df[variable] >= cutoff]
    return( [(0, (train, test))])
cluster_id = 'cluster_id'
def draw_folds_by_variable(df, variable='case_id',n_folds=5, random_state=0):
    variable_values = df[variable].value_counts().index
    if random_state:
        np.random.seed(random_state)
    out = [] # initialize empty output list, which will be (index, (test, train)) for each of n_folds
    i = 0
    for train, test in KFold(len(variable_values), n_folds, random_state=random_state, shuffle=True):
        out.append((i,
                (
                    df.index[df[variable].isin(variable_values[train])],
                    df.index[df[variable].isin(variable_values[test])]
                )
            ))
        i+=1
    return(out)
def all_scoring_metrics_plain(clf, X, y, stratified_kfold):
    out = []
    for train, test in stratified_kfold:
        y_train=y.loc[train]
        fitted_clf=clf.fit(X.loc[train], y_train)
        y_pred = fitted_clf.predict(X.loc[test])
        probs = fitted_clf.predict_proba(X.loc[test])[:,1]
        y_test=y.loc[test]
        true_positives = (y_test & y_pred).sum()
        true_negatives = ((~y_test) & (~y_pred)).sum()
        false_positives = ((~y_test) & y_pred).sum()
        false_negatives = (y_test & (~y_pred)).sum()
        assert((true_positives + true_negatives + false_positives + false_negatives) == y_test.shape[0] ) 
        f1=(2*true_positives)/float(2*true_positives + false_negatives + false_positives)
        true_positive_rate= true_positives/float(true_positives + false_negatives)
        true_negative_rate = (true_negatives/float(true_negatives + false_positives))
        accuracy = (true_positives + true_negatives)/float(true_positives + true_negatives+false_positives + false_negatives)
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_test, probs)
        roc_df=pandas.DataFrame({'fpr':roc_fpr, 'tpr':roc_tpr, 'threshold':roc_thresholds})
        scores=pandas.DataFrame({'probs':probs,'y_test':y_test})
        scores.sort_values('probs', ascending=False, inplace=True)
        out_dict = {
            'true_positive_rate':true_positive_rate,
            'true_negative_rate':true_negative_rate,
            'f1':f1,
            'frac_predicted_true':y_pred.mean(),
            'frac_predicted_true_in_sample':fitted_clf.predict(X.loc[train]).mean(),
            'precision':precision_score(y_test, y_pred),
            'recall':recall_score(y_test, y_pred),
            'accuracy':accuracy,
            'roc_auc':roc_auc_score(y_test,fitted_clf.predict_proba(X.loc[test])[:,1]),
            'fpr_at_90_tpr':roc_fpr[(roc_tpr < .9).sum()],
            'fpr_at_50_tpr':roc_fpr[(roc_tpr < .5).sum()],
            'fpr_at_10_tpr':roc_fpr[(roc_tpr < .1).sum()],
            'precision_at_20':scores[scores['probs'] > 0].iloc[0:20]['y_test'].mean(),
                }
        for i in np.arange(0,1,0.1):
            d={
            'frac_predicted_true%0.1f' % i:(probs > i).mean(),
            'precision%0.1f' % i:precision_score(y_test, probs > i),
            'recall%0.1f' % i:recall_score(y_test, probs > i)
            }
            out_dict.update(d)
        out.append(out_dict)
        out.append(out_dict)
    return pandas.DataFrame(out)
def all_scoring_metrics(clf, X, stratified_kfold):
# Note here y is in the 'class' column of X and we depend on a # classifier pipeline that transforms the X matrix to different # columns which ignore this
    out = []
    for index, (train, test) in stratified_kfold:
        y_train=X.loc[train].groupby(cluster_id)['class'].max().astype('bool')
        ipdb.set_trace()
        fitted_clf=clf.fit(X.loc[train], y_train)
        y_pred = fitted_clf.predict(X.loc[test])
        probs = fitted_clf.predict_proba(X.loc[test])[:,1]
        y_test=X.loc[test].groupby(cluster_id)['class'].max().astype('bool')
        true_positives = (y_test & y_pred).sum()
        true_negatives = ((~y_test) & (~y_pred)).sum()
        false_positives = ((~y_test) & y_pred).sum()
        false_negatives = (y_test & (~y_pred)).sum()
        assert((true_positives + true_negatives + false_positives + false_negatives) == y_test.shape[0] ) 
        f1=(2*true_positives)/float(2*true_positives + false_negatives + false_positives)
        true_positive_rate= true_positives/float(true_positives + false_negatives)
        true_negative_rate = (true_negatives/float(true_negatives + false_positives))
        accuracy = (true_positives + true_negatives)/float(true_positives + true_negatives+false_positives + false_negatives)
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_test, probs)
        roc_df=pandas.DataFrame({'fpr':roc_fpr, 'tpr':roc_tpr, 'threshold':roc_thresholds})
        scores=pandas.DataFrame({'probs':probs,'y_test':y_test})
        scores.sort_values('probs', ascending=False, inplace=True)
        out_dict = {
            'true_positive_rate':true_positive_rate,
            'true_negative_rate':true_negative_rate,
            'f1':f1,
            'frac_predicted_true':y_pred.mean(),
            'frac_predicted_true_in_sample':fitted_clf.predict(X.loc[train]).mean(),
            'precision':precision_score(y_test, y_pred),
            'recall':recall_score(y_test, y_pred),
            'accuracy':accuracy,
            'roc_auc':roc_auc_score(y_test,fitted_clf.predict_proba(X.loc[test])[:,1]),
            'fpr_at_90_tpr':roc_fpr[(roc_tpr < .9).sum()],
            'fpr_at_50_tpr':roc_fpr[(roc_tpr < .5).sum()],
            'fpr_at_10_tpr':roc_fpr[(roc_tpr < .1).sum()],
            'precision_at_20':scores[scores['probs'] > 0].iloc[0:20]['y_test'].mean(),
                }
        for i in np.arange(0,1,0.1):
            d={
            'frac_predicted_true%0.1f' % i:(probs > i).mean(),
            'precision%0.1f' % i:precision_score(y_test, probs > i),
            'recall%0.1f' % i:recall_score(y_test, probs > i)
            }
            out_dict.update(d)
        out.append(out_dict)
    return pandas.DataFrame(out)
def all_scoring_metrics_date(clf, X, stratified_kfold):
# Note here y is in the 'class' column of X and we depend on a # classifier pipeline that transforms the X matrix to different # columns which ignore this
    out = []
    for index, (train, test) in stratified_kfold:
        y_train=X.loc[train].groupby(cluster_id)['class'].max().astype('bool')
        fitted_clf=clf.fit(X.loc[train], y_train)
        y_pred = fitted_clf.predict(X.loc[test])
        probs = fitted_clf.predict_proba(X.loc[test])[:,1]
        y_test=X.loc[test].groupby(cluster_id)['class'].max().astype('bool')
        true_positives = (y_test & y_pred).sum()
        true_negatives = ((~y_test) & (~y_pred)).sum()
        false_positives = ((~y_test) & y_pred).sum()
        false_negatives = (y_test & (~y_pred)).sum()
        assert((true_positives + true_negatives + false_positives + false_negatives) == y_test.shape[0] ) 
        f1=(2*true_positives)/float(2*true_positives + false_negatives + false_positives)
        true_positive_rate= true_positives/float(true_positives + false_negatives)
        true_negative_rate = (true_negatives/float(true_negatives + false_positives))
        accuracy = (true_positives + true_negatives)/float(true_positives + true_negatives+false_positives + false_negatives)
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_test, probs)
        roc_df=pandas.DataFrame({'fpr':roc_fpr, 'tpr':roc_tpr, 'threshold':roc_thresholds})
        scores=pandas.DataFrame({'probs':probs,'y_test':y_test})
        scores.sort_values('probs', ascending=False, inplace=True)
        out_dict = {
            'true_positive_rate':true_positive_rate,
            'true_negative_rate':true_negative_rate,
            'f1':f1,
            'frac_predicted_true':y_pred.mean(),
            'frac_predicted_true_in_sample':fitted_clf.predict(X.loc[train]).mean(),
            'precision':precision_score(y_test, y_pred),
            'recall':recall_score(y_test, y_pred),
            'accuracy':accuracy,
            'roc_auc':roc_auc_score(y_test,fitted_clf.predict_proba(X.loc[test])[:,1]),
            'fpr_at_90_tpr':roc_fpr[(roc_tpr < .9).sum()],
            'fpr_at_50_tpr':roc_fpr[(roc_tpr < .5).sum()],
            'fpr_at_10_tpr':roc_fpr[(roc_tpr < .1).sum()],
            'precision_at_20':scores[scores['probs'] > 0].iloc[0:20]['y_test'].mean(),
                }
        for i in np.arange(0,1,0.1):
            d={
            'frac_predicted_true%0.1f' % i:(probs > i).mean(),
            'precision%0.1f' % i:precision_score(y_test, probs > i),
            'recall%0.1f' % i:recall_score(y_test, probs > i)
            }
            out_dict.update(d)
        out.append(out_dict)
    return pandas.DataFrame(out)
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
    def __init__(self, grouping_column=None, return_column = None, return_column_filter=None, calculation_col_filter=None):
        self.grouping_column=grouping_column
        self.return_column = return_column
        self.return_column_filter = return_column_filter
        self.calculation_col_filter = calculation_col_filter

    def fit(self, x, y=None):
        #self._feature_names = x.columns
        return self
    #def get_feature_names(self):
        #return(self._feature_names)

    def transform(self, data):
        if self.calculation_col_filter:
            main_column = [i for i in data.columns if self.calculation_col_filter in i]
        else:
            main_column = [i for i in data.columns if i != self.grouping_column]
        summary_stats=data.groupby(self.grouping_column)[main_column].describe().unstack()
        if self.calculation_col_filter:
            summary_stats=summary_stats.xs('mean', level=1, axis=1)
        else:
            summary_stats = summary_stats.xs(main_column, level=0, axis=1)
            del summary_stats['count']
            summary_stats = summary_stats.rename(columns = {i:'%s__%s' % (main_column, i) for i in summary_stats.columns})
        summary_stats = summary_stats.fillna(-1)
        if self.return_column_filter:
            return summary_stats[[i for i in summary_stats.columns if self.return_column_filter in i]]
        if self.return_column:
            return summary_stats[['%s__%s' % (main_column, self.return_column)]]
        else:
            return summary_stats

extracted_age_rf_pipeline = Pipeline([

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            #('phone_getter', Uniquifier([cluster_id])),

            #('bad_identifier', GroupbyMax([cluster_id],['bad'])),

            # Pipeline for computing price stats

            # Pipeline for computing age stats
            ('age', Pipeline([
                ('age_getter', ItemSelector([cluster_id,'age'])),
                ('averager', Summarizer(grouping_column=cluster_id)),
            ])),


        ],
    )),

    ('rf', RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True))
    # Fit a logistic regression with crossvalidation
])
imputed_age_rf_pipeline = Pipeline([

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            #('phone_getter', Uniquifier([cluster_id])),

            #('bad_identifier', GroupbyMax([cluster_id],['bad'])),

            # Pipeline for computing price stats

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
text_svd_mean_pipeline = Pipeline([
    ('averager', Summarizer(grouping_column=cluster_id, return_column_filter ='text', return_column='mean', calculation_col_filter='text')),
    ('rf', RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True))
    ])

logistic_pipeline = Pipeline([

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

    ('logistic', LogisticRegression())
    # Fit a logistic regression with crossvalidation
])
logistic_mean_only_pipeline = Pipeline([

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            #('phone_getter', Uniquifier([cluster_id])),

            #('bad_identifier', GroupbyMax([cluster_id],['bad'])),

            # Pipeline for computing price stats
            ('price', Pipeline([
                ('price_getter', ItemSelector([cluster_id,'price_imputed'])),
                ('averager', Summarizer(grouping_column=cluster_id, return_column='mean')),
            ])),

            # Pipeline for computing age stats
            ('age', Pipeline([
                ('age_getter', ItemSelector([cluster_id,'age_imputed'])),
                ('averager', Summarizer(grouping_column=cluster_id, return_column='mean')),
            ])),


        ],
    )),

    ('logistic', LogisticRegression())
    # Fit a logistic regression with crossvalidation
])
model_df['bad'] = model_df['class']==1
print('fitting text svd mean only pipeline')
text_df = pandas.read_csv('svd_features.csv')
text_df=text_df.merge(model_df[['doc_id','cluster_id','class','postdatetime']])
text_svd_mean_pipeline.fit(text_df[0:1000],model_df[0:1000].groupby(cluster_id)['class'].max())
cluster_id = 'cluster_id'
#print('fitting rf pipeline')
#pipeline.fit(model_df, model_df.groupby(cluster_id)['class'].max())
#print('fitting logistic pipeline')
#logistic_pipeline.fit(model_df, model_df.groupby(cluster_id)['class'].max())
#out =pipeline.transform(model_df)
#X = out[:,2:]
#y = out[:,1]

text_cols = {x for x in model_df.columns if x.find('text_feature__') > -1}
price_imputed_cols = {x for x in model_df.columns if x.find('price_imputed__') > -1}
age_imputed_cols = {x for x in model_df.columns if x.find('age_imputed__') > -1}
rf = RandomForestClassifier(n_jobs=1, n_estimators=40, random_state=2, oob_score=True)
num_folds=5
seed=2

logistic_mean_only_pipeline.fit(model_df, model_df.groupby(cluster_id)['class'].max())
#print('fitting rf pipeline')
#pipeline.fit(model_df, model_df.groupby(cluster_id)['class'].max())
#print('fitting logistic pipeline')
#logistic_pipeline.fit(model_df, model_df.groupby(cluster_id)['class'].max())
#out =pipeline.transform(model_df)
#X = out[:,2:]
#y = out[:,1]

text_cols = {x for x in model_df.columns if x.find('text_feature__') > -1}
price_imputed_cols = {x for x in model_df.columns if x.find('price_imputed__') > -1}
age_imputed_cols = {x for x in model_df.columns if x.find('age_imputed__') > -1}
rf = RandomForestClassifier(n_jobs=1, n_estimators=40, random_state=2, oob_score=True)
num_folds=5
seed=2

train_df=model_df.copy()
train_df.index = range(len(train_df))
train_df=train_df.reindex()
true_train_index = train_df['class'] == 1
#test_index = train_df['group'] == 'test'
false_train_index = train_df['class'] == 0
train_index = (true_train_index) | (false_train_index)

#folds = draw_folds_by_variable(train_df, 'cluster_id', n_folds=num_folds, random_state=seed)
time_fold_cutoffs = [datetime.datetime(year=2016, month=1, day=1), datetime.datetime(year=2015, month=7, day=1)]
folds = [draw_nonrandom_folds(train_df, 'postdatetime', cutoff=i)[0] for i in time_fold_cutoffs]
#folds = StratifiedKFold(train_df['class'], n_folds=num_folds, random_state=seed)

###
# Insert code to segment on class
###
# 1) do a groupby.std of class to see if there's a span
# 2) turn this into a DF
# 3) merge the df back in with a patterned name
del train_df['group']
#metrics=all_scoring_metrics(pipeline, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], train_df.loc[train_index,['class',cluster_id]].groupby(cluster_id)['class'].max(), folds)

## Do cluster scoring at phone level
#cluster_id = 'phone_1'
#phone_metrics=all_scoring_metrics(pipeline, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], folds)
#print(phone_metrics.mean())

# do cluster scoring at cluster level
cluster_id = 'cluster_id'
y_train=train_df.groupby(cluster_id)['class'].max().astype('bool')
imputed_age_rf_pipeline.fit(train_df[[i for i  in train_df.columns if i not in text_cols]], y_train)
imputed_probas = imputed_age_rf_pipeline.predict_proba(train_df[[i for i  in train_df.columns if i not in text_cols]])[:,1]
extracted_age_rf_pipeline.fit(train_df[[i for i  in train_df.columns if i not in text_cols]], y_train)
extracted_probas = extracted_age_rf_pipeline.predict_proba(train_df[[i for i  in train_df.columns if i not in text_cols]])[:,1]
analyze_df = pandas.DataFrame(y_train).copy()
analyze_df['imputed_probas'] = imputed_probas
analyze_df['extracted_probas'] = extracted_probas
analyze_df['imputed_class'] = analyze_df['imputed_probas'] > .5
analyze_df['extracted_class'] = analyze_df['extracted_probas'] > .5
print((analyze_df['imputed_class'] == analyze_df['extracted_class']).mean())
print(pandas.crosstab(analyze_df['imputed_class'],analyze_df['extracted_class']))
print(np.corrcoef(analyze_df[[i for i in analyze_df.columns if 'probas' in i]].T))
analyze_df.to_csv('age_classification.csv')
ipdb.set_trace()


print('Metrics for extracted age pipeline')
extracted_age_rf_metrics=all_scoring_metrics(extracted_age_rf_pipeline, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], folds)
extracted_age_rf_metrics.to_csv(result_dir + 'extracted_age_rf_metrics.csv')
print('Metrics for extracted age pipeline')
imputed_age_rf_metrics=all_scoring_metrics(imputed_age_rf_pipeline,  train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], folds)
imputed_age_rf_metrics.to_csv(result_dir + 'imputed_age_rf_metrics.csv')
print('Metrics for text svd  mean only pipeline')
text_svd_time_metrics=all_scoring_metrics(text_svd_mean_pipeline, text_df, folds)
text_svd_time_metrics.to_csv(result_dir + 'text_svd_time_metrics.csv')
print('Metrics for logistic mean only pipeline')
logistic_mean_only_metrics=all_scoring_metrics(logistic_mean_only_pipeline, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], folds)
logistic_mean_only_metrics.to_csv(result_dir + 'logistic_mean_only_metrics.csv')
print('Metrics for RF pipeline')
id_metrics=all_scoring_metrics(pipeline, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], folds)
id_metrics.to_csv(result_dir + 'price_age_metrics.csv')
print(id_metrics.mean())
print('Metrics for logistic pipeline')
logistic_price_metrics=all_scoring_metrics(logistic_pipeline, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], folds)
logistic_price_metrics.to_csv(result_dir + 'logistic_price_metrics.csv')

out = pandas.concat([ logistic_mean_only_metrics[['roc_auc']].rename(columns={'roc_auc':'logistic price/age w/ mean only'}), logistic_price_metrics[['roc_auc']].rename(columns={'roc_auc':'logistic price/age all'}), id_metrics[['roc_auc']].rename(columns={'roc_auc':'price/age random forest'}), imputed_age_rf_metrics[['roc_auc']].rename(columns={'roc_auc':'imputed age random forest'}), extracted_age_rf_metrics[['roc_auc']].rename(columns={'roc_auc':'extracted age random forest'}), text_svd_time_metrics[['roc_auc']].rename(columns={'roc_auc':'text svd'}) ], axis=1)
out.index = time_fold_cutoffs
out.to_csv('roc_all.csv')
out = pandas.concat([
    logistic_mean_only_metrics[['fpr_at_50_tpr']].rename(columns={'fpr_at_50_tpr':'logistic price/age w/ mean only'}),
    logistic_price_metrics[['fpr_at_50_tpr']].rename(columns={'fpr_at_50_tpr':'logistic price/age all'}),
    id_metrics[['fpr_at_50_tpr']].rename(columns={'fpr_at_50_tpr':'price/age random forest'}),
    imputed_age_rf_metrics[['fpr_at_50_tpr']].rename(columns={'fpr_at_50_tpr':'imputed age random forest'}),
    extracted_age_rf_metrics[['fpr_at_50_tpr']].rename(columns={'fpr_at_50_tpr':'extracted age random forest'}),
    text_svd_time_metrics[['fpr_at_50_tpr']].rename(columns={'fpr_at_50_tpr':'text svd'})
    ], axis=1)
out.index = time_fold_cutoffs
out.to_csv('fpr_at_50_tpr_all.csv')

ipdb.set_trace()
index_to_cdr_id_dict = {index:row['doc_id'] for index, row in train_df.iterrows()}
def folds_to_csv(folds):
    out = []
    for fold in folds:
        out.append({'fold':fold[0], 'train':[index_to_cdr_id_dict[i] for i in fold[1][0]], 'test':[index_to_cdr_id_dict[i] for i in fold[1][1]]})
    return(out)
fold_dict = folds_to_csv(folds)
open('time_folds.json','w').write(json.dumps(fold_dict))

text_df = pandas.read_csv('svd_features.csv')
text_df=text_df.merge(train_df[['doc_id','cluster_id','class','postdatetime']])
X=text_df.groupby('cluster_id')[[i for i in text_df.columns if 'text__' in i]].mean()
X.index=range(len(X))
X=X.reindex()
y=text_df.reset_index().groupby('cluster_id')['class'].max().astype('bool')
y.index=range(len(y))
rf = RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True)
cluster_folds = KFold(y.shape[0], n_folds=num_folds, random_state=seed)
time_fold_cutoffs = [datetime.datetime(year=2016, month=1, day=1), datetime.datetime(year=2015, month=7, day=1)]
cluster_folds = [draw_nonrandom_folds(X, 'postdatetime', cutoff=i)[0] for i in time_fold_cutoffs]
text_metrics=all_scoring_metrics_plain(rf, X, y, cluster_folds)
text_metrics.to_csv(result_dir + 'text_metrics.csv')

lm = LogisticRegression()
logistic_metrics=all_scoring_metrics_plain(lm, X, y, cluster_folds)
logistic_metrics.to_csv(result_dir + 'logistic_metrics.csv')

