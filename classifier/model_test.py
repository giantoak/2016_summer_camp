import pandas
import ipdb
import cPickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#model_df = pandas.read_csv('model_fit.csv', index_col='phone_1')
model_df = pandas.read_csv('imputations.csv', dtype={'phone_1':'str'})

def draw_folds_by_variable(df, variable='case_id',num_folds=5, random_state=0):
    variable_values = df[variable].value_counts().index
    if random_state:
        np.random.seed(random_state)
    out = [] # initialize empty output list, which will be (index, (test, train)) for each of num_folds
    i = 0
    for train, test in KFold(len(variable_values), num_folds, random_state=random_state, shuffle=True):
        out.append((i,
                (
                    df.index[df[variable].isin(variable_values[train])],
                    df.index[df[variable].isin(variable_values[test])]
                )
            ))
        i+=1
    return(out)
def all_scoring_metrics(clf, X, y, stratified_kfold):
    out = []
    for train, test in stratified_kfold:
        fitted_clf=clf.fit(X.loc[train], y.loc[train])
        y_pred = fitted_clf.predict(X.loc[test])
        probs = fitted_clf.predict_proba(X.loc[test])[:,1]
        y_test=y.loc[test]
        true_positives = (y_test & y_pred).sum()
        true_negatives = ((~y_test) & (~y_pred)).sum()
        false_positives = ((~y_test) & y_pred).sum()
        false_negatives = (y_test & (~y_pred)).sum()
        f1=(2*true_positives)/float(2*true_positives + false_negatives + false_positives)
        true_positive_rate= true_positives/float(true_positives + false_negatives)
        true_negative_rate = (true_negatives/float(true_negatives + false_positives))
        accuracy = (true_positives + true_negatives)/float(true_positives + true_negatives+false_positives + false_negatives)
        out.append({
            'true_positive_rate':true_positive_rate,
            'true_negative_rate':true_negative_rate,
            'f1':f1, 
            'frac_predicted_true':y_pred.mean(),
            'frac_predicted_true_in_sample':fitted_clf.predict(X.loc[train]).mean(),
            'precision':precision_score(y_test, y_pred),
            'recall':recall_score(y_test, y_pred),
            'frac_predicted_true.1':(probs > .1).mean(),
            'precision.1':precision_score(y_test, probs > .1),
            'recall.1':recall_score(y_test, probs > .1),
            'frac_predicted_true.2':(probs > .2).mean(),
            'precision.2':precision_score(y_test, probs > .2),
            'recall.2':recall_score(y_test, probs > .2),
            'frac_predicted_true.3':(probs > .3).mean(),
            'precision.3':precision_score(y_test, probs > .3),
            'recall.3':recall_score(y_test, probs > .3),
            'frac_predicted_true.4':(probs > .4).mean(),
            'precision.4':precision_score(y_test, probs > .4),
            'recall.4':recall_score(y_test, probs > .4),
            'accuracy':accuracy,
            'roc_auc':roc_auc_score(y.loc[test],fitted_clf.predict_proba(X.loc[test])[:,1])
            })
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

            ('phone_getter', Uniquifier(['phone_1'])),

            ('bad_identifier', GroupbyMax(['phone_1'],['bad'])),

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

    #('rf', RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True))
    # Fit a logistic regression with crossvalidation
])
model_df['bad'] = model_df['group']=='train_true'
out =pipeline.transform(model_df[model_df['group'] != 'test'])
X = out[:,2:]
y = out[:,1]

text_cols = {x for x in model_df.columns if x.find('text_feature__') > -1}
price_imputed_cols = {x for x in model_df.columns if x.find('price_imputed__') > -1}
age_imputed_cols = {x for x in model_df.columns if x.find('age_imputed__') > -1}
rf = RandomForestClassifier(n_jobs=-1, n_estimators=40, random_state=2, oob_score=True)
num_folds=5
seed=2

train_df=model_df[model_df['group'] != 'test']
train_df.index = range(len(train_df))
train_df=train_df.reindex()
true_train_index = train_df['group'] == 'train_true'
test_index = train_df['group'] == 'test'
false_train_index = train_df['group'] == 'train_false'
train_index = (true_train_index) | (false_train_index)
folds = StratifiedKFold(train_df['group'], n_folds=num_folds, random_state=seed)

del train_df['group']
metrics=all_scoring_metrics(rf, train_df.loc[train_index,[i for i  in train_df.columns if i not in text_cols]], true_train_index[train_index], folds)
print(metrics.mean())

#rf.fit(model_df.loc[train_index,[i for i  in model_df.columns if i not in text_cols]], true_train_index[train_index])
#rf.score(model_df.loc[train_index,[i for i  in model_df.columns if i not in text_cols]], true_train_index[train_index])
#predicted_test=rf.predict_proba(model_df.loc[test_index,[i for i  in model_df.columns if i not in text_cols]])[:,1]
