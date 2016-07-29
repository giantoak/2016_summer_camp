
# coding: utf-8

# # Classifier
# 
# No age or image data, since those seem to be the explanatory variables the classifiers love the most.
# We also leave out counts of incalls, outcalls, or "incalls and outcalls".
# 
# ## Imports

# In[1]:

from itertools import chain
import html
import ujson as json
import multiprocessing as mp
import pickle
import numpy as np
import pandas 
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold

from helpers import all_scoring_metrics
from helpers import draw_rocs
from helpers import draw_folds_by_variable


df = pandas.read_csv('phone_level_classifier_in.csv', index_col=['phone_1'])
num_folds = 10
#eval_columns = ['f1',
                #'accuracy',
                #'true_negative_rate',
                #'true_positive_rate',
                #'roc_auc',
                #'roc_fpr',
                #'roc_tpr',
                #'roc_thresholds']
#price_cols = ['duration_in_mins',
              #'price',
              #'price_per_min']


# In[2]:

#df = pd.read_pickle('data/generated/phone_level_merged_df.pkl')
#print(df.shape)
#print(df['class'].value_counts())


# In[3]:

id_cols = {'class', 'phone'}
price_cols = {x for x in df.columns if x.find('price') > -1}
duration_cols = {x for x in df.columns if x.find('duration') > -1}
flag_cols = {x for x in df.columns if x.find('flag_') > -1}
ethnicity_cols = {x for x in df.columns if x.find('ethnicity_') > -1}
image_cols = {x for x in df.columns if x.find('image') > -1}
age_cols = {x for x in df.columns if x.find('age') > -1} | set(['flag_Juvenile'])
#service_cols = set(['n_incall', 'n_outcall', 'n_incall_and_outcall'])


# In[4]:

etc = ExtraTreesClassifier(oob_score=True,
                           bootstrap=True,
                           random_state=2,
                           n_estimators=100,
                           n_jobs=-1,
                           class_weight="balanced")

rf = RandomForestClassifier(oob_score=True,
                            random_state=2,
                            n_estimators=100,
                            n_jobs=-1,
                            class_weight="balanced")


# ## Execution
# Use both Random Forests and Extra Trees to classify the data using all columns

# In[5]:

y_series = df['true']
X_df = df.loc[:, sorted(set(df.columns) - set(['true']))]
X_df=X_df.fillna(-1)
print(y_series.shape)
print(X_df.shape)


# ### Extra Trees

# In[6]:

X_df = df.loc[:, sorted(set(df.columns) - set(['true']))]
X_df=X_df.fillna(0)
print(X_df.shape)
etc_metrics = all_scoring_metrics(etc, X_df, y_series, StratifiedKFold(y_series, num_folds))
rf_metrics = all_scoring_metrics(rf, X_df, y_series, StratifiedKFold(y_series, num_folds))

metrics = pandas.DataFrame({'etc':etc_metrics.mean().values, 'rf':rf_metrics.mean().values}, index=rf_metrics.mean().keys())
metrics.loc[['f1','roc_auc','true_negative_rate','true_positive_rate']]
print(metrics.loc[['f1','roc_auc','true_negative_rate','true_positive_rate']])

X_df = df.loc[:, sorted(set(df.columns) - set(['true']) - price_cols)]
X_df=X_df.fillna(0)
print(X_df.shape)
etc_metrics = all_scoring_metrics(etc, X_df, y_series, StratifiedKFold(y_series, num_folds))
rf_metrics = all_scoring_metrics(rf, X_df, y_series, StratifiedKFold(y_series, num_folds))

metrics_no_price = pandas.DataFrame({'etc':etc_metrics.mean().values, 'rf':rf_metrics.mean().values}, index=rf_metrics.mean().keys())
metrics_no_price.loc[['f1','roc_auc','true_negative_rate','true_positive_rate']]
print(metrics_no_price.loc[['f1','roc_auc','true_negative_rate','true_positive_rate']])

# In[7]:

#this_metrics = etc_metrics
#print(this_metrics.roc_auc.mean())
#this_metrics.loc[:, ['f1', 'accuracy', 'true_negative_rate', 'true_positive_rate', 'roc_auc']]


## In[8]:

#ranked_imptncs = this_metrics.loc[:,
                                  #sorted(set(this_metrics.columns) - 
                                         #set(eval_columns))].mean().sort_values(ascending=False)
#print(ranked_imptncs.head(20))
#sns.violinplot(ranked_imptncs, inner='point')


## In[9]:

#draw_rocs(this_metrics, 'Extra Trees')


## ### Random Forest

## In[10]:

#rf_metrics = all_scoring_metrics(rf, X_df, y_series, StratifiedKFold(y_series, num_folds))


## In[11]:

#this_metrics = rf_metrics
#print(this_metrics.roc_auc.mean())
#this_metrics.loc[:, ['f1', 'accuracy', 'true_negative_rate', 'true_positive_rate', 'roc_auc']]


## In[12]:

#ranked_imptncs = this_metrics.loc[:,
                                  #sorted(set(this_metrics.columns) - 
                                         #set(eval_columns))].mean().sort_values(ascending=False)
#print(ranked_imptncs.head(20))
#sns.violinplot(ranked_imptncs, inner='point')


## In[13]:

#draw_rocs(this_metrics, 'Random Forest')

