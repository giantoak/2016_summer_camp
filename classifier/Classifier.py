
# coding: utf-8

# # Classifier
# 
# No age or image data, since those seem to be the explanatory variables the classifiers love the most.
# We also leave out counts of incalls, outcalls, or "incalls and outcalls".
# 
# ## Imports

# In[1]:

from itertools import chain
import ipdb
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
from helpers import fit_model


df = pandas.read_csv('phone_level_classifier_in.csv', index_col=['phone_1'])
#df = df[[i for i in df.columns if 'age__' not in i]]
y_series = df['true']
phone_text_averages = pandas.read_csv('phone_text_averages.csv', index_col=['phone_1'])
df = pandas.concat([df, phone_text_averages], axis=1)
#X_text = pickle.load(open('svd_text.pkl','rb'))
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
output_df_list = [] # Build a list to turn into a dataframe


# In[3]:

id_cols = {'class', 'phone'}
text_cols = {x for x in df.columns if x.find('text_feature__') > -1}
price_cols = {x for x in df.columns if x.find('price__') > -1}
price_imputed_cols = {x for x in df.columns if x.find('price_imputed__') > -1}
age_cols = {x for x in df.columns if x.find('age__') > -1}
age_imputed_cols = {x for x in df.columns if x.find('age_imputed__') > -1}
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

ages  = ['no_age','measured_age','imputed_age']
prices = ['no_price','measured_price','imputed_price']
texts = ['no_text', 'text']
model_result_list = []
ipdb.set_trace()
for a in ages:
    cols = set(df.columns) - set(['true'])
    if a== 'no_age':
        age_subtraction = set(age_cols).union(set(age_imputed_cols))
    elif a == 'measured_age':
        age_subtraction = set(age_imputed_cols)
    elif a == 'imputed_age':
        age_subtraction = set(age_cols)
    else:
        raise(Exception("Weird age issue"))
    for p in prices:
        if p== 'no_price':
            price_subtraction = set(price_cols).union(set(price_imputed_cols))
        elif p == 'measured_price':
            price_subtraction = set(price_imputed_cols)
        elif p == 'imputed_price':
            price_subtraction = set(price_cols)
        else:
            raise(Exception("Weird price issue"))
        for text in texts:
            if text == "no_text":
                text_subtraction = set(text_cols)
            elif text == 'text':
                text_subtraction = set()
            else:
                raise(Exception("Weird text issue"))
            name = '%s__%s__%s' % (a, p, text)
            model_cols = cols - age_subtraction - price_subtraction-text_subtraction
            print(len(cols))
            model_result_list.append({name:fit_model(df, y_series, model_cols, rf, num_folds=num_folds)})



# Write final output model with everything
rf_fit = rf.fit(df.fillna(0), y_series)
pickle.dump(rf_fit, open('rf_all_features.pkl','wb'))  
df.to_csv('X_model_data.csv')

out = pandas.DataFrame([i.values()[0].to_dict()['rf'] for i in model_result_list], index = [i.keys()[0] for i in model_result_list])
out.to_csv('classifier_results.csv')
def adjusted_precision(tpr, fpr, true_base_rate):
    trues = tpr *true_base_rate
    falses = fpr * (1-true_base_rate)
    return(trues/(trues + falses))
adjusted_list = []
for i in np.arange(0.05, .95, .05):
    for index, row in out.iterrows():
        for threshold in [.0001, .001, .01, .1]:
            out_item = {}
            out_item['index']=index
            out_item['population_true_positive_rate'] = threshold
            out_item['probability_cutoff'] = i
            out_item['adjusted_true_positive_rate'] = adjusted_precision(
                    tpr=row['true_positive_rate_' + ('%0.2f' % i).replace('0.','')],
                    fpr=row['false_positive_rate_' + ('%0.2f' % i).replace('0.','')],
                    true_base_rate=threshold
                    )
            adjusted_list.append(out_item)
m=pandas.DataFrame(adjusted_list)
m.to_csv('adjusted_true_positive_rate.csv', index=False)
m_choice = m.loc[(m['population_true_positive_rate'] ==.01) & (m['probability_cutoff'] == .50),['index','adjusted_true_positive_rate']]
out = out.merge(m_choice, left_index=True, right_on='index')
out=out.set_index('index')
#row_order= [
        #'no_age__no_price__no_text',
        #'no_age__measured_price__no_text',
        #'measured_age__no_price__no_text',
        #'no_age__imputed_price__no_text',
        #'imputed_age__no_price__no_text',
        #'no_age__no_price__text',
        #'imputed_age__imputed_price__no_text',
        #]
#column_order=[
        #'roc_auc',
        #'fpr_at_50tpr',
        #'adjusted_true_positive_rate',
        #]
#rows={
        #'no_age__no_price__no_text':'No Age, No Price, No Text',
        #'no_age__measured_price__no_text':'No Age, Price, No Text',
        #'no_age__imputed_price__no_text':'No Age, Imputed Price, No Text',
        #'measured_age__no_price__no_text':'Age, No Price, No Text',
        #'imputed_age__no_price__no_text':'Imputed Age, No Price, No Text',
        #'no_age__no_price__text':'No Age, No Price, Text',
        #'imputed_age__imputed_price__no_text':'Age, Price, No Text',
        #}
#columns={
        #'roc_auc':'ROC AUC',
        #'fpr_at_50tpr':'FP at 50% TP',
        #'adjusted_true_positive_rate':'Pop. Precision: 1%'
        #}
row_order= [
        'measured_age__no_price__no_text',
        'measured_age__measured_price__no_text',
        'measured_age__imputed_price__no_text',
        'measured_age__imputed_price__text',
        'measured_age__no_price__text',
        ]
column_order=[
        'roc_auc',
        'fpr_at_50tpr',
        'adjusted_true_positive_rate',
        ]
rows={
        'measured_age__no_price__no_text':'Age/Ethnicity/Counts Only',
        'measured_age__measured_price__no_text':'With Extracted Price',
        'measured_age__imputed_price__no_text':'With Imputed Price',
        'measured_age__imputed_price__text':'With Imputed Price and Text',
        'measured_age__no_price__text':'Text, no Price',
        }
columns={
        'roc_auc':'ROC AUC',
        'fpr_at_50tpr':'FP at 50% TP',
        'adjusted_true_positive_rate':'Pop. Precision: 1%'
        }

out_df = out.loc[row_order,column_order]
out_df = out_df.rename(columns=columns, index=rows)
out_df.to_latex('classifier_results.tex', formatters=[lambda x: '%0.3f' % x, lambda x: '%0.3f' % x, lambda x: '%0.3f' % x])

ipdb.set_trace()

out_df = pandas.DataFrame([i.values()[0] for i in output_df_list], index=[i.keys()[0] for i in output_df_list])
out_df = out_df.loc[['no_price_no_text','price_no_text','text_and_price','text_no_price']]
out_df = out_df[['roc_auc','fpr_at_50tpr','fpr_at_50tpr_std','fpr_at_10tpr','fpr_at_10tpr_std']]
out_df = out_df.rename(index={'no_price_no_text':'Age/Ethnicity/Counts Only', 'price_no_text':'With Imputed Price','text_and_price':'With Price and Text','text_no_price':'Text, No Price'})
out_df = out_df.rename(columns ={'roc_auc':'ROC AUC','fpr_at_50tpr':'FP at 50% TP'})
out_df.to_latex('classifier_results.tex', formatters=[lambda x: '%0.3f' % x, lambda x: '%0.3f' % x, lambda x: '%0.3f' % x])
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
output_df_list.append({'price_no_text':{
    'roc_auc':metrics.loc['roc_auc','rf'],
    'fpr_at_50tpr':metrics.loc['fpr_at_50tpr','rf'],
    'fpr_at_50tpr_std':rf_metrics['fpr_at_50tpr'].std(),
    'fpr_at_10tpr':metrics.loc['fpr_at_10tpr','rf'],
    'fpr_at_10tpr_std':rf_metrics['fpr_at_10tpr'].std(),
    }})
print(metrics.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr','fpr_at_10tpr']])

X_df = df.loc[:, sorted(set(df.columns) - set(['true']) - price_cols)]
X_df=X_df.fillna(0)
print(X_df.shape)
etc_metrics = all_scoring_metrics(etc, X_df, y_series, StratifiedKFold(y_series, num_folds))
rf_metrics = all_scoring_metrics(rf, X_df, y_series, StratifiedKFold(y_series, num_folds))

metrics_no_price = pandas.DataFrame({'etc':etc_metrics.mean().values, 'rf':rf_metrics.mean().values}, index=rf_metrics.mean().keys())
metrics_no_price.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr','fpr_at_10tpr']]
output_df_list.append({'no_price_no_text':{
    'roc_auc':metrics_no_price.loc['roc_auc','rf'],
    'fpr_at_50tpr':metrics_no_price.loc['fpr_at_50tpr','rf'],
    'fpr_at_50tpr_std':rf_metrics['fpr_at_50tpr'].std(),
    'fpr_at_10tpr':metrics.loc['fpr_at_10tpr','rf'],
    'fpr_at_10tpr_std':rf_metrics['fpr_at_10tpr'].std(),
    }})
print(metrics_no_price.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr','fpr_at_10tpr']])

print('Adding Text')
X_df = df.loc[:, sorted(set(df.columns) - set(['true']))]
X_df = pandas.concat([X_df, phone_text_averages], axis=1)
X_df=X_df.fillna(0)
etc_metrics = all_scoring_metrics(etc, X_df, y_series, StratifiedKFold(y_series, num_folds))
rf_metrics = all_scoring_metrics(rf, X_df, y_series, StratifiedKFold(y_series, num_folds))

metrics_with_text = pandas.DataFrame({'etc':etc_metrics.mean().values, 'rf':rf_metrics.mean().values}, index=rf_metrics.mean().keys())
metrics_with_text.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr','fpr_at_10tpr']]
output_df_list.append({'text_and_price':{
    'roc_auc':metrics_with_text.loc['roc_auc','rf'],
    'fpr_at_50tpr':metrics_with_text.loc['fpr_at_50tpr','rf'],
    'fpr_at_50tpr_std':rf_metrics['fpr_at_50tpr'].std(),
    'fpr_at_10tpr':metrics.loc['fpr_at_10tpr','rf'],
    'fpr_at_10tpr_std':rf_metrics['fpr_at_10tpr'].std(),
    }})
print(metrics_with_text.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr','fpr_at_10tpr']])

print('Text no price')
X_df = df.loc[:, sorted(set(df.columns) - set(['true'])- set(price_cols))]
X_df = pandas.concat([X_df, phone_text_averages], axis=1)
X_df=X_df.fillna(0)
etc_metrics = all_scoring_metrics(etc, X_df, y_series, StratifiedKFold(y_series, num_folds))
rf_metrics = all_scoring_metrics(rf, X_df, y_series, StratifiedKFold(y_series, num_folds))

metrics_with_text_no_price = pandas.DataFrame({'etc':etc_metrics.mean().values, 'rf':rf_metrics.mean().values}, index=rf_metrics.mean().keys())
metrics_with_text_no_price.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr']]
output_df_list.append({'text_no_price':{
    'roc_auc':metrics_with_text_no_price.loc['roc_auc','rf'],
    'fpr_at_50tpr':metrics_with_text_no_price.loc['fpr_at_50tpr','rf'],
    'fpr_at_50tpr_std':rf_metrics['fpr_at_50tpr'].std(),
    'fpr_at_10tpr':metrics.loc['fpr_at_10tpr','rf'],
    'fpr_at_10tpr_std':rf_metrics['fpr_at_10tpr'].std(),
    }})
print(metrics_with_text_no_price.loc[['f1','roc_auc','fpr_at_50tpr','fpr_at_lowest_tpr','tpr_at_lowest_tpr','fpr_at_10tpr']])

out_df = pandas.DataFrame([i.values()[0] for i in output_df_list], index=[i.keys()[0] for i in output_df_list])
out_df = out_df.loc[['no_price_no_text','price_no_text','text_and_price','text_no_price']]
out_df = out_df[['roc_auc','fpr_at_50tpr','fpr_at_50tpr_std','fpr_at_10tpr','fpr_at_10tpr_std']]
out_df = out_df.rename(index={'no_price_no_text':'Age/Ethnicity/Counts Only', 'price_no_text':'With Imputed Price','text_and_price':'With Price and Text','text_no_price':'Text, No Price'})
out_df = out_df.rename(columns ={'roc_auc':'ROC AUC','fpr_at_50tpr':'FP at 50% TP'})
out_df.to_latex('classifier_results.tex', formatters=[lambda x: '%0.3f' % x, lambda x: '%0.3f' % x, lambda x: '%0.3f' % x])

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

