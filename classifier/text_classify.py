import pandas
import cPickle
import pickle
import json
from helpers import fit_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
df = pandas.read_csv('evaluation_text_features.csv', index_col=['cluster_id'])

num_folds=3
text_cols = {x for x in df.columns if x.find('text_feature__') > -1}
age_cols = set({})
price_cols = set({})
price_imputed_cols = {x for x in df.columns if x.find('price_imputed__') > -1}
age_imputed_cols = {x for x in df.columns if x.find('age_imputed__') > -1}
cols = set(df.columns) - set(['group'])
rf = RandomForestClassifier(oob_score=True,
                            random_state=2,
                            n_estimators=100,
                            n_jobs=-1,
                            class_weight="balanced")

df_eval = df[df['group']=='test']
df = df[df['group'] != 'test']
y_series = df['group'] == 'train_true'
ages  = ['no_age','imputed_age']
prices = ['no_price','imputed_price']
texts = ['no_text', 'text']
model_result_list = []
for a in ages:
    cols = set(df.columns) - set(['group'])
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
            model_cols = cols - age_subtraction - price_subtraction-text_subtraction
            if not model_cols:
                continue
            name = '%s__%s__%s' % (a, p, text)
            print(len(cols))
            model_result_list.append({name:fit_model(df, y_series, model_cols, rf, num_folds=num_folds)})

all_cols = sorted(set(df.columns) - set(['group']) - set(price_imputed_cols) - set(age_imputed_cols))
rf_all = rf.fit(df[all_cols].fillna(0), y_series)

cPickle.dump(rf_all, open('text_only_model.pkl','wb'))
out = pandas.DataFrame([i.values()[0].to_dict()['rf'] for i in model_result_list], index = [i.keys()[0] for i in model_result_list])
out.to_csv('classifier_results_text.csv')
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
m.to_csv('adjusted_true_positive_rate_text.csv', index=False)
m_choice = m.loc[(m['population_true_positive_rate'] ==.01) & (m['probability_cutoff'] == .50),['index','adjusted_true_positive_rate']]
out = out.merge(m_choice, left_index=True, right_on='index')
out=out.set_index('index')

row_order= [
        'no_age__no_price__text',
        'imputed_age__imputed_price__no_text',
        'imputed_age__imputed_price__text',
        ]
column_order=[
        'roc_auc',
        'fpr_at_50tpr',
        'adjusted_true_positive_rate',
        ]
rows={
        'no_age__no_price__text':'Text Only',
        'imputed_age__imputed_price__no_text':'Imputed features, no text',
        'imputed_age__imputed_price__text':'Imputed Features and Text',
        }
columns={
        'roc_auc':'ROC AUC',
        'fpr_at_50tpr':'FP at 50% TP',
        'adjusted_true_positive_rate':'Pop. Precision: 1%'
        }

out_df = out.loc[row_order,column_order]
out_df = out_df.rename(columns=columns, index=rows)
out_df.to_latex('classifier_results_text_only.tex', formatters=[lambda x: '%0.3f' % x, lambda x: '%0.3f' % x, lambda x: '%0.3f' % x])

# Begin scoring
eval_probs=rf_all.predict_proba(df_eval[all_cols].fillna(0))[:,1]
s=pandas.DataFrame({'score':eval_probs, 'cluster_id':df_eval.index})
with open('evaluation_scores_text_only.jl','w') as f:
    for line in json.loads(s[['cluster_id','score']].T.to_json()).values():
        f.write(ujson.dumps(line) + '\n')

