import os
import pandas as pd
import sqlalchemy
import ujson as json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
import cPickle as pickle

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor

# Constant paths
MEMEX_AD_FEATURES_REPO_PATH = '/home/ubuntu/memex_ad_features'
TRUE_POSITIVES_JSON_PATH = os.path.join(MEMEX_AD_FEATURES_REPO_PATH,
                                        'true_positives_text.json')
TRUE_NEGATIVES_CSV_PATH = os.path.join(MEMEX_AD_FEATURES_REPO_PATH,
                                       'negative_sample.csv')

PRICE_IMPUTER_EXTRACTOR_PATH = os.path.join(MEMEX_AD_FEATURES_REPO_PATH,
                                            'price_imputation_text_extractor.pkl')

PRICE_IMPUTER_MODEL_PATH = os.path.join(MEMEX_AD_FEATURES_REPO_PATH,
                                        'price_imputation_model.pkl')

ESCORT_CDR_2_DATA_PATH = '/home/ubuntu/memexHack1/data/escort_cdr_2'
ESCORT_CDR_2_CONTENT_PATH = os.path.join(ESCORT_CDR_2_DATA_PATH,
                                         'content.tsv')
ESCORT_CDR_2_PHONES_PATH = os.path.join(ESCORT_CDR_2_DATA_PATH,
                                        'phones-combined.tsv')

# CDR IDs For Classifier
true_cdr_ids = set(json.loads(line)['doc_id'] for line
                   in open(TRUE_POSITIVES_JSON_PATH))
false_cdr_ids = set(pd.read_csv(TRUE_NEGATIVES_CSV_PATH).cdr_id.unique())
all_cdr_ids = true_cdr_ids.union(false_cdr_ids)

# Get CDR IDs from dd_dump_v2 sqlite database (Lattice data from April QPR)
df = pd.read_csv('dd_id_to_cdr_id.csv')
true_cdr_mapping = df[df['cdr_id'].isin(all_cdr_ids)]
true_cdr_mapping['true'] = true_cdr_mapping['cdr_id'].isin(true_cdr_ids)
true_cdr_mapping.to_csv('classifier_all_ids.csv', index=False)


# Get matching set of content
all_dd_ids = set(true_cdr_mapping['dd_id'].tolist())
print('Filtering dd content')
study_content = []
with open(ESCORT_CDR_2_CONTENT_PATH) as f:
    for line in f:
        dd_id_str, site, typ, url, text, content = line.split('\t')
        if int(dd_id_str) in all_dd_ids:
            study_content.append(line.strip())
with open('study_content.tsv', 'w') as f:
    for i in study_content:
        f.write(i + '\n')


# Merge content into unified "study_data" DF
content = pd.read_table('study_content.tsv',
                        header=None,
                        names=['dd_id', 'site', 'type',
                               'url', 'text', 'content'])
study_data = true_cdr_mapping.merge(content, how='left')
study_data.loc[study_data['content'].isnull(), 'content'] = ''


# Impute Prices
new_cv = pickle.load(open(PRICE_IMPUTER_EXTRACTOR_PATH, 'rb'))
rf_new = pickle.load(open(PRICE_IMPUTER_EXTRACTOR_PATH, 'rb'))

X = new_cv.transform(study_data['text'])
price = rf_new.predict(X)
study_data['price_imputed'] = price

# Parse content...?


def try_json(x):
    try:
        return(json.loads(x))
    except:
        return({})

study_extractions = pd.DataFrame(
    study_data['content'].apply(try_json).tolist())

# Clean up age feature
study_extractions['age'] = study_extractions[
    'age'].convert_objects(convert_numeric=True)


# Parse Ethnicity features
def split_or_na(x):
    try:
        return(x.split('|'))
    except:
        return([])

ethnicities = set({})
for i in study_extractions['ethnicity']:
    for j in split_or_na(i):
        ethnicities.add(j.strip())


def in_or_na(ethnicity, x):
    try:
        return (ethnicity in x)
    except:
        return(False)
for ethnicity in ethnicities:
    study_extractions['ethnicity__' + ethnicity] = study_extractions[
        'ethnicity'].apply(lambda x: in_or_na(ethnicity, x))



study_data = pd.concat([study_data, study_extractions], axis=1)
phones = pd.read_csv(ESCORT_CDR_2_PHONES_PATH,
                     sep='\t',
                     names=['dd_id', 'phone'])
out = study_data.merge(phones,
                       how='left')
out = out[out['phone'].notnull()]
out['phone_1'] = out['phone'].apply(lambda x: x.split("|")[0])
out = out[out['text'] != '\N']
out.to_csv('classifier_in.csv',
           sep='\t',
           encoding='utf-8',
           index=False)


all_text = out.groupby('phone_1')['text'].apply(lambda x: ' '.join(x.tolist()))
price_data = out.groupby('phone_1')['price_imputed'].describe().unstack()
del price_data['count']
price_data = price_data.rename(
    columns={i: 'price__' + i for i in price_data.columns})
age_data = out.groupby('phone_1')['age'].describe().unstack()
age_data = age_data.rename(columns={i: 'age__' + i for i in age_data.columns})
ethnicities = out.groupby(
    'phone_1')[[i for i in out.columns if 'ethnicity__' in i]].mean()
group = pd.DataFrame(out.groupby('phone_1')['true'].max())


m = pd.concat([price_data, age_data, ethnicities, group], axis=1)
m.to_csv('phone_level_classifier_in.csv', index=True)
