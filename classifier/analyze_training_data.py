import ipdb
import json
import numpy as np
import datetime
import ujson
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dateutil.parser import parse

def try_parse(date_str):
    try:
        return(parse(date_str))
    except:
        return datetime.datetime(year=1800, month=1, day=1)
def normalize_city(city_str):
    try:
        city_str = city_str.replace(' Escorts\n</a>\n</div>\n</div>','').lower()
        city_str = city_str.replace(', ',',')
        return city_str
    except:
        return('')
cluster_id = 'cluster_id'
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
        #if data.has_key('price'):
            #ipdb.set_trace()
        if data.has_key('extractions'):
            #if len(data['extractions'].keys()) > 12:
                #ipdb.set_trace()
            if data['extractions'].has_key('phonenumber'):
                #out['rate_info'] = data['extractions']['rate']
                out[cluster_id] = data['extractions']['phonenumber']['results'][0]
            if data['extractions'].has_key('city'):
                out['city'] = data['extractions']['city']['results'][0]
            if data['extractions'].has_key('age'):
                out['age'] = data['extractions']['age']['results'][0]
            #if data['extractions'].has_key('price'):
                #ipdb.set_trace()
            if data['extractions'].has_key('posttime'):
                out['posttime'] = data['extractions']['posttime']['results'][0]
                
        out_list.append(out)
        print(count)
        count += 1

import pandas
true_df = pandas.DataFrame(out_list)
true_df['city'].value_counts().to_csv('raw_city_counts.csv', encoding='utf-8')
true_df['city']=true_df['city'].apply(normalize_city)
true_df['city'].value_counts().to_csv('cleaned_city_counts.csv', encoding='utf-8')
true_df['postdatetime'] = true_df['posttime'].apply(try_parse)
true_df['month'] = true_df['postdatetime'].apply(lambda x: x.month)
true_df['year'] = true_df['postdatetime'].apply(lambda x: x.year)
true_df.groupby(['year','month']).size().to_csv('post_month_counts.csv')
true_df.groupby(['year','month'])['class'].aggregate([np.mean, lambda x: np.std(x)/np.sqrt(len(x)), np.size]).rename(columns={'<lambda>':'stderr'}).to_csv('positives_rate.csv') 
true_df.groupby(['city'])['class'].aggregate([np.mean, lambda x: np.std(x)/np.sqrt(len(x)), np.size]).rename(columns={'<lambda>':'stderr'}).to_csv('positives_rate_cities.csv', encoding='utf-8') 
