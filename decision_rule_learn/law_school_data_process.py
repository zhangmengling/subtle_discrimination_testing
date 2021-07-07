import os
import tempfile
import pandas as pd

import six.moves.urllib as urllib
import pprint

# import tensorflow_model_analysis as tfma
from google.protobuf import text_format

import tensorflow as tf
# tf.compat.v1.enable_v2_behavior()

#
# # Download the LSAT dataset and setup the required filepaths.
# _DATA_ROOT = tempfile.mkdtemp(prefix='lsat-data')
# _DATA_PATH = 'https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv'
# _DATA_FILEPATH = os.path.join(_DATA_ROOT, 'bar_pass_prediction.csv')
#
# data = urllib.request.urlopen(_DATA_PATH)
#
# _LSAT_DF = pd.read_csv(data)
#
# # To simpliy the case study, we will only use the columns that will be used for
# # our model.
# _COLUMN_NAMES = [
#   'dnn_bar_pass_prediction',
#   'gender',
#   'lsat',
#   'pass_bar',
#   'race1',
#   'ugpa',
# ]
#
# _LSAT_DF.dropna()
# _LSAT_DF['gender'] = _LSAT_DF['gender'].astype(str)
# _LSAT_DF['race1'] = _LSAT_DF['race1'].astype(str)
# _LSAT_DF = _LSAT_DF[_COLUMN_NAMES]
#
# _LSAT_DF.head()


import requests
import csv
from contextlib import closing
#
# # 文件地址
# url = "https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv"
#
# # 读取数据
# with closing(requests.get(url, stream=True)) as r:
#     f = (line.decode('gbk') for line in r.iter_lines())
#     reader = csv.reader(f, delimiter=',', quotechar='"')
#     law_school_list = []
#     for row in reader:
#         law_school_list.append(row)
#         # print(row)
#
# # law_school_list.to_csv("../dataset/law_school.csv")
#
# test=pd.DataFrame(data=law_school_list)
# print(test)
# test.info()
# test.to_csv('../dataset/law_school.csv',encoding='gbk')

data = "../dataset/law_school.csv"
df = pd.read_csv(data, keep_default_na=False)

df.info()


#  bar1
for index, row in df.iterrows():
    if row['bar1'] == 'P':
        df.loc[index, 'bar1'] = 1
    else:
        df.loc[index, 'bar1'] = 0

# bar2
for index, row in df.iterrows():
    if row['bar2'] == 'P':
        df.loc[index, 'bar2'] = 1
    else:
        df.loc[index, 'bar2'] = 0

for index, row in df.iterrows():
    if row['race'] == 1 or row['race'] == 8:
        df.loc[index, 'other'] = 1
        df.loc[index, 'asian'] = 0
        df.loc[index, 'black'] = 0
        df.loc[index, 'hisp'] = 0
        df.loc[index, 'white'] = 0
    if row['race'] == 2:
        df.loc[index, 'other'] = 0
        df.loc[index, 'asian'] = 1
        df.loc[index, 'black'] = 0
        df.loc[index, 'hisp'] = 0
        df.loc[index, 'white'] = 0
    if row['race'] == 3:
        df.loc[index, 'other'] = 0
        df.loc[index, 'asian'] = 0
        df.loc[index, 'black'] = 1
        df.loc[index, 'hisp'] = 0
        df.loc[index, 'white'] = 0
    if row['race'] == 4 or row['race'] == 5 or row['race'] == 6:
        df.loc[index, 'other'] = 0
        df.loc[index, 'asian'] = 0
        df.loc[index, 'black'] = 0
        df.loc[index, 'hisp'] = 1
        df.loc[index, 'white'] = 0
    if row['race'] == 7:
        df.loc[index, 'other'] = 0
        df.loc[index, 'asian'] = 0
        df.loc[index, 'black'] = 0
        df.loc[index, 'hisp'] = 0
        df.loc[index, 'white'] = 1

for index, row in df.iterrows():
    # print("-->fam_inc:", row['fam_inc'])
    if row['fam_inc'] == '':
        print("-->fam_inc=' '")
        df.loc[index, 'fam_inc'] = '-1'
    if row['age'] == '':
        df.loc[index, 'age'] = '-1'
    if row['parttime'] == '':
        df.loc[index, 'parttime'] = '-1'

df.to_csv('../dataset/law_school_data.csv')
df.info()


