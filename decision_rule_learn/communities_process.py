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

data = "../dataset/communities.csv"
df = pd.read_csv(data, keep_default_na=False)
df.info()

for index, row in df.iterrows():
    if float(row['ViolentCrimesPerPop']) > 0.15:
        df.loc[index, 'high_crime_per'] = 1
    else:
        df.loc[index, 'high_crime_per'] = 0

df.to_csv('../dataset/communities.csv')


