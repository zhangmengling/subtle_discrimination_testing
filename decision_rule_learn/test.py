# %matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from random import seed
from random import randrange
from utils.utils_tf import model_prediction, model_argmax
from random import choice
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.utils_tf import model_train, model_eval
from models.tutorial_models import dnn

from data.census import census_data
from data.bank import bank_data
from data.credit import credit_data
from data.compas_two_year import compas_data
from data.law_school import law_school_data
from data.communities import communities_data
from utils.config import census, credit, bank, compas, law_school, communities

# url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
# df = pd.read_csv(url)
# df.info()

# compas_data = "compas-scores-two-years.csv"
# df = pd.read_csv(compas_data)
# print(df)
# df.info()
#
# # turn into a binary classification problem
# # create feature is_med_or_high_risk
# df['is_med_or_high_risk']  = (df['decile_score']>=5).astype(int)
# df.info()
# df.to_csv("compas_data.csv")
# read_csv = pd.read_csv("compas_data.csv")
# print(read_csv)


'''
def generate_random_data(max_num, conf, sess, x, preds):

    params = conf.params
    sensitive_feature_set = []
    all_data = []
    while len(all_data) < max_num:
        data = []
        for i in range(params):
            d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
            data.append(d)
        # all_data.append(data)
        probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
        model_label = np.argmax(probs)  # GET index of max value in n_probs
        data.append(model_label)
        all_data.append(data)
    return all_data

df_label = pd.read_csv("../dataset/compas_data_binary_3sensitive.csv")
df = df_label.drop(columns=['is_med_or_high_risk', 'label'], inplace=False)
random_seed = 999

data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data,
        "law_school": law_school_data, "communities": communities_data}
data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas,
               "law_school": law_school, "communities": communities}


dataset = "census"
filename = "../dataset/census.csv"
X, Y, input_shape, nb_classes = data[dataset]()
config = tf.ConfigProto()
conf = data_config[dataset]
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
x = tf.placeholder(tf.float32, shape=input_shape)
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
model = dnn(input_shape, nb_classes)
preds = model(x)
# print("-->preds ", preds)
saver = tf.train.Saver()
saver.restore(sess, "../models/census/test.model")
# grad_0 = gradient_graph(x, preds)
# tfops = tf.sign(grad_0)

all_data = generate_random_data(1000, conf, sess, x, preds)
print("-->all_data")
print(all_data)
'''

population = ['red', 'red', 'red', 'blue', 'green']

random.sample(['red', 'blue'], counts=[4, 2], k=5)
print(random.sample(['red', 'red', 'red', 'red', 'blue', 'blue'], k=5))
print(random.sample(population, k=5))





def generate_random_data(max_num, conf, sess, x, preds):

    params = conf.params
    limits = conf.constraints

    all_data = []
    while len(all_data) < max_num:
        if limits != []:
            data = [0] * params
            for limit in limits:
                print("-->cons", limit)
                if len(limit) == 1:
                    i = choice(limit)
                    data[i] = 1
                else:
                    i1 = choice([0, 1])
                    if i1 == 1:
                        data[limit[0]] = 1
                        i2 = choice(limit[1])
                        data[i2] = 1
            all_data.append(data)
        else:
            data = []
            for i in range(params):
                d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
                data.append(d)
            # all_data.append(data)
            probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            data.append(model_label)
            all_data.append(data)

    return all_data


def generate_random_data_itemset(max_num, conf, sess, x, preds, itemset, is_satisfy):
    params = conf.params
    limits = conf.constraints
    all_data = []

    feature_name = conf.feature_name
    feature_indexs = []
    feature_values = []
    for i in itemset:
        feature = i.split("=")[0]
        value = i.split("=")[1]
        index = feature_name.index(feature)
        feature_indexs.append(index)
        feature_values.append(int(value))

    while len(all_data) < max_num:
        if limits != []:
            data = [0] * params
            for limit in limits:
                print("-->cons", limit)
                if len(limit) == 1:
                    i = choice(limit)
                    data[i] = 1
                else:
                    i1 = choice([0, 1])
                    if i1 == 1:
                        data[limit[0]] = 1
                        i2 = choice(limit[1])
                        data[i2] = 1
            all_data.append(data)
        else:
            data = []
            for i in range(params):
                d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
                data.append(d)
            # all_data.append(data)
            probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            data.append(model_label)
            all_data.append(data)


        # data = []
        # for i in range(params):
        #     d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
        #     data.append(d)
        # # all_data.append(data)
        # probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
        # model_label = np.argmax(probs)  # GET index of max value in n_probs
        # data.append(model_label)

        # confirm sensitive features satisfy given itemset
        is_append = True
        if is_satisfy == True:
            for i in range(0, len(feature_indexs)):
                data[feature_indexs[i]] = feature_values[i]
        else:
            for i in range(0, len(feature_indexs)):
                if data[feature_indexs[i]] == feature_values[i]:
                    is_append = False

        if is_append == True:
            all_data.append(data)
        # all_data.append(data)
    # return data
    return np.array(all_data)








