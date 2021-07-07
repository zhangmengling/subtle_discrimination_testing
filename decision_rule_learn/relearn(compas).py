
# code for IDS with deterministic local search
# requires installation of python package apyori: https://pypi.org/project/apyori/

import numpy as np
import pandas as pd
import math
from apyori import apriori

import itertools


import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import random
from random import choice
import matplotlib.pyplot as plt

from tensorflow.python.platform import flags
from data.census import census_data
from data.bank import bank_data
from data.credit import credit_data
from data.compas_two_year import compas_data
from data.law_school import law_school_data
from data.communities import communities_data
from utils.utils_tf import model_train, model_eval
from models.tutorial_models import dnn

from utils.config import census, credit, bank, compas, law_school, communities

from pandas.core.frame import DataFrame


from utils.utils_tf import model_prediction, model_argmax
from csv import reader

from itertools import combinations, permutations


FLAGS = flags.FLAGS


# rule is of the form if A == a and B == b, then class_1
# one of the member variables is itemset - a set of patterns {(A,a), (B,b)}
# the other member variable is class_label (e.g., class_1)
class rule:
    
    def __init__(self,feature_list,value_list,class_label):
        self.itemset = set()
        self.class_label = None
        self.add_item(feature_list,value_list)
        self.set_class_label(class_label)
    
    def add_item(self,feature_list,value_list):
        
        if len(feature_list) != len(value_list):
            print("Some error in inputting feature value pairs")
            return
        for i in range(0,len(feature_list)):
            self.itemset.add((feature_list[i],value_list[i]))
    
    def print_rule(self):
        s = "If "
        for item in self.itemset:
            s += str(item[0]) + " == " +str(item[1]) + " and "
        s = s[:-5]
        s += ", then "
        s += str(self.class_label)
        print(s)
        
    def all_predicates_same(self, r):
        return self.itemset == r.itemset
    
    def class_label_same(self,r):
        return self.class_label == r.class_label
            
    def set_class_label(self,label):
        self.class_label = label
        
    def get_length(self):
        return len(self.itemset)
    
    def get_cover(self, df):
        dfnew = df.copy()
        for pattern in self.itemset: 
            dfnew = dfnew[dfnew[pattern[0]] == pattern[1]]
        return list(dfnew.index.values)

    def get_correct_cover(self, df, Y):
        indexes_points_covered = self.get_cover(df) # indices of all points satisfying the rule
        Y_arr = pd.Series(Y)                    # make a series of all Y labels
        labels_covered_points = list(Y_arr[indexes_points_covered])   # get a list only of Y labels of the points covered
        correct_cover = []
        for ind in range(0,len(labels_covered_points)):
            if labels_covered_points[ind] == self.class_label:
                correct_cover.append(indexes_points_covered[ind])
        return correct_cover, indexes_points_covered
    
    def get_incorrect_cover(self, df, Y):
        correct_cover, full_cover = self.get_correct_cover(df, Y)
        return (sorted(list(set(full_cover) - set(correct_cover))))


# below function basically takes a data frame and a support threshold and returns itemsets which satisfy the threshold
def run_apriori(df, support_thres):
    # the idea is to basically make a list of strings out of df and run apriori api on it 
    # return the frequent itemsets
    dataset = []
    for i in range(0,df.shape[0]):
        temp = []
        for col_name in df.columns:
            temp.append(col_name+"="+str(df[col_name][i]))
        dataset.append(temp)

    # print("-->run apriori!!")

    results = list(apriori(dataset, min_support=support_thres))

    # print("-->finish running apriori")


    # print("--> apriori results:", results)

    list_itemsets = []
    for ele in results:
        temp = []
        for pred in ele.items:
            temp.append(pred)
        list_itemsets.append(temp)

    return list_itemsets


# This function converts a list of itemsets (stored as list of lists of strings) into rule objects
def createrules(freq_itemsets, labels_set):
    # create a list of rule objects from frequent itemsets 
    list_of_rules = []
    for one_itemset in freq_itemsets:
        feature_list = []
        value_list = []
        for pattern in one_itemset:
            fea_val = pattern.split("=")
            feature_list.append(fea_val[0])
            value_list.append(fea_val[1])
        for each_label in labels_set:
            temp_rule = rule(feature_list,value_list,each_label)
            list_of_rules.append(temp_rule)

    return list_of_rules


# compute the maximum length of any rule in the candidate rule set
def max_rule_length(list_rules):
    len_arr = []
    for r in list_rules:
        len_arr.append(r.get_length())
    return max(len_arr)


# compute the number of points which are covered both by r1 and r2 w.r.t. data frame df
def overlap(r1, r2, df):
    return sorted(list(set(r1.get_cover(df)).intersection(set(r2.get_cover(df)))))


# computes the objective value of a given solution set
def func_evaluation(soln_set, list_rules, df, Y, lambda_array):
    # evaluate the objective function based on rules in solution set 
    # soln set is a set of indexes which when used to index elements in list_rules point to the exact rules in the solution set
    # compute f1 through f7 and we assume there are 7 lambdas in lambda_array
    f = [] #stores values of f1 through f7; 
    
    # f0 term
    f0 = len(list_rules) - len(soln_set) # |S| - size(R)
    f.append(f0)
    
    # f1 term
    Lmax = max_rule_length(list_rules)
    sum_rule_length = 0.0
    for rule_index in soln_set:
        sum_rule_length += list_rules[rule_index].get_length()
    
    f1 = Lmax * len(list_rules) - sum_rule_length
    f.append(f1)
    
    # f2 term - intraclass overlap
    sum_overlap_intraclass = 0.0
    for r1_index in soln_set:
        for r2_index in soln_set:
            if r1_index >= r2_index:
                continue
            if list_rules[r1_index].class_label == list_rules[r2_index].class_label:
                sum_overlap_intraclass += len(overlap(list_rules[r1_index], list_rules[r2_index],df))
    f2 = df.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_intraclass
    f.append(f2)
    
    # f3 term - interclass overlap
    sum_overlap_interclass = 0.0
    for r1_index in soln_set:
        for r2_index in soln_set:
            if r1_index >= r2_index:
                continue
            if list_rules[r1_index].class_label != list_rules[r2_index].class_label:
                sum_overlap_interclass += len(overlap(list_rules[r1_index], list_rules[r2_index],df))
    f3 = df.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_interclass
    f.append(f3)
    
    # f4 term - coverage of all classes
    classes_covered = set() # set
    for index in soln_set:
        classes_covered.add(list_rules[index].class_label)
    f4 = len(classes_covered)
    f.append(f4)
    
    # f5 term - accuracy
    sum_incorrect_cover = 0.0
    for index in soln_set:
        sum_incorrect_cover += len(list_rules[index].get_incorrect_cover(df,Y))
    f5 = df.shape[0] * len(list_rules) - sum_incorrect_cover
    f.append(f5)
    
    #f6 term - cover correctly with at least one rule
    atleast_once_correctly_covered = set()
    for index in soln_set:
        correct_cover, full_cover = list_rules[index].get_correct_cover(df,Y)
        atleast_once_correctly_covered = atleast_once_correctly_covered.union(set(correct_cover))
    f6 = len(atleast_once_correctly_covered)
    f.append(f6)
    
    obj_val = 0.0
    for i in range(7):
        obj_val += f[i] * lambda_array[i]
    
    #print(f)
    return obj_val


# deterministic local search algorithm which returns a solution set as well as the corresponding objective value
def deterministic_local_search(list_rules, df, Y, lambda_array, epsilon):
    # step by step implementation of deterministic local search algorithm in the 
    # FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 4-5)
    
    #initialize soln_set
    soln_set = set()
    n = len(list_rules)
    
    # step 1: find out the element with maximum objective function value and initialize soln set with it
    each_obj_val = []
    for ind in range(len(list_rules)):
        each_obj_val.append(func_evaluation(set([ind]), list_rules, df, Y, lambda_array))
        
    best_element = np.argmax(each_obj_val)
    soln_set.add(best_element)
    S_func_val = each_obj_val[best_element]
    
    restart_step2 = False
    
    # step 2: if there exists an element which is good, add it to soln set and repeat
    while True:
        
        each_obj_val = []
        
        for ind in set(range(len(list_rules))) - soln_set:
            func_val = func_evaluation(soln_set.union(set([ind])), list_rules, df, Y, lambda_array)
            
            if func_val > (1.0 + epsilon/(n*n)) * S_func_val:
                soln_set.add(ind)
                print("Adding rule "+str(ind))
                S_func_val = func_val
                restart_step2 = True
                break
                
        if restart_step2:
            restart_step2 = False
            continue
            
        for ind in soln_set:
            func_val = func_evaluation(soln_set - set([ind]), list_rules, df, Y, lambda_array)
            
            if func_val > (1.0 + epsilon/(n*n)) * S_func_val:
                soln_set.remove(ind)
                print("Removing rule "+str(ind))
                S_func_val = func_val
                restart_step2 = True
                break
        
        if restart_step2:
            restart_step2 = False
            continue
        
        s1 = func_evaluation(soln_set, list_rules, df, Y, lambda_array)
        s2 = func_evaluation(set(range(len(list_rules))) - soln_set, list_rules, df, Y, lambda_array)
        
        print(s1)
        print(s2)
        
        if s1 >= s2:
            return soln_set, s1
        else: 
            return set(range(len(list_rules))) - soln_set, s2

def select_data(df, itemset, if_correct, if_multi):

    def conform(row, itemset):
        same = True
        for item in itemset:
            feature = item.split("=")[0]
            value = item.split("=")[1]
            if str(row[feature]) != value:
                same = False
        return same

    if if_multi == True:
        if if_correct == True:
            conform_dataset = []
            conform_index = []
            for index, row in df.iterrows():
                for item in itemset:
                    if conform(row, item) == True:
                        conform_dataset.append(row.tolist())
                        conform_index.append(index)
                        break
            return conform_dataset, conform_index
        else:
            not_conform_dataset = []
            not_conform_index = []
            # for row in df.rows:
            for index, row in df.iterrows():
                for item in itemset:
                    if conform(row, item) == False:
                        not_conform_dataset.append(row.tolist())
                        not_conform_index.append(index)
            return not_conform_dataset, not_conform_index

    else:
        if if_correct == True:
            conform_dataset = []
            conform_index = []
            # for row in df.rows:
            for index, row in df.iterrows():
                # print("-->row", row)
                if conform(row, itemset) == True:
                    conform_dataset.append(row.tolist())
                    conform_index.append(index)
            return conform_dataset, conform_index
        else:
            not_conform_dataset = []
            not_conform_index = []
            # for row in df.rows:
            for index, row in df.iterrows():
                # print("-->row", row)
                if conform(row, itemset) == False:
                    not_conform_dataset.append(row.tolist())
                    not_conform_index.append(index)
            return not_conform_dataset, not_conform_index

def calculate_probability(df_label, data, index, if_in_df):
    if if_in_df == True:
        positive_num = 0
        for i in index:
            all_num = len(index)
            # print(df_label.iloc[i][-1])
            label = df_label.iloc[i][-1]
            if label == 1:
                positive_num = positive_num + 1
        probability = positive_num/float(all_num)
        return probability

def fairness_score(df, df_label, itemset):
    selected_data, selected_index = select_data(df, itemset, True, False)
    other_data, other_index = select_data(df, itemset, False, False)
    # print("-->selected_index", len(selected_index))
    # print("-->other_index", len(other_index))
    select_prob = calculate_probability(df_label, selected_data, selected_index, True)
    other_prob = calculate_probability(df_label, other_data, other_index, True)
    score = select_prob - other_prob
    return abs(score)

def get_max_itemset_fairscore(dataset, conf):
    all_data = np.array(dataset)
    sensitive_feature = conf.sensitive_params[:]
    sensitive_feature = [x - 1 for x in sensitive_feature]
    sensitive_feature.append(conf.params)
    # print("-->sensitive_feature", sensitive_feature)
    data = all_data[:, sensitive_feature]
    # print("-->data", data)

    column_name = conf.sensitive_feature_name[:]
    column_name.append("label")
    new_data_label = pd.DataFrame(data, columns=column_name)
    # new_data_label = DataFrame(data)
    # print("-->generated_data")

    new_data = new_data_label.drop(new_data_label.columns[-1], axis=1, inplace=False)
    # print("-->new_data", new_data)

    itemsets = run_apriori(new_data, 0.01)
    print("-->itemsets:")
    # print(itemsets)
    print(len(itemsets))

    fair_scores = []
    for itemset in itemsets:
        itemset = list(itemset)
        fair_score = fairness_score(new_data, new_data_label, itemset)
        fair_scores.append(fair_score)
    max_fair_score = max(fair_scores)
    max_itemset = itemsets[fair_scores.index(max(fair_scores))]
    print("-->max itemset and max fairness score:")
    print(max_itemset, max_fair_score)
    return max_itemset, max_fair_score

# multiple constrains for data selection
def fairness_score_multi(df, itemset, if_multi):
    selected_data, selected_index = select_data(df, itemset, True, if_multi)
    other_data, other_index = select_data(df, itemset, False, if_multi)
    # print("-->selected_data", selected_data)
    # print("-->selected_index", len(selected_index))
    # print("-->other_data", other_data)
    # print("-->other_index", len(other_index))
    select_prob = calculate_probability(df, df_label, selected_data, selected_index, True)
    other_prob = calculate_probability(df, df_label, other_data, other_index, True)
    score = select_prob - other_prob
    return abs(score)


# df_label = pd.read_csv("../dataset/compas_data_binary_3sensitive.csv")
# df_label = pd.read_csv("../dataset/census_3sensitive.csv")
# df_label = pd.read_csv("../dataset/credit_2sensitive.csv")
# df_label = pd.read_csv("../dataset/bank_1sensitive.csv")
# df_label = pd.read_csv("../dataset/law_2sensitive.csv")
# df_label = pd.read_csv("../dataset/communities_3sensitive.csv")
# print("-->df", df_label)

# df = df_label.drop(columns=['is_med_or_high_risk', 'label'], inplace=False)
# df = df_label.drop(columns=['is_over_50k', 'label'], inplace=False)
# df = df_label.drop(columns=['is_good', 'label'], inplace=False)
# df = df_label.drop(columns=['if_get_term', 'label'], inplace=False)
# df = df_label.drop(columns=['pass_bar', 'label'], inplace=False)
# df = df_label.drop(columns=['high_crime_per', 'label'], inplace=False)
# print("-->df", df)

# itemsets = run_apriori(df, 0.01)
# print("-->itemsets:")
# print(itemsets)
#
# print(len(itemsets))

'''
# itemset = ['age_degree=2']
# selected_dataset, selected_index = select_data(df, itemset, True)
# print("-->selected_dataset", len(selected_dataset), type(selected_dataset))
# print(selected_dataset)
#
# fair_score = fairness_score(df, itemset)
# print("-->fair score:", fair_score)
'''


# fair_scores = []


# for itemset in itemsets:
#     itemset = list(itemset)
#     print("-->item", itemset)
#     fair_score = fairness_score(df, itemset)
#     print("-->fair score:", fair_score)
#     fair_scores.append(fair_score)
#
# print(min(fair_scores), max(fair_scores))



'''
from itertools import combinations, permutations

# itemset = [['race=Caucasian', 'age_degree=9'],['race=Caucasian', 'age_degree=8']]
# itemset = [['sex=1', 'race=0', 'age=4'], ['race=0', 'age=5', 'sex=1']]
# # itemset = [['sex=1', 'race=0', 'age=4']]
# # itemset = ['race=Caucasian', 'age_degree=9']
# fair_score = fairness_score_multi(df, itemset, True)
# print("-->fair score:", fair_score)
'''








def generate_random_data(max_num, conf, sess, x, preds):

    params = conf.params
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


def generate_random_data_itemset(max_num, conf, sess, x, preds, itemset, is_satisfy):
    params = conf.params
    limits = conf.constraints
    limits_no = conf.constraints_no
    all_data = []

    feature_name = conf.feature_name
    feature_indexs = []
    feature_values = []
    for i in itemset:
        feature = i.split("=")[0]
        value = i.split("=")[1]
        index = feature_name.index(feature)
        feature_indexs.append(index)

        if len(value) != 1:
            value_range = eval(value)
            feature_values.append(random.randint(value_range[0], value_range[1]))
        else:
            feature_values.append(int(value))

    while len(all_data) < max_num:
        if limits != []:
            data = [0] * params
            # satisfy itemset
            # for i in range(0, len(feature_indexs)):
            #     data[feature_indexs[i]] = feature_values[i]
            for i in limits_no:
                d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
                data[i] = d
            for limit in limits:
                # if np.array(limit).ndim == 1:
                if isinstance(limit[0],list):
                    # print("-->limit", limit)
                    i1 = choice([0, 1])
                    if i1 == 1:
                        data[limit[0][0]] = 1
                        i2 = choice(limit[1])
                        data[i2] = 1
                else:
                    i = choice(limit)
                    data[i] = 1
        else:
            data = []
            for i in range(params):
                d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
                data.append(d)

        is_append = True
        if is_satisfy == True:
            for index in conf.sensitive_params:
                data[index] = 0
            for i in range(0, len(feature_indexs)):
                data[feature_indexs[i]] = feature_values[i]
        else:
            for i in range(0, len(feature_indexs)):
                if data[feature_indexs[i]] == feature_values[i]:
                    is_append = False

        if is_append == True:
            probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            data.append(model_label)
            all_data.append(data)
        # all_data.append(data)
    # return data
    return np.array(all_data)





############ for range itemset generate random data #############
def generate_random_data_itemset1(max_num, conf, sess, x, preds, itemset, is_satisfy):
    params = conf.params
    limits = conf.constraints
    limits_no = conf.constraints_no
    all_data = []

    feature_name = conf.feature_name
    feature_indexs = []
    feature_values = []

    if len(itemset) == 3:
        for index in range(0, len(itemset)):
            i = itemset[index]
            feature = i.split("=")[0]
            value = i.split("=")[1]
            index = feature_name.index(feature)
            feature_indexs.append(index)

            if len(value) != 1:
                value_range = eval(value)
                feature_values.append(random.randint(value_range[0], value_range[1]))
            else:
                feature_values.append(int(value))
    else:
        for index in range(0, len(itemset)-1):
            i = itemset[index]
            feature = i.split("=")[0]
            value = i.split("=")[1]
            index = feature_name.index(feature)
            feature_indexs.append(index)

            if len(value) != 1:
                value_range = eval(value)
                feature_values.append(random.randint(value_range[0], value_range[1]))
            else:
                feature_values.append(int(value))

        race_feature = []
        race_value = []
        for index in range(len(itemset)-2, len(itemset)):
            # print("-->index", index)
            i = itemset[index]
            race_feature.append(i.split("=")[0])
            race_value.append(i.split("=")[1])
        # print("-->race_feature", race_feature)
        # print("-->race_value", race_value)
        random_race_index = choice([0, 1])
        r_feature = race_feature[random_race_index]
        r_value = race_feature[random_race_index]
        r_index = [feature_name.index(r) for r in race_feature]

        index = feature_name.index(r_feature)
        feature_indexs.append(index)
        feature_values.append(r_value)


    for i in itemset:
        feature = i.split("=")[0]
        value = i.split("=")[1]
        index = feature_name.index(feature)
        feature_indexs.append(index)

        if len(value) != 1:
            value_range = eval(value)
            feature_values.append(random.randint(value_range[0], value_range[1]))
        else:
            feature_values.append(int(value))

    while len(all_data) < max_num:
        if limits != []:
            data = [0] * params
            # satisfy itemset
            # for i in range(0, len(feature_indexs)):
            #     data[feature_indexs[i]] = feature_values[i]
            for i in limits_no:
                d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
                data[i] = d
            for limit in limits:
                # if np.array(limit).ndim == 1:
                if isinstance(limit[0],list):
                    # print("-->limit", limit)
                    i1 = choice([0, 1])
                    if i1 == 1:
                        data[limit[0][0]] = 1
                        i2 = choice(limit[1])
                        data[i2] = 1
                else:
                    i = choice(limit)
                    data[i] = 1
            # all_data.append(data)
            # probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
            # model_label = np.argmax(probs)  # GET index of max value in n_probs
            # data.append(model_label)
        else:
            data = []
            for i in range(params):
                d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
                data.append(d)

        is_append = True
        if is_satisfy == True:
            for index in conf.sensitive_params:
                data[index] = 0
            for i in range(0, len(feature_indexs)):
                data[feature_indexs[i]] = feature_values[i]
            if feature_values[-1] == 0:
                if len(itemset) != 3:
                    all_race = [2, 3, 4, 5, 6, 7]
                    for i in range(0, len(r_index)):
                        all_race.remove(r_index[i])
                    race_positive = choice(all_race)
                    data[race_positive] = 1
                else:
                    all_race = [2, 3, 4, 5, 6, 7]
                    all_race.remove(feature_indexs[-1])
                    race_positive = choice(all_race)
                    data[race_positive] = 1
        else:
            for i in range(0, len(feature_indexs)):
                if data[feature_indexs[i]] == feature_values[i]:
                    # print("-->false data", data)
                    is_append = False
            if len(itemset) != 3:
                for i in range(0, len(r_index)):
                    if data[r_index[i]] == int(race_value[i]):
                        is_append = False

        if is_append == True:
            probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            data.append(model_label)
            all_data.append(data)
        # all_data.append(data)
    print("-->all_data", all_data)
    return np.array(all_data)





# itemset = ['recePctWhite=0.0,0.1’])
# def satisfy_sensitive_range(x, conf, itemset):
#     result = True
#
#     for item in itemset:
#         feature = item.split("=")[0]
#         feature_range = [float(item.split("=")[1].split(",")[0]), float(item.split("=")[1].split(",")[1])]
#         feature_index = conf.feature_name.index(feature)
#         if x[feature_index] < feature_range[0] or x[feature_index] > feature_range[1]:
#             result = False
#     return result


def satisfy_sensitive_range(x, conf, itemset):
    result = True

    for item in itemset:
        feature = item.split("=")[0]
        feature_value = item.split("=")[1]

        if len(feature_value) == 1:
            feature_range = [int(feature_value)]
        else:
            feature_range = eval(feature_value)
            # feature_range = [float(item.split("=")[1].split(",")[0]), float(item.split("=")[1].split(",")[1])]

        feature_index = conf.feature_name.index(feature)
        if feature == 'race' or feature == 'sex':
            if x[feature_index] not in feature_range:
                result = False
        elif feature == 'age':
            if len(feature_range) == 1:
                if x[feature_index] not in feature_range:
                    result = False
            else:
                if x[feature_index] < int(feature_range[0]) or x[feature_index] > int(feature_range[1]):
                    result = False
    return result




#### generate samples satisfying itemset ( based on perturbation)
# itemset = [[0.0, 0.1], [...], [...]]
def generate_pert_samples(dataset, conf, max_num, itemset, sess, x, preds, is_satisfy):
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
    feature_names = [item.split("=")[0] for item in itemset]
    feature_indexs = [conf.feature_name.index(feature_name) for feature_name in feature_names]

    # set perturbation range for non_sensitive features
    input_range = conf.input_bounds
    for r in input_range:
        if isinstance(r[0], float):
            r = 0.05

    # pert_range = [-0.05, 0.05]
    pert_range = [-1, 0, 1]

    def clip(input, conf):
        """
        Clip the generating instance with each feature to make sure it is valid
        """
        for i in range(len(input)):
            input[i] = max(input[i], conf.input_bounds[i][0])
            input[i] = min(input[i], conf.input_bounds[i][1])
        return input

    def perturbation_sample(data, max_num, pert_range):
        constraints = conf.constraints
        # for i in range(1, len(constraints)):

        pert_samples = []

        while len(pert_samples) < max_num:
            sample = data[:]
            for i in range(1, len(constraints)):
                is_perturb = choice([0, 1])
                if is_perturb == 1:
                    pert_value = choice(constraints[i])
                    for index in constraints[i]:
                        sample[index] = 0
                    sample[pert_value] = 1
            new_sample = clip(sample, conf)
            # add prediction label
            probs = model_prediction(sess, x, preds, np.array([new_sample]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            list(new_sample).append(model_label)

            pert_samples.append(new_sample)
        return pert_samples


    all_new_samples = []
    if is_satisfy == True:
        while len(all_new_samples) < max_num:
            for data in dataset:
                # if satisfy_sensitive(data, sensitive_feature_set, itemset) == True:
                if satisfy_sensitive_range(data, conf, itemset) == True:
                    new_samples = perturbation_sample(data, 1, pert_range)
                    all_new_samples.append(list(new_samples[0]))
                    if len(all_new_samples) > max_num:
                        break
            if all_new_samples == []:
                break
    else:
        while len(all_new_samples) < max_num:
            for data in dataset:
                if satisfy_sensitive_range(data, conf, itemset) == False:
                    print("--> satisfy false")
                # if satisfy_sensitive(data, sensitive_feature_set, itemset) == False:
                    new_samples = perturbation_sample(data, 1, pert_range)
                    all_new_samples.append(list(new_samples[0]))
                    if len(all_new_samples) > max_num:
                        break
            if all_new_samples == []:
                break

    # all_new_samples = choice(all_new_samples, max_num)
    # all_new_samples = list(set(all_new_samples))
    # print("-->all_new_samples")
    # print(all_new_samples)
    return all_new_samples






# data = []
# for i in range(params):
#     d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
#     data.append(d)
# # all_data.append(data)
# probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
# model_label = np.argmax(probs)  # GET index of max value in n_probs
# data.append(model_label)


def satisfy_sensitive(x, sensitive_feature, sensitive_range):
    item = 0
    result = True
    for feature in sensitive_feature:
        # print(x[feature], [list(sensitive_range)[item]])
        if x[feature] not in [list(sensitive_range)[item]]:
            result = False
        item = item + 1
    return result


def generate_proposition(dataset, conf): # input dataset: X (32561, 13)
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
    sensitive_input_bounds = []
    for s in sensitive_feature_set:
        sensitive_input_bounds.append(conf.input_bounds[s])

    sensitive_input_range = []
    for input_bounds in sensitive_input_bounds:
        inputs_range = list(range(input_bounds[0], input_bounds[1] + 1))
        sensitive_input_range.append(inputs_range)
    print("-->sensitive_input_bounds", sensitive_input_range)

    # for i in range(len(sensitive_input_range)):
    #     locals()['s' + str(i)] = sensitive_input_range[i]
    # s1 = sensitive_input_range[0]
    # s2 = sensitive_input_range[1]
    # s3 = sensitive_input_range[2]
    # combination = list(itertools.product(s1, s2, s3))

    # change itemset
    s1 = sensitive_input_range[0]
    # s2 = sensitive_input_range[2]
    combination = list(itertools.product(s1))
    print("-->res", combination)

    propositions = []
    all_num = len(dataset)
    for sensitive_range in combination:
        satisfy_num = 0
        for data in dataset:
            # print(data, sensitive_range)
            if satisfy_sensitive(data, sensitive_feature_set, sensitive_range) == True:
                # print("True")
                satisfy_num = satisfy_num + 1
        # print("-->satisfy_num", satisfy_num)
        pro = float(satisfy_num)/all_num
        propositions.append(pro)
    return propositions, combination

###################################  generate all possible itemsets based on conf ################################
def generate_all_itemsets(conf):
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
    print("-->sensitive_feautre_set", sensitive_feature_set)
    sensitive_input_bounds = []
    for s in sensitive_feature_set:
        sensitive_input_bounds.append(conf.input_bounds[s])

    sensitive_input_range = []
    for input_bounds in sensitive_input_bounds:
        inputs_range = list(range(input_bounds[0], input_bounds[1] + 1))
        sensitive_input_range.append(inputs_range)
    # print("-->sensitive_input_bounds", sensitive_input_range)

    # for i in range(len(sensitive_input_range)):
    #     locals()['s' + str(i)] = sensitive_input_range[i]
    s1 = sensitive_input_range[0]
    s2 = sensitive_input_range[1]
    s3 = sensitive_input_range[2]

    s3 = [2, 3, 4, 5, 6, 7]
    s3 = list(itertools.product(s3, [0, 1]))

    combination = list(itertools.product(s1, s2, s3))

    # all_itemsets = []
    # # for item in combination:
    # for item_index in range(0, len(combination)):
    #     itemset = []
    #     # item = list(item)
    #     item = list(combination[item_index])
    #     for i in range(0, len(item)):
    #         one_item = conf.sensitive_feature_name[i] + "=" + str(list(item)[i])
    #         itemset.append(one_item)
    #     all_itemsets.append(itemset)
    #
    # # all_itemsets1 = all_itemsets*6
    # all_itemsets1 = []
    # for num in range(6):
    #     for item in all_itemsets:
    #         feature_name = conf.sensitive_feature_name[num+2]
    #         new_item = item[:]
    #         new_item[-1] = feature_name + "=" + item[-1].split("=")[1]
    #         # item[-1] = feature_name + "=" + item[-1].split("=")[1]
    #         all_itemsets1.append(new_item)

    all_itemsets = []
    for item in combination:
        ##  e.g. (1, 0, (2, 0))
        itemset = []
        item = list(item)
        for i in range(0, len(item)):
            # item = 1 / 0 / (2, 0)
            if i == 0 or i == 1:
                one_item = conf.sensitive_feature_name[i] + "=" + str(item[i])
                itemset.append(one_item)
            else:
                race_item = item[i]
                one_item = conf.sensitive_feature_name[race_item[0]] + "=" + str(race_item[1])
                itemset.append(one_item)
                # for race
                # race = (2, 0) / ((2, 0), (2, 1))

                # if len(str(race_item[0])) != 1:
                #     for race in race_item:
                #         one_item = conf.sensitive_feature_name[race[0]] + "=" + str(race[1])
                #         itemset.append(one_item)
                # else:
                #     one_item = conf.sensitive_feature_name[race_item[0]] + "=" + str(race_item[1])
                #     itemset.append(one_item)

        all_itemsets.append(itemset)

    return all_itemsets


####################  generate all range itemsets (e.g. ['age=1,5', 'sex=1', 'race=white']  #####################
def generate_all_itemset_range(conf):
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
    print("-->sensitive_feautre_set", sensitive_feature_set)
    sensitive_input_bounds = []
    for s in sensitive_feature_set:
        sensitive_input_bounds.append(conf.input_bounds[s])

    sensitive_input_range = []
    for input_bounds in sensitive_input_bounds:
        inputs_range = list(range(input_bounds[0], input_bounds[1] + 1))
        sensitive_input_range.append(inputs_range)

    sensitive_name = conf.sensitive_feature_name


    print("-->sensitive feature set range")
    print(sensitive_feature_set)
    print(sensitive_input_range)
    print(sensitive_name)

    s1 = sensitive_input_range[0]
    s2 = sensitive_input_range[1]
    s3 = sensitive_input_range[2]

    # com1 = list(itertools.product(s1))
    # age
    s1 = s1 + list(itertools.combinations(s1, 2))
    del s1[7]
    print("-->s1", s1)

    s3 = [2, 3, 4, 5, 6, 7]
    s3 = list(itertools.product(s3, [0, 1]))
    # s3_1 = list(itertools.combinations([2, 3, 4, 5, 6, 7], 2))
    # s3_1 = list(itertools.product(s3_1, [0, 1]))
    s3 = s3 + list(itertools.combinations(s3, 2))

    print(s3)

    s3_new = []
    for s in s3:
        # print(s)
        if len(str(s[0])) == 1:
            s3_new.append(s)
        if len(str(s[0])) != 1:
            if s[0][0] != s[1][0] and int(s[0][1]) + int(s[1][1]) == 2:
                s3_new.append(s)
    s3 = s3_new[:]

    # for s in s3:
    #     print(s)
    #     if len(str(s[0])) != 1:
    #         if s[0][0] == s[1][0]:
    #             if int(s[0][1])+int(s[1][1]) == 0:
    #                 print(int(s[0][1])+int(s[1][1]) )
    #                 s3.remove(s)
    print("-->s3", s3)

    combination = list(itertools.product(s1, s2, s3))
    print("-->combination:", combination)

    all_itemsets = []
    for item in combination:
        ##  e.g. (1, 0, (2, 0))
        itemset = []
        item = list(item)
        for i in range(0, len(item)):
            # item = 1 / 0 / (2, 0)
            if i == 0 or i == 1:
                one_item = conf.sensitive_feature_name[i] + "=" + str(item[i])
                itemset.append(one_item)
            else:
                # for race
                # race = (2, 0) / ((2, 0), (2, 1))
                race_item = item[i]
                if len(str(race_item[0])) != 1:
                    for race in race_item:
                        one_item = conf.sensitive_feature_name[race[0]] + "=" + str(race[1])
                        itemset.append(one_item)
                else:
                    one_item = conf.sensitive_feature_name[race_item[0]] + "=" + str(race_item[1])
                    itemset.append(one_item)

        all_itemsets.append(itemset)

    return all_itemsets











def generate_sensitive_part(max_num, conf, pro, com):
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
    pro_num = [round(x*max_num) for x in pro]
    print("-->pro_num", pro_num)
    all_sensitive_feature = []
    for i in range(len(pro_num)):
        sensitive_feature = list(com[i])
        # print("-->sensitive_feature", sensitive_feature)
        for n in range(pro_num[i]):
            all_sensitive_feature = all_sensitive_feature + [sensitive_feature]
        new_all_sensitive_feature = [x for x in all_sensitive_feature if x]
    return new_all_sensitive_feature


def generate_pro_data(max_num, conf, sess, x, preds, sensitive_feature):
    max_num = len(sensitive_feature)
    params = conf.params
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
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

    new_data = []
    # for data in all_data:
    for i in range(len(all_data)):
        data = all_data[i]
        sensitive_f = sensitive_feature[i]
        for j in range(len(sensitive_feature_set)):
            data[sensitive_feature_set[j]] = sensitive_f[j]
        new_data.append(data)
    return new_data

def get_sensitive_data(all_data, sensitive_feature_set):
    print("-->feature_set", sensitive_feature_set)
    data = np.array(all_data)
    new_data = data[:, sensitive_feature_set]
    return new_data










#################################### hypothesis testing SPRT test ####################################
# threshold: fairness score (e.g. 0.2, 0.3...)
def sprt_detect(threshold, conf, sess, x, preds):

    def sprt_one_figure(prs, accept_pr, deny_pr, threshold):
        length = len(prs)
        Y = list(range(0, length))
        title_name = "threshold=" + str(threshold)
        plt.title(title_name)
        accept_prs = [accept_pr]*length
        deny_prs = [deny_pr]*length
        plt.plot(Y, accept_prs, color='black', linestyle="--", label="accept_bound")
        plt.plot(Y, deny_prs, color='black', linestyle=":", label="deny_bound")
        plt.plot(Y, prs, label="threshold=" + str(threshold))
        plt.legend()
        plt.xlabel('number of detected testing')
        plt.ylabel('rate')
        plt.show()

    def calculate_sprt_ratio(c, n):
        '''
        :param c: number of model which lead to label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''
        p1 = threshold + sigma
        p0 = threshold - sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))

    # threshold = 0.75
    sigma = 0.05
    beta = 0.05
    alpha = 0.05
    max_iteration = 1000

    accept_pr = np.log((1 - beta) / alpha)
    deny_pr = np.log(beta / (1 - alpha))

    print("-->accept/deny pr:", accept_pr, deny_pr)
    print("-->p0, p1:", threshold + sigma, threshold - sigma)

    # same_count = 0
    satisfy_count = 0
    total_count = 0

    # length = len(original_labels)

    prs = []
    for i in range(0, max_iteration):
        print("-->iteration:", i)
        #######################  generate 1000 new samples and calculate maximum fairness score  ##################
        pr = calculate_sprt_ratio(satisfy_count, total_count)
        print("-->", satisfy_count, total_count)
        print("-->pr:", pr)
        prs.append(pr)
        total_count += 1
        all_data = generate_random_data(1000, conf, sess, x, preds)

        max_itemset, max_fair_score = get_max_itemset_fairscore(all_data, conf)

        if max_fair_score >= threshold:
            satisfy_count += 1

        if pr >= accept_pr:
            print("Accept -->last pr:", pr)
            prs[-1] = accept_pr
            sprt_one_figure(prs, accept_pr, deny_pr, threshold)
            return True, satisfy_count, total_count, prs, accept_pr, deny_pr
        if pr <= deny_pr:
            print("Deny -->last pr:", pr)
            prs[-1] = deny_pr
            sprt_one_figure(prs, accept_pr, deny_pr, threshold)
            return False, satisfy_count, total_count, prs, accept_pr, deny_pr
        if total_count >= max_iteration:
            return 0, satisfy_count, total_count, prs, accept_pr, deny_pr



##########################  confidence interval of population proposition ###########################
def calculate_confidence_interval(samples):
    # positive_num = 0
    # total_num = len(samples)
    # for x in samples:
    #      if x == 1:
    #          positive_num += 1
    positive_num = samples.count(1)
    total_num = len(samples)
    proportion = float(positive_num)/total_num
    print("-->proportion", proportion)
    se = np.sqrt(proportion * (1 - proportion) / total_num)
    print("-->se", se)

    z_score = 1.96
    lcb = proportion - z_score * se
    ucb = proportion + z_score * se
    print("-->confidence interval:", lcb, ucb)
    print("-->%s +- %s" % (proportion, z_score * se))

    return proportion, z_score * se


##########################  confidence interval of mean ###########################
def calculate_confidence_interval1(samples):
    positive_num = samples.count(1)
    total_num = len(samples)
    proportion = float(positive_num) / total_num

    # mean = np.mean(samples)
    # print("-->mean", mean)

    # se1 = np.sqrt(proposition * (1 - proposition) / total_num)

    stdDev = np.std(samples, ddof=1)  ## divide n-1
    print("-->stdDev", stdDev)

    z_score = 1.96
    lcb = proportion - z_score * (stdDev / np.sqrt(total_num))
    ucb = proportion + z_score * (stdDev / np.sqrt(total_num))
    print("-->confidence interval:", lcb, ucb)

    return lcb, ucb



def generate_pert_samples_with_labels(dataset, conf, max_num, itemset, sess, x, preds, is_satisfy, is_positive):
    '''
    is_positive: True-->generate positive samples; Flase-->generate negative samples
    '''
    sensitive_feature_set = [x - 1 for x in conf.sensitive_params]
    feature_names = [item.split("=")[0] for item in itemset]
    # feature_indexs = [conf.feature_name.index(feature_name) for feature_name in feature_names]

    if is_positive == True:
        label = 1
    else:
        label = 0

    # def change_sensitive_range(x, itemset):
    #     for item in itemset:
    #         feature = item.split("=")[0]
    #         feature_value = item.split("=")[1]
    #
    #         feature_index = conf.feature_name.index(feature)
    #         if feature == 'race' or feature == 'sex':
    #             if len(feature_value) == 1:
    #                 feature_range = [int(feature_value)]
    #                 x[feature_index] = feature_range[0]
    #             else:
    #                 feature_range = eval(feature_value)
    #                 x[feature_index] = random.choice(feature_range)
    #         elif feature == 'age':
    #             if len(feature_value) == 1:
    #                 feature_range = [int(feature_value)]
    #                 x[feature_index] = feature_range[0]
    #             else:
    #                 feature_range = eval(feature_value)
    #                 x[feature_index] = random.randint(feature_range[0], feature_range[1])
    #     return x
    #
    # input_range = conf.input_bounds
    # print(feature_names)
    # print(feature_indexs)
    # print(input_range)

    def change_sensitive_range(x, itemset):
        for i in range(0, 8):
            x[i] = 0

        for item in itemset:
            feature = item.split("=")[0]
            feature_value = item.split("=")[1]

            if feature == "male":
                feature_index = conf.feature_name.index(feature)
                x[feature_index] = feature_value
            elif feature == "age_degree":
                feature_index = conf.feature_name.index(feature)
                if len(feature_value) == 1:
                    x[feature_index] = int(feature_value)
                else:
                    feature_range = eval(feature_value)
                    x[feature_index] = random.randint(feature_range[0], feature_range[1])
            else:
                feature_index = conf.feature_name.index(feature)
                x[feature_index] = 1

        return x

    input_range = conf.input_bounds
    print(feature_names)
    # print(feature_indexs)
    print(input_range)

    def clip(input, conf):
        """
        Clip the generating instance with each feature to make sure it is valid
        """
        for i in range(len(input)):
            input[i] = max(input[i], conf.input_bounds[i][0])
            input[i] = min(input[i], conf.input_bounds[i][1])
        return input

    def select_positive_samples(is_positive):
        pos_neg_data = []
        for data in dataset:
            data = new_sample = clip(data, conf)
            probs = model_prediction(sess, x, preds, np.array([data]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            if model_label == label: # ==0
                data = list(data)
                data.append(label)
                pos_neg_data.append(data)
        return pos_neg_data

    def change_based_itemset(new_dataset):
        satisfy_data = []
        random_index = random.sample(range(0, len(new_dataset)), max_num)
        print("len random_index", len(random_index))
        for index in random_index:
            data = new_dataset[index]
            new_data = change_sensitive_range(data, itemset)
            satisfy_data.append(list(new_data))
        return satisfy_data

    pos_nes_dataset = select_positive_samples(is_positive)
    all_new_samples = change_based_itemset(pos_nes_dataset)

    half_index = random.sample(range(0, len(all_new_samples)), len(all_new_samples)//2)
    for i in range(0, len(all_new_samples)):
        if i in half_index:
            all_new_samples[i][2] = 0
        else:
            all_new_samples[i][7] = 0

    return all_new_samples








# df_label = pd.read_csv("../dataset/compas_data_binary_3sensitive.csv")
# df = df_label.drop(columns=['is_med_or_high_risk', 'label'], inplace=False)
# random_seed = 999

data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data,
        "law_school": law_school_data, "communities": communities_data}
data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas,
               "law_school": law_school, "communities": communities}


dataset = "compas"
filename = "../dataset/compas_data_binary.csv"
X, Y, input_shape, nb_classes = data[dataset]()

# print("-->X", X)
# print(X.shape)

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
# saver.restore(sess, "../models/credit/999/test.model")
saver.restore(sess, "../models/compas_two/1999/test.model")
# grad_0 = gradient_graph(x, preds)
# tfops = tf.sign(grad_0)





all_fair_scores = []
all_scores = []
all_itemsets = [['age_degree=(6, 9)', 'male=1', 'other_race=1', 'Hispanic=1']]
for itemset in all_itemsets:
    print("-->itemset:", itemset)
    # all_data = generate_random_data_itemset(1000, conf, sess, x, preds, itemset, True)
    generated_data = generate_pert_samples_with_labels(list(X), conf, 1000, itemset, sess, x, preds, True, True)

print("-->generated_data")
print(generated_data)
file_path = "../dataset/compas_2.csv"
data_pd = DataFrame(generated_data)
data_pd.to_csv(file_path)




