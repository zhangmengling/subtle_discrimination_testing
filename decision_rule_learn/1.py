import pandas as pd
import numpy as np
# df = pd.read_csv('../dataset/compas_data_binary.csv')
# print(df.info)
#
# df['Sex1'] = df.sex.replace({1: "Male", 0: "Female"})
#
# dx = df[["AHD", "Sex1"]].dropna()
#
# pd.crosstab(dx.AHD, dx.Sex1)
#
# print(df.info)
# print(dx.info)
# print(pd)
#
# # dx = df[["AHD", "Sex1"]].dropna()
#
# # pd.crosstab(dx.AHD, dx.Sex1)
#
# # df.groupby("Sex1").agg({"Chol": [np.mean, np.std, np.size]})
# # print(df)
#
# df.groupby("sex").agg({"chol": [np.mean, np.std, np.size]})
# print("-->df", df)
#
#
# df = df[lambda x: x['sex'] == 0]
# # df.loc(df['sex'] == 0)
# print("------>df", df)
#
# labels = df.iloc[:,-2]
# # labels = df.iloc[:,4]
# print("------>label", labels)

# p_fm = 25/(72+25)
# n = 72+25
#
# se_female = np.sqrt(p_fm * (1 - p_fm) / n)
#
# z_score = 1.96
# lcb = p_fm - z_score* se_female  #lower limit of the CI
# ucb = p_fm + z_score* se_female  #upper limit of the CI
#
# print("-->confidence interval:", lcb, ucb)



# from statsmodels.stats.proportion import proportion_confint
# lower, upper = proportion_confint(88, 100, 0.05)
# print('lower=%.3f, upper=%.3f' % (lower, upper))


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

    return lcb, ucb

def calculate_confidence_interval1(samples):
    positive_num = samples.count(1)
    total_num = len(samples)
    proposition = float(positive_num) / total_num

    mean = np.mean(samples)
    print("-->mean", mean)

    # se1 = np.sqrt(proposition * (1 - proposition) / total_num)

    stdDev = np.std(samples, ddof=1)  ## divide n-1
    # stdDev = np.std(samples)   ## divide n
    print("-->stdDev", stdDev)

    z_score = 1.96
    lcb = mean - z_score * (stdDev/np.sqrt(total_num))
    ucb = mean + z_score * (stdDev/np.sqrt(total_num))
    print("-->confidence interval:", lcb, ucb)

    return lcb, ucb

# a = list(labels)
# print("-->a", a)
# print(type(a))
#
# calculate_confidence_interval(a)
# calculate_confidence_interval1(a)

import pandas as pd
import numpy as np
from csv import reader
import statistics

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

filename = "../dataset/communities.csv"

dataset = load_csv(filename)

print("-->dataset", dataset)

data = np.array(dataset[1:])
print("-->dataset", data)

all_stdDec = []
print("-->data[0]", data[0])

for i in range(0, len(data[0])):
    l = data[:,i]
    int_l = [float(x) for x in l]
    print(int_l)
    all_stdDec.append(statistics.stdev(int_l))

print("-->all stdDec", all_stdDec)

small_stdDev = []
median_stdDev = []
high_stdDev = []
# for std in all_stdDec:
for i in range(0, len(all_stdDec)):
    std = all_stdDec[i]
    if std < 0.2:
        small_stdDev.append(i)
    elif std < 0.4 and std >= 0.2:
        median_stdDev.append(i)
    else:
        high_stdDev.append(i)

print("-->small:", small_stdDev)
print("-->median", median_stdDev)
print("-->high", high_stdDev)

# statistics.stdev(A_rank)
