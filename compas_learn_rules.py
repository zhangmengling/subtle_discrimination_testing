# %matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

compas_data = "./compas_data_binary.csv"
df = pd.read_csv(compas_data)
print(df)
df.info()

# turn into a binary classification problem
# create feature is_med_or_high_risk

# sex
# male = 1, female = 0
df['male'] = (df['sex'] == 'Male').astype(int)

# age
# 1, 2, 3, 4, 5, 6, 7, 8, 9
'''
for index, row in df.iterrows():
    if row['age'] <= 20:
        df.loc[index, 'age_degree'] = 1
    elif row['age'] > 20 and row['age'] <= 25:
        df.loc[index, 'age_degree'] = 2
    elif row['age'] > 25 and row['age'] <= 30:
        df.loc[index, 'age_degree'] = 3
    elif row['age'] > 30 and row['age'] <= 35:
        df.loc[index, 'age_degree'] = 4
    elif row['age'] > 35 and row['age'] <= 40:
        df.loc[index, 'age_degree'] = 5
    elif row['age'] > 40 and row['age'] <= 45:
        df.loc[index, 'age_degree'] = 6
    elif row['age'] > 45 and row['age'] <= 50:
        df.loc[index, 'age_degree'] = 7
    elif row['age'] > 50 and row['age'] <= 55:
        df.loc[index, 'age_degree'] = 8
    else:
        df.loc[index, 'age_degree'] = 9
'''
# race
# African_American = 1, Caucasian = 2, Hispanic = 3, Asian = 4, Native American = 5, other = 6
# for index, row in df.iterrows():
#     if row['race'] == 'African-American':
#         df.loc[index, 'race_degree'] = 1
#     elif row['race'] == 'Caucasian':
#         df.loc[index, 'race_degree'] = 2
#     elif row['race'] == 'Hispanic':
#         df.loc[index, 'race_degree'] = 3
#     elif row['race'] == 'Asian':
#         df.loc[index, 'race_degree'] = 4
#     elif row['race'] == 'Native American':
#         df.loc[index, 'race_degree'] = 5
#     else:
#         df.loc[index, 'race_degree'] = 6
'''
for index, row in df.iterrows():
    if row['race'] == 'African-American':
        df.loc[index, 'African-American'] = 1
        df.loc[index, 'Caucasian'] = 0
        df.loc[index, 'Hispanic'] = 0
        df.loc[index, 'Asian'] = 0
        df.loc[index, 'Native American'] = 0
        df.loc[index, 'other_race'] = 0
    elif row['race'] == 'Caucasian':
        df.loc[index, 'Caucasian'] = 1
        df.loc[index, 'African-American'] = 0
        df.loc[index, 'Hispanic'] = 0
        df.loc[index, 'Asian'] = 0
        df.loc[index, 'Native American'] = 0
        df.loc[index, 'other_race'] = 0
    elif row['race'] == 'Hispanic':
        df.loc[index, 'Hispanic'] = 1
        df.loc[index, 'Caucasian'] = 0
        df.loc[index, 'African-American'] = 0
        df.loc[index, 'Asian'] = 0
        df.loc[index, 'Native American'] = 0
        df.loc[index, 'other_race'] = 0
    elif row['race'] == 'Asian':
        df.loc[index, 'Asian'] = 1
        df.loc[index, 'African-American'] = 0
        df.loc[index, 'Caucasian'] = 0
        df.loc[index, 'Hispanic'] = 0
        df.loc[index, 'Native American'] = 0
        df.loc[index, 'other_race'] = 0
    elif row['race'] == 'Native American':
        df.loc[index, 'Native American'] = 1
        df.loc[index, 'Asian'] = 0
        df.loc[index, 'African-American'] = 0
        df.loc[index, 'Caucasian'] = 0
        df.loc[index, 'Hispanic'] = 0
        df.loc[index, 'other_race'] = 0
    else:
        df.loc[index, 'other_race'] = 1
        df.loc[index, 'Native American'] = 0
        df.loc[index, 'Asian'] = 0
        df.loc[index, 'African-American'] = 0
        df.loc[index, 'Caucasian'] = 0
        df.loc[index, 'Hispanic'] = 0
'''
# non protected features
# juv_fel_count (juvenile felony count)

# juv_misd_count (juvenile misdemeanor count)

# juv_other_count (juvenile other count)

# priors_count

# c_charge_degree(charge degree for this item F/M) --> is_felony
# for index, row in df.iterrows():
#     if row['c_charge_degree'] == 'M':
#         df.loc[index, 'felony'] = 0
#         df.loc[index, 'misdemeanor'] = 1
#     else:
#         df.loc[index, 'felony'] = 1
#         df.loc[index, 'misdemeanor'] = 0

# r_charge_degree(charge degree for recidivate) -->charge_degree_r
# for index, row in df.iterrows():
# #     if row['r_charge_degree'] == '(M1)':
# #         df.loc[index, 'charge_degree_r'] = 2
# #     elif row['r_charge_degree'] == '(F3)' or row['r_charge_degree'] == '(F5)' or row['r_charge_degree'] == '(F6)' or row['r_charge_degree'] == '(F7)':
# #         df.loc[index, 'charge_degree_r'] = 3
# #     elif row['r_charge_degree'] == '(F2)':
# #         df.loc[index, 'charge_degree_r'] = 4
# #     elif row['r_charge_degree'] == '(F1)':
# #         df.loc[index, 'charge_degree_r'] = 5
# #     elif row['r_charge_degree'] == '(M2)' or row['r_charge_degree'] == 'CO3' or row['r_charge_degree'] == 'MO3':
# #         df.loc[index, 'charge_degree_r'] = 1
# #     else:
# #         df.loc[index, 'charge_degree_r'] = 0
'''
for index, row in df.iterrows():
    if row['r_charge_degree'] == '(M1)':
        df.loc[index, 'M1'] = 1
        df.loc[index, 'M2'] = 0
        df.loc[index, 'F3'] = 0
        df.loc[index, 'F2'] = 0
        df.loc[index, 'F1'] = 0
    elif row['r_charge_degree'] == '(F3)' or row['r_charge_degree'] == '(F5)' or row['r_charge_degree'] == '(F6)' or row['r_charge_degree'] == '(F7)':
        df.loc[index, 'M1'] = 0
        df.loc[index, 'M2'] = 0
        df.loc[index, 'F3'] = 1
        df.loc[index, 'F2'] = 0
        df.loc[index, 'F1'] = 0
    elif row['r_charge_degree'] == '(F2)':
        df.loc[index, 'M1'] = 0
        df.loc[index, 'M2'] = 0
        df.loc[index, 'F3'] = 0
        df.loc[index, 'F2'] = 1
        df.loc[index, 'F1'] = 0
    elif row['r_charge_degree'] == '(F1)':
        df.loc[index, 'M1'] = 0
        df.loc[index, 'M2'] = 0
        df.loc[index, 'F3'] = 0
        df.loc[index, 'F2'] = 0
        df.loc[index, 'F1'] = 1
    elif row['r_charge_degree'] == '(M2)' or row['r_charge_degree'] == 'CO3' or row['r_charge_degree'] == 'MO3':
        df.loc[index, 'M1'] = 0
        df.loc[index, 'M2'] = 1
        df.loc[index, 'F3'] = 0
        df.loc[index, 'F2'] = 0
        df.loc[index, 'F1'] = 0
    else:
        df.loc[index, 'M1'] = 0
        df.loc[index, 'M2'] = 0
        df.loc[index, 'F3'] = 0
        df.loc[index, 'F2'] = 0
        df.loc[index, 'F1'] = 0
'''

#juv_fel_count
for index, row in df.iterrows():
    if row['juv_fel_count'] == 0:
        df.loc[index, 'juv_fel_count_0'] = 1
        df.loc[index, 'juv_fel_count_1'] = 0
        df.loc[index, 'juv_fel_count_2plus'] = 0
    elif row['juv_fel_count'] == 1:
        df.loc[index, 'juv_fel_count_0'] = 0
        df.loc[index, 'juv_fel_count_1'] = 1
        df.loc[index, 'juv_fel_count_2plus'] = 0
    else:
        df.loc[index, 'juv_fel_count_0'] = 0
        df.loc[index, 'juv_fel_count_1'] = 0
        df.loc[index, 'juv_fel_count_2plus'] = 1

#juv_misd_count
for index, row in df.iterrows():
    if row['juv_misd_count'] == 0:
        df.loc[index, 'juv_misd_count_0'] = 1
        df.loc[index, 'juv_misd_count_1'] = 0
        df.loc[index, 'juv_misd_count_2plus'] = 0
    elif row['juv_misd_count'] == 1:
        df.loc[index, 'juv_misd_count_0'] = 0
        df.loc[index, 'juv_misd_count_1'] = 1
        df.loc[index, 'juv_misd_count_2plus'] = 0
    else:
        df.loc[index, 'juv_misd_count_0'] = 0
        df.loc[index, 'juv_misd_count_1'] = 0
        df.loc[index, 'juv_misd_count_2plus'] = 1

# juv_other_count
for index, row in df.iterrows():
    if row['juv_other_count'] == 0:
        df.loc[index, 'juv_other_count_0'] = 1
        df.loc[index, 'juv_other_count_1'] = 0
        df.loc[index, 'juv_other_count_2plus'] = 0
    elif row['juv_other_count'] == 1:
        df.loc[index, 'juv_other_count_0'] = 0
        df.loc[index, 'juv_other_count_1'] = 1
        df.loc[index, 'juv_other_count_2plus'] = 0
    else:
        df.loc[index, 'juv_other_count_0'] = 0
        df.loc[index, 'juv_other_count_1'] = 0
        df.loc[index, 'juv_other_count_2plus'] = 1

# priors_count
for index, row in df.iterrows():
    if row['priors_count'] == 0:
        df.loc[index, 'priors_count_0'] = 1
        df.loc[index, 'priors_count_1'] = 0
        df.loc[index, 'priors_count_2'] = 0
        df.loc[index, 'priors_count_3plus'] = 0
    elif row['priors_count'] == 1:
        df.loc[index, 'priors_count_0'] = 0
        df.loc[index, 'priors_count_1'] = 1
        df.loc[index, 'priors_count_2'] = 0
        df.loc[index, 'priors_count_3plus'] = 0
    elif row['priors_count'] == 2:
        df.loc[index, 'priors_count_0'] = 0
        df.loc[index, 'priors_count_1'] = 0
        df.loc[index, 'priors_count_2'] = 1
        df.loc[index, 'priors_count_3plus'] = 0
    else:
        df.loc[index, 'priors_count_0'] = 0
        df.loc[index, 'priors_count_1'] = 0
        df.loc[index, 'priors_count_2'] = 0
        df.loc[index, 'priors_count_3plus'] = 1


df.info()
df.to_csv("compas_data_binary.csv")
read_csv = pd.read_csv("dataset/compas_data_binary.csv")
print(read_csv)












