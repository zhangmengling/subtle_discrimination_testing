# -*- coding: utf-8 -*
class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])  #sensitive：gender
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    sensitive_params = [1, 8, 9]
    # age, race, sex

    sensitive_feature_name = ['age', 'race', 'sex']

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race", "sex", "√",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    pert_range = [[-1, 0, 1], [-1, 0, 1], [-2, -1, 0, 1, 2], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                  [-1, 0, 1], [-1, 0, 1], [0], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], [0]]

    constraints = []

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    sensitive_params = [9, 13]
    # sex, age

    sensitive_feature_name = ['sex', 'age']

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                                                                      "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    pert_range = [[-1, 0, 1], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], [-1, 0, 1], [-1, 0, 1],
                  [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1],[-1, 0, 1],[-1, 0, 1],[0]]

    constraints = []
    constraints_no = []

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    # input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    sensitive_params = [1]
    # age

    sensitive_feature_name = ['age']

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                                                                      "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    pert_range = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [0], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], [-1, 0, 1],
                  [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

    constraints = []

class compas:
    """
        Configuration of dataset Bank Marketing
        """

    # the size of total features
    params = 29

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    feature_name = ['age_degree', 'male', 'other_race', 'Native American', 'Asian', 'African-American', 'Caucasian',
                    'Hispanic', 'juv_fel_count_0', 'juv_fel_count_1', 'juv_fel_count_2plus', 'juv_misd_count_0',
                    'juv_misd_count_1', 'juv_misd_count_2plus', 'juv_other_count_0', 'juv_other_count_1', 'juv_other_count_2plus',
                    'priors_count_0', 'priors_count_1', 'priors_count_2', 'priors_count_3plus', 'felony', 'misdemeanor',
                    'is_recid', 'M1', 'M2', 'F3', 'F2', 'F1']

    sensitive_params = [1, 2, 3, 4, 5, 6, 7, 8]

    sensitive_feature_name = ['age_degree', 'male', 'other_race', 'Native American', 'Asian', 'African-American',
                              'Caucasian', 'Hispanic']

    # the name of each class
    class_name = ["not_med_or_high_risk", "is_med_or_high_risk"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28]

    constraints_no = [0, 1]
    constraints = [[2, 3, 4, 5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19, 20], [21, 22], [[23], [24, 25, 26, 27, 28]]]

    # sensitive_constraints = [[1], [2], [3, 4, 5, 6, 7, 8]]

class law_school:
    """
           Configuration of dataset Bank Marketing
           """

    # the size of total features
    params = 15

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 10])
    input_bounds.append([1, 10])
    input_bounds.append([1, 10])
    input_bounds.append([1, 2])
    input_bounds.append([1, 8])
    input_bounds.append([1, 6])
    input_bounds.append([11, 48])  # # float
    input_bounds.append([1.5, 3.9])  # # float
    input_bounds.append([-3.35, 3.25])  # # float
    input_bounds.append([-6.44, 3.445])  # # float
    input_bounds.append([0, 0])
    input_bounds.append([0, 0])
    input_bounds.append([-1, 5])
    input_bounds.append([-69, -1])
    input_bounds.append([-1, 1])


    pert_range = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                  [-1, 0, 1],
                  [-0.1, 0, 0.1],
                  [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04,
                   0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                  [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04,
                   0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                  [0], [0], [-1, 0, 1], [-1, 0, 1], [-1, 0, 0, 0, 1]]

    feature_name = ['decile1b', 'decile3', 'decile1', 'sex', 'race', 'cluster', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'bar1',
                    'bar2', 'fam_inc', 'age', 'parttime']

    sensitive_params = [4, 5]

    sensitive_feature_name = ['sex', 'race']

    # the name of each class
    class_name = ["not_pass_bar", "pass_bar"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    constraints_no = []
    constraints = []

class communities:
    """
               Configuration of dataset Bank Marketing
               """

    # the size of total features
    params = 100
    # the valid religion of each feature

    percentage_range = [[0.0, 1.0]]

    # input_bounds = []
    # input_bounds.append([1, 10])
    input_bounds = percentage_range*70
    input_bounds.append([0.0, 0.5, 1])  # No.71 feature
    input_bounds2 = percentage_range*29
    input_bounds = input_bounds + input_bounds2

    sensitive_params = [2, 3, 4, 5, 6, 7, 8, 9, 38, 39, 40]
    sensitive_feature_name = ['recePctblack', 'recePctWhite', 'recePctAsian', 'racePctHisp',
                              'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
                              'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv']

    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                            71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                            94, 95, 96, 97, 98, 99]

    pert_range = []

    feature_name = ['population', 'householdsize', 'recePctblack', 'recePctWhite', 'recePctAsian', 'racePctHisp',
                    'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'PctUrban', 'medIncome',
                    'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc',
                    'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap',
                    'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
                    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce',
                    'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
                    'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg',
                    'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig',
                    'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
                    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
                    'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
                    'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone',
                    'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian',
                    'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
                    'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85',
                    'PctSameState85', 'LandArea', 'PopDens', 'PctUsePubTrans', 'LemasPctOfficDrugUn', 'high_crime_per']


    # constraints = [[10, 11], [27, 28]]
    # constrains_no = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    #                  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #                  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    #                  75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    # totalPctDiv is between malePctDiv and femalePctDiv

    # if 28 is 0, 27 must be 0
    constraints_no = []
    constraints = []



