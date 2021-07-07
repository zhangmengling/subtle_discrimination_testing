import numpy as np
import sys
sys.path.append("../")

def compas_data():
    """
    Prepare the data of dataset Bank Marketing
    :return: X, Y, input shape and number of classes
    """
    # print("-->bank_data")
    X = []
    Y = []
    i = 0
    with open("../dataset/compas_data_binary.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = list(map(int, line1[:-1]))
            L = list(map(int, line1[:-1]))    # NO binary data: [14:-1] (10 attributes) ; binary data: [18:-1] (29 attributes)
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 29)
    nb_classes = 2

    # print(X, Y, input_shape, nb_classes)
    return X, Y, input_shape, nb_classes

# X, Y, input_shape, nb_classes = compas_data()
# print("-->x", X[0], X.shape)


