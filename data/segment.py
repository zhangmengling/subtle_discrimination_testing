import numpy as np
import sys
sys.path.append("../")

def segment_data(*self):
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("../dataset/all_train.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            print(line1[:-1])
            L = list(map(float, line1[:-1]))
            X.append(L)
            if str(line1[-1]) == "1":
                Y.append([1, 0, 0])
            elif str(line1[-1]) == "2":
                Y.append([0, 1, 0])
            else:
                Y.append([0, 0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    # print(X)
    print(Y)

    input_shape = (None, 126)
    nb_classes = 3

    return X, Y, input_shape, nb_classes