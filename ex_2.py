import sys
from random import shuffle
import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# import pandas as pd

"""
def k_fold(full_train_x, full_train_y):
    scores_knn = []
    scores_pa = []
    scores_preceptron = []
    y_train_arr = np.array(full_train_y)
    x_train_arr = np.array(full_train_x)
    kf = KFold(n_splits=5)
    kf.get_n_splits(full_train_x)
    for train_index, test_index in kf.split(full_train_x):
        x_train, x_test, y_train, y_test = x_train_arr[train_index], x_train_arr[test_index], \
                                           y_train_arr[train_index], y_train_arr[test_index]

        y_pred_knn = KNN(x_train, x_test, y_train, 8)
        scores_knn.append(accuracy_score(y_test, y_pred_knn))
        y_pred_precptron = Preceptron(x_train, y_train, x_test)
        scores_preceptron.append(accuracy_score(y_test, y_pred_precptron))
        y_pred_pa = Passive_Aggressive(x_train, y_train, x_test)
        scores_pa.append(accuracy_score(y_test, y_pred_pa))
    print(np.average(scores_knn))
    print(np.average(scores_preceptron))
    print(np.average(scores_pa))
"""


"""
def feature_selection_backward_elimination(x_train, y_train, x_test):
    # The worst features in ascending order - 5, 8, 2, 3
    feature_selection_x_train = []
    feature_selection_x_test = []
    # n = pd.DataFrame(x_train)
    # print(n.var())
    # 5,4,6
    for x in x_train:
        list_x = list(x)
        list_x.remove(list_x[0])
        list_x.remove(list_x[8])
        feature_selection_x_train.append(list_x)
    for x in x_test:
        list_x_test = list(x)
        list_x_test.remove(list_x_test[0])
        list_x_test.remove(list_x_test[8])
        feature_selection_x_test.append(list_x_test)
    return np.array(feature_selection_x_train), y_train, np.array(feature_selection_x_test)
"""


# Function : convert_to_float
# convert each feature from str to float
def convert_to_float(features):
    for i in range(len(features)):
        features[i] = float(features[i])
    return features


# Function: read_file
# read file to list , and  convert
# categorical attribute to a numerical one - 'W' to [1,0], 'R' to [0,1] - 'one hot encoding'
def read_file(file_mame):
    file_list = []
    with open(file_mame) as fp:
        lines = fp.readlines()
        for line in lines:
            features = line.strip().split(',')
            if 'W' in features:
                features.remove(features[features.index('W')])
                features.append('1')
                features.append('0')
            if 'R' in features:
                features.remove(features[features.index('R')])
                features.append('0')
                features.append('1')
            features_as_float = convert_to_float(features)
            file_list.append(features_as_float)
    return np.array(file_list)


# Function : min_max_normalization
# The minimum value of feature (column) gets transformed into a 0,
# The maximum value gets transformed into a 1
# Every other value gets transformed into a decimal between 0 and 1.
def min_max_normalization(training_examples, test_x):
    normal_train = []
    normal_test = []
    trans_train = np.transpose(training_examples)
    trans_test = np.transpose(test_x)
    for i in range(len(trans_train)):
        min_val = np.min(trans_train[i])
        max_val = np.max(trans_train[i])
        if min_val == max_val:
            continue
        v1 = (trans_train[i] - min_val) / (max_val - min_val)
        v2 = (trans_test[i] - min_val) / (max_val - min_val)
        normal_train.append(v1)
        normal_test.append(v2)
    normal_train = np.transpose(normal_train)
    normal_test = np.transpose(normal_test)
    return normal_train, normal_test


# Function: get_label
# return the largest label
# In case of equal - return the labeling with the low value
def get_label(count_label_0, count_label_1, count_label_2):
    if count_label_0 > count_label_2 and count_label_0 > count_label_1:
        return 0
    if count_label_1 > count_label_0 and count_label_1 > count_label_2:
        return 1
    if count_label_2 > count_label_1 and count_label_2 > count_label_0:
        return 2
    if count_label_0 == count_label_1 and count_label_2 < count_label_0:
        return 0
    if count_label_1 == count_label_2 and count_label_0 < count_label_2:
        return 1
    if count_label_2 == count_label_0 and count_label_1 < count_label_0:
        return 0
    # all counters are equal
    return 0


# Function: KNN - K-Nearest Neighbor
# KNN is one of the topmost machine learning algorithms
# calculate distance - use euclidean distance
# find K nearest neighbors
# the class with the most votes is taken as the prediction
def KNN(train_x_normal, test_x_normal, train_y_list, k):
    test_x_len = len(test_x_normal)
    labels = []
    for i in range(test_x_len):
        distance = []
        # find the distance between this test to all the train
        for j in range(len(train_x_normal)):
            # p - tuple of the distance and the index of this x_train
            p = (np.linalg.norm(train_x_normal[j] - test_x_normal[i]), j)
            distance.append(p)
        # sort by distance
        distance.sort(key=lambda x: x[0])
        # take the first k - the nearest neighbor
        top_k = distance[:k]
        count_label_0 = 0
        count_label_1 = 0
        count_label_2 = 0
        for neighbor in top_k:
            index = neighbor[1]
            if train_y_list[index] == 0:
                count_label_0 += 1
            if train_y_list[index] == 1:
                count_label_1 += 1
            if train_y_list[index] == 2:
                count_label_2 += 1
        labels.append(get_label(count_label_0, count_label_1, count_label_2))
    return labels


# Function: Preceptron
# Perceptron distinguishes between different types of samples
# by multiplying the sample vector by the weights vector
def Preceptron(x_train, y_train, x_test):
    x_train_shape = np.array(x_train).shape
    labels = []
    epochs = 150
    eta = 0.001
    # set weight
    # w = np.full((3, x_train_shape[1] + 1), 1 / x_train_shape[1])
    w = np.random.uniform(0, 1, (3, x_train_shape[1] + 1))
    for e in range(epochs):
        for x, y in zip(x_train, y_train):
            y = int(y)
            # add bias
            x = np.insert(np.array(x), 1, 0)
            # prediction
            y_hat = np.argmax(np.dot(w, x.transpose()))
            if y != y_hat:
                w[y, :] += eta * x
                w[y_hat, :] -= eta * x
    for x in x_test:
        # add bias
        x = np.insert(np.array(x), 1, 0)
        y_hat_for_test = np.argmax(np.dot(w, x.transpose()))
        labels.append(int(y_hat_for_test))
    return labels


# Function: calc_loss
# calculate the loss - hing loss function
def calc_loss(w_y, w_y_hat, x):
    real_w = np.dot(w_y, x)
    predict_w = np.dot(w_y_hat, x)
    loss = max(0, 1 - real_w + predict_w)
    return loss


# Function: calc_tau
# calculate the tau in pa algorithm
def calc_tau(w_y, w_y_hat, x):
    loss = calc_loss(w_y, w_y_hat, x)
    x_arr = np.array(x)
    # vec of zero
    if np.all(x_arr == 0):
        return 1
    norm_x = 2 * np.power(np.linalg.norm(x_arr), 2)
    ret = loss / norm_x
    return ret


# Function: Passive_Aggressive
# This is an online algorithm, implemented as offline
# The purpose of the algorithm is to be right about every sample it receives,
# we want to find the w so that the difference
# between it and the w obtained in previous iteration will be minimal
def Passive_Aggressive(x_train, y_train, x_test):
    epochs = 50
    labels = []
    x_train_shape = np.array(x_train).shape
    # init w
    # w = np.random.uniform(0, 1, (3, x_train_shape[1] + 1))
    w = np.full((3, x_train_shape[1] + 1), 1 / x_train_shape[1])
    for e in range(epochs):
        for x, y in zip(x_train, y_train):
            y = int(y)
            x = np.array(x)
            # add bias
            x = np.insert(x, 1, 0)
            # first, remove the w that represent the real class
            fit_w = np.delete(w, y, 0)
            # represents the maximum prediction that is not y- the true prediction
            y_hat = int(np.argmax(np.dot(fit_w, x)))
            if y <= y_hat:
                # we would like this prediction to be larger in 1 of all the examples
                y_hat += 1
            tau = calc_tau(w[y, :], w[y_hat, :], x)
            w[y, :] += tau * x
            w[y_hat, :] -= tau * x
    for x in x_test:
        x = np.array(x)
        x = np.insert(x, 1, 0)
        y_hat_for_test = np.argmax(np.dot(w, x.transpose()))
        labels.append(int(y_hat_for_test))
    return labels


# Function:shuffle_data
# shuffle the data
def shuffle_data(x_train, train_y):
    c = list(zip(x_train, train_y))
    shuffle(c)
    return x_train, train_y


def main():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    training_examples_list = read_file(train_x)
    train_y = np.loadtxt(train_y)
    test_list = read_file(test_x)
    # normalize the examples, and the teat
    x_train_normal, x_test_normal = min_max_normalization(training_examples_list, test_list)
    # shuffle data
    x_train_normal, train_y = shuffle_data(x_train_normal, train_y)
    # k_fold(x_train_normal, train_y)
    # x_train_fit, y_train_fit, x_test_fit = feature_selection_backward_elimination(x_train_normal, train_y, x_test_normal)
    knn_labels = KNN(x_train_normal, x_test_normal, train_y, 8)
    preceptron_labels = Preceptron(x_train_normal, train_y, x_test_normal)
    pa_labels = Passive_Aggressive(x_train_normal, train_y, x_test_normal)
    """
    # for statistic
    for i in range(10):
        x_train_sh, train_y_sh = shuffle_data(x_train_normal, train_y)
        k_fold(x_train_sh, train_y_sh)
    """
    for i in range(len(preceptron_labels)):
        print(f"knn: {knn_labels[i]}, perceptron: {preceptron_labels[i]}, pa: {pa_labels[i]} ")


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
