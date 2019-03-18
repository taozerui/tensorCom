import pickle
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, coo_matrix
from numpy.random import RandomState
from completion import NMF, LogisticPCA

with open('./../data/ml-1m/ratings.pkl', 'rb') as f:
    rating = pickle.load(f)

def binary(rating):
    for k, v in enumerate(rating.data):
        if v >= 4:
            rating.data[k] = 1
        elif v < 4:
            rating.data[k] = 0

    return rating

def split(rating, ratio=0.85, seed=123):
    ratingCache = rating.copy()
    ratingCache.data = ratingCache.data + 1
    ratingNum = len(rating.data)
    fullIndex = np.r_[0:ratingNum]
    trainNum = round(ratio * ratingNum)
    rdm = RandomState(seed)
    trainIndex = np.sort(rdm.choice(ratingNum, trainNum, replace=False))
    testIndex = np.sort(np.setdiff1d(fullIndex, trainIndex))
    rowIndex = ratingCache.nonzero()[0]
    colIndex = ratingCache.nonzero()[1]
    trainRating = csc_matrix((rating.data[trainIndex], (rowIndex[trainIndex], colIndex[trainIndex])),
                             shape=rating.shape)
    testRating = csc_matrix((rating.data[testIndex], (rowIndex[testIndex], colIndex[testIndex])),
                             shape=rating.shape)

    return trainRating, testRating

def rmse(testRating, estimateRating):
    testRating = coo_matrix(testRating)
    rowIndex = testRating.row
    colIndex = testRating.col
    testRating = csc_matrix(testRating)
    num = len(colIndex)
    rmse = 0
    error = 0
    for i in range(num):
        a = rowIndex[i]
        b = colIndex[i]
        rmse += (testRating[a, b] - estimateRating[a, b]) ** 2
        if estimateRating[a, b] > 0:
            label = 1
        elif estimateRating[a, b] <= 0:
            label = 0
        else:
            raise ValueError(f'Invalide value for ({a}, {b}): {estimateRating[a, b]}!')

        if label != testRating[a, b]:
            error += 1

    return rmse / num, error / num

def nmf_main():
    rating = binary(rating)
    trainRating, testRating = split(rating)

    features = np.r_[2]
    errorTot = []
    misclassTot = []
    for i in features:
        model = NMF(i)
        model.fit(trainRating, alpha=0.01, beta=0.01,
                  learning_rate=0.0001, max_iter=5e3,
                  tol=1e-1, print_loss=True)

        error, misclass = rmse(testRating, model.Xhat)
        print(f'Model latent feature {len(model)}, the misclassification error is {misclass:.3f}.')
        errorTot.append(error)
        misclassTot.append(misclass)

    return errorTot, misclassTot

rating = binary(rating)
trainRating, testRating = split(rating)

features = np.r_[2]
errorTot = []
misclassTot = []
for i in features:
    model = LogisticPCA(i)
    model.fit(trainRating, max_iter=100,
              tol=1e-3, print_loss=True)

    error, misclass = rmse(testRating, model.Xhat)
    print(f'Model latent feature {len(model)}, the misclassification error is {misclass:.3f}.')
    errorTot.append(error)
    misclassTot.append(misclass)
