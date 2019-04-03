import pickle

import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, coo_matrix
from numpy.random import RandomState

from tensorCom.matrix import NMF_Completion, LogisticPCA_Completion

def binary(rating):
    for k, v in enumerate(rating.data):
        if v >= 4:
            rating.data[k] = 1
        elif v < 4:
            rating.data[k] = 0

    return rating

def split(rating, ratio=0.8, seed=123):
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
        if estimateRating[a, b] > 0.5:
            label = 1
        elif estimateRating[a, b] <= 0.5:
            label = 0
        else:
            raise ValueError(f'Invalide value for ({a}, {b}): {estimateRating[a, b]}!')

        if label != testRating[a, b]:
            error += 1

    return rmse / num, error / num

def main():
    with open('./data/ml-100k/ratings.pkl', 'rb') as f:
        rating = pickle.load(f)
    rating = binary(rating)
    trainRating, testRating = split(rating)
    #trainRating = rating[0:100, 0:100]
    #testRating = rating

    features = np.r_[2, 3, 4, 5]
    errorTot = []
    misclassTot = []
    for i in features:
        model1 = LogisticPCA_Completion(i)
        model1.fit(trainRating, max_iter=1e4, learning_rate=0.01,
                  tol=1e-2, print_loss=True)
        model1.plotIter()
        model2 = NMF_Completion(i)
        model2.fit(trainRating, alpha=0.01, beta=0.01,
                  learning_rate=0.0001, max_iter=5,
                  tol=1e-1, print_loss=True)

        errorPCA, misclassPCA = rmse(testRating, model1.prob)
        errorNMF, misclassNMF = rmse(testRating, model2.Xhat)
        print(f'Model latent feature {len(model1)}')
        print(f'The misclassification error is | NMF: {misclassNMF:.3f} | Logistic PCA: {misclassPCA:.3f}.')

if __name__ == '__main__':
    main()
