import pickle
import ipdb
import sparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from completion import Tucker
from numpy.random import RandomState

with open('../data/rating_final.pkl', 'rb') as f:
    data = pickle.load(f)

def binary(rating):
    for k, v in enumerate(rating.data):
        if v == 2:
            rating.data[k] = 1
        elif v < 2:
            rating.data[k] = 0

    return rating

def index(x):
    out = []
    for i in x:
        for j in range(3):
            out.append(i * 3 + j)

    return out

def split(rating, ratio=0.85, seed=123):
    ratingNum = int(len(rating.data) / 3)
    fullIndex = np.r_[0:ratingNum]
    trainNum = round(ratio * ratingNum)
    rdm = RandomState(seed)
    trainIndex = np.sort(rdm.choice(ratingNum, trainNum, replace=False))
    trainIndex = index(trainIndex)
    testIndex = np.sort(np.setdiff1d(fullIndex, trainIndex))
    testNum = len(testIndex)
    testIndex = index(testIndex)
    rowIndex = rating.coords[0]
    colIndex = rating.coords[1]
    trainCoords = np.vstack((rowIndex[trainIndex], colIndex[trainIndex], [0, 1, 2] * trainNum))
    testCoords = np.vstack((rowIndex[testIndex], colIndex[testIndex], [0, 1, 2] * testNum))
    trainRating = sparse.COO(trainCoords, rating.data[trainIndex], shape=rating.shape)
    testRating = sparse.COO(testCoords, rating.data[testIndex], shape=rating.shape)

    return trainRating, testRating

def rmse(testRating, estimateRating):
    rowIndex = testRating.coords[0]
    colIndex = testRating.coords[1]
    num = len(colIndex)
    rmse = 0
    error = 0
    for i in range(num):
        a = rowIndex[i]
        b = colIndex[i]
        for j in range(3):
            rmse += (testRating[a, b, j] - estimateRating[a, b, j]) ** 2
            if estimateRating[a, b, j] > 0.5:
                label = 1
            elif estimateRating[a, b, j] <= 0.5:
                label = 0
            else:
                raise ValueError(f'Invalide value for ({a}, {b}, {j}): {estimateRating[a, b, j]}!')

            if label != testRating[a, b, j]:
                error += 1

    num = num * 3
    return rmse / num, error / num

ranks = np.r_[1:5]
data = binary(data)
trainRating, testRating = split(data)
for r in ranks:
    model = Tucker((r, r, 2))
    model.fit(trainRating, learning_rate=0.0001,
              max_iter=5e4, print_loss=False)
    squareE, error = rmse(testRating, model.Xhat)
    print(f'For Tucker ranks ({r}, {r}, 2), the mis-classification error is {error:.3f}.')
