import csv
import pickle

import numpy as np
import scipy.sparse as ssparse
from scipy.sparse import csc_matrix
import ipdb

with open('u.data', 'r') as f:
    datareader = csv.reader(f)
    users = []
    movies = []
    ratings = []
    for row in datareader:
        splitRow = row[0].split()
        userID = int(splitRow[0]) - 1
        movieID = int(splitRow[1]) - 1
        rating = int(splitRow[2])
        users.append(userID)
        movies.append(movieID)
        ratings.append(rating)
userNum = max(users) + 1
movieNum = max(movies) + 1
ratingMatrix = csc_matrix((ratings, (users, movies)), shape=(userNum, movieNum))

with open('ratings.pkl', 'wb') as f:
    pickle.dump(ratingMatrix, f)
