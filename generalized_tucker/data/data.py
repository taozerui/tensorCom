import pickle
import csv
import ipdb
import numpy as np
from itertools import islice

with open('./rating_final.csv', 'r') as f:
    dataloader = csv.reader(f)
    user = []
    place = []
    rating = []
    food = []
    service = []
    for line in islice(dataloader, 1, None):
        user.append(int(line[0][1:]))
        place.append(int(line[1]))
        rating.append(int(line[2]) + 1)
        food.append(int(line[3]) + 1)
        service.append(int(line[4]) + 1)

userSet = set(user)
placeSet = set(place)
rating_final = np.zeros((len(userSet), len(placeSet), 3))
for i in range(len(user)):
    user_i = user[i]
    place_i = place[i]
    rowIndex = list(userSet).index(user_i)
    colIndex = list(placeSet).index(place_i)
    rating_final[rowIndex, colIndex, 0] = rating[i]
    rating_final[rowIndex, colIndex, 1] = food[i]
    rating_final[rowIndex, colIndex, 2] = service[i]

with open('rating_final.pkl', 'wb') as f:
    pickle.dump(rating_final, f)
