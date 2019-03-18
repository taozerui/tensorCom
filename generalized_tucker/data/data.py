import pickle
import csv
import ipdb
import sparse
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
        rating.append(int(line[2]))
        food.append(int(line[3]))
        service.append(int(line[4]))

userSet = set(user)
placeSet = set(place)
rating_final = np.zeros((len(userSet), len(placeSet), 3))
axes1 = []
axes2 = []
axes3 = []
data = []
for i in range(len(user)):
    user_i = user[i]
    place_i = place[i]
    rowIndex = list(userSet).index(user_i)
    colIndex = list(placeSet).index(place_i)
    '''
    rating_final[rowIndex, colIndex, 0] = rating[i]
    rating_final[rowIndex, colIndex, 1] = food[i]
    rating_final[rowIndex, colIndex, 2] = service[i]
    '''
    for s in range(3):
        axes1.append(rowIndex)
        axes2.append(colIndex)
        axes3.append(s)
        if s == 0:
            data.append(rating[i])
        elif s == 1:
            data.append(food[i])
        elif s == 2:
            data.append(service[i])

coords = np.vstack((axes1, axes2, axes3))
data = np.array(data)
rating_final = sparse.COO(coords, data, shape=(len(userSet), len(placeSet), 3))

#with open('rating_final.pkl', 'wb') as f:
#    pickle.dump(rating_final, f)
