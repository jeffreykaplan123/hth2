import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import urllib.request
import json

endpoint = 'https://maps.googleapis.com/maps/api/directions/json?'
api_key = 'AIzaSyD6RAWPhZujfg7aFJkyhQO508mZncNyTk4'
origin = input('Start: ').replace(' ', '+')
destination = input('Destination: ').replace(' ', '+')
nav_req = 'origin={}&destination={}&key={}'.format(origin, destination, api_key)
request = endpoint + nav_req
response = urllib.request.urlopen(request).read()
direction = json.loads(response)
routes = direction['routes']
month = input('Month: ')
day = input('day: ')
hour = input('hour: ')
weekday = input('day: ')
vis = input('visibility: ')
Light = input('Light: ')
rdcond = input('RoadConditions: ')
test = np.empty([1, 9])
for i in range(len(routes[0]['legs'][0]['steps'])):
    test = np.concatenate((test, np.array([[month, day, hour, weekday, routes[0]['legs'][0]['steps'][i]['end_location']['lng'], routes[0]['legs'][0]['steps'][i]['end_location']['lat'], vis, Light, rdcond]])), axis=0)

test = test[1:, :]
dataset = pd.read_csv('KSI_CLEAN.csv')
dataset.insert(0, 'collision', 1)

X = dataset.iloc[:, [3, 4, 5, 7, 8, 9, 23, 24, 25]].values
y = dataset.iloc[:, 0].values
for i in range(12556):
    X = np.concatenate((X,np.array([[random.randint(1,13), random.randint(1,31), random.randint(7,16), random.randint(1,8), np.random.choice(X[:, 4]), np.random.choice(X[:, 5]), np.random.choice(X[:, 6]), np.random.choice(X[:, 7]), np.random.choice(X[:, 8])]])), axis=0)
    y = np.concatenate((y, [0]), axis=0)

columns_to_encode = [6, 7, 8]
columns_to_scale = [4, 5]
ohe = OneHotEncoder(sparse=False)
encoded_columns = ohe.fit_transform(X[:,columns_to_encode])
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(X[:, columns_to_scale])
X = X[:, [0,1,2,3]]
X = np.concatenate([X, encoded_columns], axis=1)
X = np.concatenate([X, scaled_columns], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
encoded_columns = ohe.transform(test[:, columns_to_encode])
scaled_columns = scaler.transform(test[:, columns_to_scale])
test = test[:, [0, 1, 2, 3]]
test = np.concatenate([test, encoded_columns], axis=1)
test = np.concatenate([test, scaled_columns], axis=1)
count0 = 0
count1 = 0

pred = classifier.predict(test)
for i in range(len(pred)):
    if pred[i] == 1:
        count1 += 1
    else:
        count0 += 1
counttotal = (count0/(count1+count0))*10
print('safety score (out of 10):  ' + str(counttotal))
