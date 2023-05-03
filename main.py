
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


files = ["webscraping/Billboard Hot 100.csv", "webscraping/BUTTER.csv", "webscraping/Electric Lady Studios.csv", "webscraping/Fresh Finds.csv", "webscraping/Best of the Decade For You.csv"]
frames = []
for f in files:
    frames.append(pd.read_csv(f, encoding='cp1252'))
df = pd.concat(frames)
print(df)
#normalized
X_train_initial = df.to_numpy()[:,1:-1]

X_train_normed_initial = pd.DataFrame()

for j in range(X_train_initial.shape[1]):
    #gets entire column
    Xj = X_train_initial[:,j]
    sigma = np.std(Xj)
    mu = np.mean(Xj)
    normed = ( Xj - mu ) / sigma
    #assigns column
    X_train_normed_initial[j] = normed
    
print(X_train_normed_initial)

X_train_normed_initial = X_train_normed_initial.to_numpy()
y_train = df.to_numpy()[:,df.shape[1]-1]
def dist(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i]-x2[i])**2
    return np.sqrt(distance)

def nearestNeighbors(k, X_data, y_data, toPredict):
    #TODO: change toPredict to take in an array of features (currently takes in an index of data array)
    #   also make it so we keep either the indices or something of the closest songs to be able to then fetch the title/artist
    #   OR we can just store the titles/artists as we go.
    closest = {10**20: None}
    for i in range(len(X_data)):
        currMax = max(closest.keys())
        distance = dist(X_data[i], X_data[toPredict])
        if distance < currMax:
            # HERE WE CAN JUST ADD SONG, TITLE AFTER Y_DATA TO HAVE DICTIONARY STORE A TRIPLE OF (POPULARITY, SONG, TITLE)
            closest[distance] = y_data[i]
        if len(closest) > k:
            closest.pop(currMax)
    total = 0
    for key in closest:
        total += closest.get(key)
    return(total/len(closest.keys()))

print(nearestNeighbors(5, X_train_normed_initial, y_train, 0))