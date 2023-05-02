
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


files = ["webscraping\Billboard Hot 100.csv", "webscraping\BUTTER.csv", "webscraping\Electric Lady Studios.csv", "webscraping\Fresh Finds.csv", "webscraping\Best of the Decade For You.csv"]
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



