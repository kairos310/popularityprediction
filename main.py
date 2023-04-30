import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


files = ["webscraping/Billboard Hot 100.csv", "webscraping/BUTTER.csv", "webscraping\Electric Lady Studios.csv", "webscraping\Fresh Finds.csv", "webscraping\Best of the Decade For You.csv"]
frames = []
for f in files:
    frames.append(pd.read_csv(f))
df = pd.concat(frames)

#normalized
X_train_initial = df.to_numpy()[:,0:-1]

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


