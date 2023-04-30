import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("webscraping/Billboard Hot 100.csv");
df2 = pd.read_csv("webscraping/BUTTER.csv");
df = pd.concat([df1, df2])



#normalized
X_train_initial = df.to_numpy()[:,0:-1]

print(X_train_initial)
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