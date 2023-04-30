import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("webscraping/Billboard Hot 100.csv");
df2 = pd.read_csv("webscraping/BUTTER.csv");
df = pd.concat([df1, df2])

print(df);