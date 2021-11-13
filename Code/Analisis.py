import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../BD/price_classification.csv")
dataset_values = dataset.values

titles = dataset.columns.values
n_clases = dataset["price_range"].nunique()

'''Creacion de graficas'''

'''Dispersion'''
plt.figure()
sns.set()
for i in range(dataset.shape[1] - 1):
    sns.scatterplot(data=dataset, x=titles[i], y=titles[-1], palette='pastel')
    plt.savefig("../Graficas/disp/Caracteristica" + str(i+1) + ".png")
    plt.clf()
    

'''De barras'''

sns.histplot(data=dataset, x="price_range", palette='pastel')
plt.savefig("../Graficas/hist/histograma_preus.png")
plt.clf()


for i in range(dataset.shape[1] - 1):
    sns.histplot(data=dataset, x=titles[i], hue="price_range", multiple="dodge", shrink=.8, palette='pastel')
    plt.savefig("../Graficas/hist/Caracteristica" + str(i+1) + ".png")
    plt.clf()

'''Mapa de calor'''
fig, ax = plt.subplots(figsize=(20,20))
cmap = sns.color_palette('pastel', as_cmap=True)
sns.heatmap(dataset.corr(), ax=ax, cmap=cmap, vmin=0, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig("../Graficas/heatmap/mapa-de-calor.png")
plt.clf()