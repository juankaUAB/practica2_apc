import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../BD/price_classification.csv")
dataset_values = dataset.values

title_x = ["battery_power","bluetooth","clock_speed","dual_sim","pixels_fc","fourg"
         ,"int_memory","m_depth","m_weight","n_cores","pixels_pc","px_height","px_width",
         "ram","sc_height","sc_width","talk_time","threeg","touch_screen","wifi"]
title_y = "Price Range"

X = dataset_values[:,:20]
y = dataset_values[:,-1]

n_clases = dataset["price_range"].nunique()

plt.figure()
for i in range(X.shape[1]):
    plt.xlabel(title_x[i])
    plt.ylabel(title_y)
    plt.scatter(X[:,i], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.savefig("../Graficas/disp/Caracteristica" + str(i+1) + ".png")
    plt.clf()
