import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score,  precision_score, recall_score, roc_curve,roc_auc_score, auc
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_curve


dataset = pd.read_csv("../BD/price_classification.csv")
dataset_values = dataset.values

titles = dataset.columns.values
n_clases = dataset["price_range"].nunique()


X = dataset_values[:,:-1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

y = dataset_values[:,-1]


#Calculem l'score del model logistic amb diferents particions de train i test
particions = [0.5, 0.8, 0.7]

for part in particions:
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=part)
    modelLogistic = LogisticRegression()
    modelLogistic.fit(x_train, y_train)
    predictions = modelLogistic.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='macro')
    rec = recall_score(y_test, predictions, average='macro')
    conf_mat = confusion_matrix(y_test, predictions).T
    print(f'Accuracy:{acc}')
    print(f'Precision:{prec}')
    print(f'Recall:{rec}')
    plot_confusion_matrix(modelLogistic, x_test, y_test, normalize='true', cmap='jet')
    print(modelLogistic.score(x_test, y_test))
  
    
#Calculem l'score de fer el cross validation  
scores = cross_val_score(modelLogistic, X, y, cv=5)
print(scores.mean())


#Fem el leave one out amb el model logistic i calculem la mitjana del score
scores2 = LeaveOneOut()
numero = scores2.get_n_splits(X)
llista = []
for train_index, test_index in scores2.split(X):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     modelProva = LogisticRegression()
     modelProva.fit(X_train, y_train)
     llista.append(modelProva.score(X_test, y_test))
    
print(np.array(llista).mean())






