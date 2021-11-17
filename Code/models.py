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
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("../BD/price_classification.csv")
dataset_values = dataset.values

titles = dataset.columns.values
n_clases = dataset["price_range"].nunique()

X = dataset_values[:,:-1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

y = dataset_values[:,-1]
particions = [0.5, 0.8, 0.7]
models = [svm.SVC(probability=True)]
nom_models = ["Support Vector Machines"]

for i,model in enumerate(models):
    print("---- ", nom_models[i], " ----")
    print("Parametres per defecte: " + str(model.get_params()))
    print("")
    for part in particions:
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=part)
        model.fit(x_train, y_train)
        print("Score del model amb ", part, ": ", model.score(x_test, y_test))
    print("")
    for k in range(2,7):
        scores = cross_val_score(model, X, y, cv=k)
        print("Score promig amb k-fold = ", k, " : ", scores.mean())
    print("")
    print("Classes trobades: ", model.classes_)
    print("Atributs trobats: ", model.n_features_in_)
    print("")
    scores2 = LeaveOneOut()
    numero = scores2.get_n_splits(X)
    llista = []
    l = 0
    for train_index, test_index in scores2.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        llista.append(model.score(X_test, y_test))
    print("Score mitja del Leave One Out: ", np.array(llista).mean())
    print("")
    
    """Generar corbes ROC i PR"""
    if nom_models[i] != "Perceptron":
        x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=0.8)
        model.fit(x_t,y_t)
        probs = model.predict_proba(x_v)
        # Compute Precision-Recall and plot curve
        precision = {}
        recall = {}
        average_precision = {}
        plt.figure()
        for j in range(n_clases):
            precision[j], recall[j], _ = precision_recall_curve(y_v == j, probs[:, j])
            average_precision[j] = average_precision_score(y_v == j, probs[:, j])
    
            plt.plot(recall[j], precision[j],
            label='Precision-recall curve of class {0} (area = {1:0.2f})'
                                   ''.format(j, average_precision[j]))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right")
        plt.savefig("../Graficas-A/pr/curva-pr" + str(nom_models[i]) + ".png")
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for j in range(n_clases):
            fpr[j], tpr[j], _ = roc_curve(y_v == j, probs[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])
    
        # Compute micro-average ROC curve and ROC area
        # Plot ROC curve
        plt.figure()
        for j in range(n_clases):
            plt.plot(fpr[j], tpr[j], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(j, roc_auc[j]))
        plt.legend()
        plt.savefig("../Graficas-A/roc/curva-roc" + str(nom_models[i]) + ".png")
    




