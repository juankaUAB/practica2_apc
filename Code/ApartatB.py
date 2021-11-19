import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.datasets import load_wine

dataset = pd.read_csv("../BD/price_classification.csv")
dataset_values = dataset.values

titles = dataset.columns.values
n_clases = dataset["price_range"].nunique()

'''Creacion de Graficas de la base de Datos asignada a nuestro subgrupo'''

'''Dispersion'''
plt.figure()
sns.set()
for i in range(dataset.shape[1] - 1):
    sns.scatterplot(data=dataset, x=titles[i], y=titles[-1], palette='pastel', hue="price_range")
    plt.savefig("../Graficas-A/disp/Caracteristica" + str(i+1) + ".png")
    plt.clf()
    

'''De barras'''

sns.histplot(data=dataset, x="price_range", palette='pastel')
plt.savefig("../Graficas-A/hist/histograma_preus.png")
plt.clf()


for i in range(dataset.shape[1] - 1):
    sns.histplot(data=dataset, x=titles[i], hue="price_range", multiple="dodge", shrink=.8, palette='pastel')
    plt.savefig("../Graficas-A/hist/Caracteristica" + str(i+1) + ".png")
    plt.clf()

'''Mapa de calor'''
fig, ax = plt.subplots(figsize=(20,20))
cmap = sns.color_palette('pastel', as_cmap=True)
sns.heatmap(dataset.corr(), ax=ax, cmap=cmap, vmin=0, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig("../Graficas-A/heatmap/mapa-de-calor.png")
plt.clf()


'''pruebas con un dataset de sklearn (wine)'''
wine = load_wine()
X = wine.data

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

y = wine.target
nom_atributs = wine.feature_names
nom_classes = wine.target_names.reshape(-1)
fig, sub = plt.subplots(1, 2, figsize=(16,6))
sub[0].scatter(X[:,0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
sub[1].scatter(X[:,1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')


''' comparacion de 4 modelos diferentes (KNN, Decision Tree, Logistic Regression y SVM)'''
particions = [0.5, 0.8, 0.7]

for part in particions:
    x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=part)
    
    #Creem el regresor logístic
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)

    # l'entrenem
    logireg.fit(x_t, y_t)

    print ("Correct classification Logistic ", part, "% of the data: ", logireg.score(x_v, y_v))
    
    #Creem el SVM
    svc = svm.SVC(C=10.0, kernel='linear', gamma=0.6, probability=True)

    # l'entrenem 
    svc.fit(x_t, y_t)
    probs = svc.predict_proba(x_v)
    print ("Correct classification SVM      ", part, "% of the data: ", svc.score(x_v, y_v))
    
    #Creem el KNN
    knn = KNeighborsClassifier()
    
    #l'entrenem
    knn.fit(x_t,y_t)
    print("Correct classification KNN ", part, "% of the data: ", knn.score(x_v,y_v))
    
    #Creem el DecisionTree
    tree = DecisionTreeClassifier()
    
    #l'entrenem
    tree.fit(x_t,y_t)
    print("Correct classification Decision Tree ", part, "% of the data: ", tree.score(x_v,y_v))
    
    print("")
    

n_clases = len(nom_classes)

''' creacion curvas roc y pr'''
models = [LogisticRegression(), svm.SVC(probability=True), KNeighborsClassifier(), DecisionTreeClassifier()]
nom_models = ["LogisticRegression","SVM","KNN","DecisionTree"]
for o,model in enumerate(models):
    x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=0.8)
    model.fit(x_t,y_t)
    probs = model.predict_proba(x_v)
    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_clases):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])

        plt.plot(recall[i], precision[i],
        label='Precision-recall curve of class {0} (area = {1:0.2f})'
                               ''.format(i, average_precision[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")
    plt.savefig("../Graficas-B/pr/curva-pr" + str(nom_models[o]) + ".png")

        

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_clases):
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    rnd_fpr, rnd_tpr, _ = roc_curve(y_v>0, np.zeros(y_v.size))
    plt.plot(rnd_fpr, rnd_tpr, linestyle='--', label='Sense capacitat predictiva')
    for i in range(n_clases):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.legend()
    plt.savefig("../Graficas-B/roc/curva-roc" + str(nom_models[o]) + ".png")


'''visualizar la clasificacion'''
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def show_C_effect(X, y, C=1.0, gamma=0.7, degree=3):


    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    #C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(nom_atributs[0])
        ax.set_ylabel(nom_atributs[1])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.savefig("../Graficas-B/c-effect.png")
    
show_C_effect(X[:,:2], y, C=0.1)
