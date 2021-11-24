from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_curve, average_precision_score, \
    roc_auc_score, roc_curve, auc, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# from skopt import BayesSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
pd.set_option("display.max_columns", None)


def analyseData(database):
    print("Informacio de la base de dades:")
    print(database.info())
    print("---------\ncapçalera:")
    print(database.head())
    sns.heatmap(database.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()
    print("---------\npercentatges de nuls en cada variable:")
    missing = pd.DataFrame(database.isnull().sum(), columns=['No. of missing values'])
    missing['% missing_values'] = (missing / len(database)).round(2) * 100
    print(missing)
    # Fer proves amb la columna resultat

def ModifyDatabase(database):
    databaseChangeRain = database.copy()
    databaseChangeRain['RainTomorrow'] = [1 if i == 'Yes' else 0 for i in databaseChangeRain['RainTomorrow']]
    # print("valors duplicats: ", database.duplicated().sum())
    # Fer proves amb la columna resultat
    # print(databaseChangeRain['RainTomorrow'].value_counts())
    # sns.heatmap(database.corr(), annot=True, linewidths=.5, cmap='rocket');
    # plt.show()
    cols_to_drop = ['Date']
    databaseChangeRain.drop(columns=cols_to_drop, inplace=True)
    x = databaseChangeRain.drop(['RainTomorrow'], axis=1)
    y = databaseChangeRain['RainTomorrow']
    # plotVariable=['Rainfall', 'Evaporation', 'WindSpeed9am','WindSpeed3pm','MinTemp']
    # creacio d'un plot per veure la distribucio de les dades
    # plotVariablesBox(databaseChangeRain, plotVariable)
    return x, y


def plotVariablesBox(database, plotVariable):
    fig, axes = plt.subplots(len(plotVariable), 1, figsize=(10, 10), sharey=False)
    fig.suptitle('Distribution')
    for index, variable in enumerate(plotVariable):
        sns.boxplot(x=variable, data=database, palette='Set2', ax=axes[index])
        axes[index].set_title("")
    # Rainfall

    plt.tight_layout()
    plt.show()

def fixMissingValues(df):
    # print((df.isnull().sum() / len(df)) * 100)
    # els atributs continus s'omplen amb la mitjana
    df['MinTemp'] = df['MinTemp'].fillna(df['MinTemp'].mean())
    df['MaxTemp'] = df['MinTemp'].fillna(df['MaxTemp'].mean())
    df['Rainfall'] = df['Rainfall'].fillna(df['Rainfall'].mean())
    df['Evaporation'] = df['Evaporation'].fillna(df['Evaporation'].mean())
    df['Sunshine'] = df['Sunshine'].fillna(df['Sunshine'].mean())
    df['WindGustSpeed'] = df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
    df['WindSpeed9am'] = df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
    df['WindSpeed3pm'] = df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
    df['Humidity9am'] = df['Humidity9am'].fillna(df['Humidity9am'].mean())
    df['Humidity3pm'] = df['Humidity3pm'].fillna(df['Humidity3pm'].mean())
    df['Pressure9am'] = df['Pressure9am'].fillna(df['Pressure9am'].mean())
    df['Pressure3pm'] = df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
    df['Cloud9am'] = df['Cloud9am'].fillna(df['Cloud9am'].mean())
    df['Cloud3pm'] = df['Cloud3pm'].fillna(df['Cloud3pm'].mean())
    df['Temp9am'] = df['Temp9am'].fillna(df['Temp9am'].mean())
    df['Temp3pm'] = df['Temp3pm'].fillna(df['Temp3pm'].mean())

    # aquests s'omplen agafant la moda de l'atribut pel fet que no pots treure una mitjana de dades discretes
    df['RainToday'] = df['RainToday'].fillna(df['RainToday'].mode()[0])
    df['RainTomorrow'] = df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
    df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
    df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
    df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
    # print((df.isnull().sum() / len(df)) * 100)

    return df

def fixMissingValuesMode(df):
    # hacemos la moda en los Nan a toda la base de datos
    # toda la base de datos ponemos moda
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    # print("Valores nulos ?(con moda):")
    xModedf = df.copy()
    for i in variables:
        xModedf[i].fillna(xModedf[i].mode()[0], inplace=True)
    # print(i, ": ", xModedf[i].isna().sum())
    return xModedf

def fixMissingValuesMedian(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xMediandf = df.copy()
    # toda la base de datos ponemos mediana menos a los de tipo objeto a esos no se les puede calcular la mediana
    # print("Valores nulos ?(con mediana):")
    for i in variables:
        if (np.dtype(xMediandf[i]) == 'object'):
            xMediandf[i].fillna(xMediandf[i].mode()[0], inplace=True)
        else:
            xMediandf[i].fillna(xMediandf[i].median(), inplace=True)
    # print(i, ": ", xMediandf[i].isna().sum())
    return xMediandf

def fixMissingValuesMean(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xMeandf = df.copy()
    # print("Valores nulos ?(con media):")
    # toda la base de datos ponemos media menos a los de tipo objeto a esos no se les puede calcular la mediana
    for i in variables:
        if (np.dtype(xMeandf[i]) == 'object'):
            xMeandf[i].fillna(xMeandf[i].mode()[0], inplace=True)
        else:
            xMeandf[i].fillna(xMeandf[i].mean(), inplace=True)
    # print(i, ": ", xMeandf[i].isna().sum())
    return xMeandf

def cleanAndEnchanceData(df):
    # esborrant dia (dada identificadora, no vàlides pels models)
    df = df.drop(columns=['Date'])

    # posant un valor numèric a les dades categòriques per poder-les tractar aritmèticament
    encoder = preprocessing.LabelEncoder()
    df['Location'] = encoder.fit_transform(df['Location'])
    df['WindDir9am'] = encoder.fit_transform(df['WindDir9am'])
    df['WindDir3pm'] = encoder.fit_transform(df['WindDir3pm'])
    df['WindGustDir'] = encoder.fit_transform(df['WindGustDir'])

    # les de dalt dona igual el valor numèric que agafin, aquestes millor fer-les manualment
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

    return df

def EnchanceData(x):
    # esborrant dia (dada identificadora, no vàlides pels models)
    variablesCategoric = list(x.select_dtypes(include=['object']).columns)
    # transformer = ColumnTransformer(
    #                         transformers=[('notCategoric',
    #                                         OrdinalEncoder(sparse='False', drop='first'),
    #                                        variablesCategoric)],
    #                         remainder='passthrough')
    transformer = ColumnTransformer(
        transformers=[('notCategoric',
                       OneHotEncoder(sparse='False', drop='first'),
                       variablesCategoric)],
        remainder='passthrough')
    return transformer.fit_transform(x)


def standarise2(df, with_mean=False):
    # scaler = preprocessing.MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df
    # df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)


def standarise(df):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    return df

def balanceData(X, y):
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    # bàsicament pel fet que un 80% de les Y són 0 i un 20% són 1
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    return X, y

def removeOutliers(df):
    return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

def NormalitzeData(df):
    skew_limit = 0.75
    skew_value = df[df.columns].skew()
    skew_cols = skew_value[abs(skew_value) > skew_limit]
    cols = skew_cols.index
    return cols

def deleteHighlyCorrelatedAttributes(df):
    return df.drop(['Temp3pm','Temp9am','Humidity9am'],axis=1)


def logisticRegression(X_test, X_train, y_test, y_train, proba=False):
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    logireg.fit(X_train, y_train.values.ravel())  # https://www.geeksforgeeks.org/python-pandas-series-ravel/
    if proba:
        y_pred = logireg.predict_proba(X_test)
        # return y_pred

    y_pred = logireg.predict(X_test)
    print("\nLogistic")
    printMetrics(y_pred, y_test)


def svcLinear(X_test, X_train, y_test, y_train):
    # https://scikit-learn.org/stable/modules/svm.html#complexity
    svc = svm.LinearSVC(C=2,max_iter=500, penalty="l2",random_state=0, tol=1e-4)
    svc.fit(X_train, y_train.values.ravel())
    y_pred = svc.predict(X_test)
    print("\nSVC Linear")
    printMetrics(y_pred, y_test)



def svc(X_test, X_train, y_test, y_train, proba=False, kernels=['rbf']):
    for kernel in kernels:
        ## Per les proves
        # svc = svm.SVC(C=2, kernel=kernel, probability=False, random_state=0, tol=0.0001, max_iter=500)
        # svc.fit(X_train, y_train)
        # # if proba:
        # #     y_pred = svc.predict_proba(X_test.head(100))
        # #     return y_pred
        # y_pred = svc.predict(X_test)
        # print("\nSVC")
        # # print("Accuracy: ", accuracy_score(y_test, y_pred))
        # # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 score: ", f1_score(y_test, y_pred))
        svc = svm.SVC(C=100, kernel=kernel, probability=True, random_state=0)
        svc.fit(X_train.head(1000), y_train.head(1000).values.ravel())
        if proba:
            y_pred = svc.predict_proba(X_test.head(100))
            return y_pred
        y_pred = svc.predict(X_test.head(100))
        print("\nSVC")
        printMetrics(y_pred, y_test.head(100))


def xgbc(X_test, X_train, y_test, y_train, proba=False):
    xgbc = XGBClassifier(objective='binary:logistic', use_label_encoder =False, n_estimators=5,gamma=0.5, random_state=0)
    xgbc.fit(X_train,y_train.values.ravel())
    if proba:
        y_pred = xgbc.predict_proba(X_test)
        return y_pred

    y_pred = xgbc.predict(X_test)
    print("\nXGBC")
    printMetrics(y_pred, y_test)



def rfc(X_test, X_train, y_test, y_train, proba=False):
    clf = RandomForestClassifier(max_leaf_nodes=15,n_estimators=100, ccp_alpha=0.0,bootstrap=True, random_state=0)
    clf.fit(X_train,y_train)
    if proba:
        y_pred = clf.predict_proba(X_test)
        return y_pred

    y_pred = clf.predict(X_test)
    print("\nRandom Forest")
    printMetrics(y_pred, y_test)


def knn(X_test, X_train, y_test, y_train, neighbors=2, proba=False):
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights="uniform", p=2)
    knn.fit(X_train, y_train)
    if proba:
        y_pred = knn.predict_proba(X_test)
        return y_pred

    y_pred = knn.predict(X_test)
    # print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))

def plotCurves(X_test, X_train, y_test, y_train, models):

    for model in models:
        ns_probs = [0 for _ in range(len(y_test))]

        y_probs = None

        if model == 'logistic':
            y_probs = logisticRegression(X_test, X_train, y_test, y_train, proba=True)
        elif model == 'svc':
            y_probs = svc(X_test, X_train, y_test, y_train, proba=True)
        elif model == 'xgbc':
            y_probs = xgbc(X_test, X_train, y_test, y_train, proba=True)
        elif model == 'rfc':
            y_probs = rfc(X_test, X_train, y_test, y_train, proba=True)

        precision = {}
        recall = {}
        average_precision = {}
        plt.figure()
        plt.title(model)
        for i in range(2):
            if model=='svc': y_test = y_test.head(100)
            precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_probs[:, i])
            average_precision[i] = average_precision_score(y_test == i, y_probs[:, i])

            plt.plot(recall[i], precision[i],
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right")

        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        # Plot ROC curve
        plt.figure()
        for i in range(2):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
        plt.legend()
        plt.title(model)
        plt.show()

def transformutilsColumns(X,liersSkew):
    pt = PowerTransformer(standardize=False)
    X[liersSkew] = pt.fit_transform(X[liersSkew])
    return X

def comparePolyDegree(X, y, degrees=[3]):
    X = X.head(100)
    y = y.head(100)

    # només té dos valors i no permet fer bé un plot
    X = X.drop(columns=['RainToday'])

    selector = SelectKBest(chi2, k=6)
    selector.fit(X, y)
    print(X.columns[selector.get_support(indices=True)])  # top 2 columns

    # al ser un plot 2D hem de buscar quines són les 2 millores característiques a mostrar
    selector = SelectKBest(chi2, k=2)
    # selector = SelectKBest(f_classif, k=2)
    selector.fit(X, y)
    X_new = selector.transform(X)
    print(X.columns[selector.get_support(indices=True)])  # top 2 columns
    names = X.columns[selector.get_support(indices=True)].tolist()  # top 2 columns


    models = []
    for d in degrees:
        poly_svc = svm.SVC(kernel='poly',degree=d, C=50).fit(X_new, y)
        models.append(poly_svc)

    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X_new[:, 0].min() - 0.1, X_new[:, 0].max() + 0.1
    y_min, y_max = X_new[:, 1].min() - 0.1, X_new[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = 'SVC POLY'

    y['RainTomorrow'] = y['RainTomorrow'].map({1: 'green', 0: 'black'})
    colors = y['RainTomorrow'].to_list()

    for i, clf in enumerate(models):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X_new[:, 0], X_new[:, 1], c=colors, cmap=plt.cm.coolwarm)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        textDegree = ', d=' + str(degrees[i])
        plt.title(titles + textDegree)

    plt.show()

def compareRbfGamma(X, y, Cs=[1], gammas=[1]):
    # només té dos valors i no permet fer bé un plot
    # X = X.drop(columns=['RainToday','Temp9am', 'Temp3pm','Cloud9am', 'Cloud3pm','Pressure9am', 'Pressure3pm','Humidity9am','WindSpeed3pm','WindSpeed9am','WindDir3pm'])
    X = X.drop(columns=['RainToday'])

    X = X.head(100)
    y = y.head(100)





    selector = SelectKBest(chi2, k=6)
    selector.fit(X, y)
    print(X.columns[selector.get_support(indices=True)])  # top 2 columns

    # al ser un plot 2D hem de buscar quines són les 2 millores característiques a mostrar
    selector = SelectKBest(chi2, k=2)
    # selector = SelectKBest(f_classif, k=2)
    selector.fit(X, y)
    X_new = selector.transform(X)
    print(X.columns[selector.get_support(indices=True)])  # top 2 columns
    names = X.columns[selector.get_support(indices=True)].tolist()  # top 2 columns


    models = []
    for g, c in zip(gammas,Cs):
        rbf_svc = svm.SVC(kernel='rbf', gamma=g, C=c).fit(X_new, y)
        models.append(rbf_svc)


    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X_new[:, 0].min() - 0.1, X_new[:, 0].max() + 0.1
    y_min, y_max = X_new[:, 1].min() - 0.1, X_new[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = 'SVC RBF'


    y['RainTomorrow'] = y['RainTomorrow'].map({1: 'green', 0: 'black'})
    colors = y['RainTomorrow'].to_list()

    for i, clf in enumerate(models):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X_new[:, 0], X_new[:, 1], c=colors, cmap=plt.cm.coolwarm)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        textGamma = ', g=' + str(gammas[i])
        textC = ', C=' + str(Cs[i])
        plt.title(titles + textC + textGamma)

    plt.show()

def compareDifferentkernels(X, y, C=1, gamma=1):
    X = X.head(100)
    y = y.head(100)

    # només té dos valors i no permet fer bé un plot
    X = X.drop(columns=['RainToday'])

    selector = SelectKBest(chi2, k=6)
    selector.fit(X, y)
    print(X.columns[selector.get_support(indices=True)])  # top 2 columns


    # al ser un plot 2D hem de buscar quines són les 2 millores característiques a mostrar
    selector = SelectKBest(chi2, k=2)
    # selector = SelectKBest(f_classif, k=2)
    selector.fit(X, y)
    X_new = selector.transform(X)
    print(X.columns[selector.get_support(indices=True)])  # top 2 columns
    names = X.columns[selector.get_support(indices=True)].tolist()  # top 2 columns

    # title for the plots
    textC = ', C='+str(C)
    textGamma= ', g='+str(gamma)
    titles = ['SVC linear',
              'SVC Sigmoid',
              'SVC RBF',
              'SVC poly degree=3']

    svc = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X_new, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X_new, y)
    poly_svc = svm.SVC(kernel='poly', gamma=gamma, degree=3, C=C).fit(X_new, y)
    sigmoid_svc = svm.SVC(kernel='sigmoid', gamma=gamma, C=C).fit(X_new, y)
    # lin_svc = svm.LinearSVC(C=C).fit(X_new, y)

    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X_new[:, 0].min() - 0.1, X_new[:, 0].max() + 0.1
    y_min, y_max = X_new[:, 1].min() - 0.1, X_new[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))




    y['RainTomorrow'] = y['RainTomorrow'].map({1: 'green', 0: 'black'})
    colors = y['RainTomorrow'].to_list()

    for i, clf in enumerate((svc, sigmoid_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X_new[:, 0], X_new[:, 1], c=colors, cmap=plt.cm.coolwarm)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i]+textC+textGamma)

    plt.show()



    # y_pred = lin_svc.predict(X_new[:500,:])
    # print("Accuracy: ", accuracy_score(y.head(500), y_pred))
    # print("f1 score: ", f1_score(y.head(500), y_pred))

def RandomSearchRFC(X_train, y_train):
    param = {'n_estimators': [100, 300, 500],
             'max_depth': [4, 5, 6],
             'min_samples_split': [2, 4, 6],
             'min_samples_leaf': [1, 3, 5]}

    random_forest = RandomForestClassifier()
    search = RandomizedSearchCV(random_forest, param_distributions=param, n_iter=50, n_jobs=-1, cv=3, random_state=1)
    result = search.fit(X_train, y_train)

    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

def BayesianOptimizationRFC(X_train, y_train):
    param = {'n_estimators': [100, 300, 500],
             'max_depth': [4, 5, 6],
             'min_samples_split': [2, 4, 6],
             'min_samples_leaf': [1, 3, 5]}

    random_forest = RandomForestClassifier()
    search = BayesSearchCV(random_forest, search_spaces=param, n_jobs=-1, cv=3)
    result = search.fit(X_train, y_train)

    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

def decicionTree(X_test, X_train, y_test, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("\nDecision Tree")
    printMetrics(y_pred, y_test)


def baggingDecicionTree(X_test, X_train, y_test, y_train):
    bagDt = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                bootstrap=True, n_jobs=-1, oob_score=True)
    bagDt.fit(X_train, y_train)
    y_pred = bagDt.predict(X_test)
    print("\nBagging (Decision Tree)")
    printMetrics(y_pred, y_test)


def printMetrics(y_pred, y_test):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


def baggingRandomForest(X_test, X_train, y_test, y_train):
    bagDt = BaggingClassifier(RandomForestClassifier(), n_estimators=40, max_samples=1250,
                                bootstrap=True, n_jobs=-1, oob_score=True)
    bagDt.fit(X_train, y_train)
    y_pred = bagDt.predict(X_test)
    print("\nBagging (Random Forest)")
    printMetrics(y_pred, y_test)


def baggingXGBC(X_test, X_train, y_test, y_train):
    bagDt = BaggingClassifier(XGBClassifier(objective='binary:logistic', use_label_encoder =False, n_estimators=5,gamma=0.5, random_state=0), n_estimators=40, max_samples=1250,
                                bootstrap=True, n_jobs=-1, oob_score=True)
    bagDt.fit(X_train, y_train)
    y_pred = bagDt.predict(X_test)
    print("\nBagging (XGBClassifier)")
    printMetrics(y_pred, y_test)

def aprenentatges(X, y, models, sizes):
    for size in sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        X_train, y_train = balanceData(X_train, y_train)
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nLogistic")
            printMetrics(y_pred, y_test)

def kfold(X, y):
    for k in range(2, 7):
        kf = KFold(n_splits=k)
        print(kf.get_n_splits(X))
        for train_index, test_index in kf.split(X):
         X_train, X_test = X[train_index], X[test_index]
         y_train, y_test = y[train_index], y[test_index]
         X_train, y_train = balanceData(X_train, y_train)
         # logisticRegression(X_test, X_train, y_test, y_train)
         # svc(X_test, X_train, y_test, y_train, False, ["poly"])
         # rfc(X_test, X_train, y_test, y_train)
         # xgbc(X_test, X_train, y_test, y_train)
         # svcLinear(X_test, X_train, y_test, y_train)
         knn(X_test, X_train, y_test, y_train, 2)

def strack_modle(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balanceData(X_train, y_train)
    base_models = [('xgb',
                    XGBClassifier(objective='binary:logistic', use_label_encoder=False, n_estimators=5, gamma=0.5,
                                  random_state=0)), ('lr', LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001))]
    meta_model = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, cv=5)

    stacking_model.fit(X_train, y_train)
    y_pred= stacking_model.predict(X_test)
    print(f1_score(y_test, y_pred))

def main():
    database = pd.read_csv('./weatherAUS.csv')

    analyseData(database)

    database = fixMissingValues(database)
    database = cleanAndEnchanceData(database)
    # database = removeOutliers(database)
    y = database[['RainTomorrow']]
    X = database.drop(columns=('RainTomorrow'))
    X = standarise(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balanceData(X_train, y_train)

    logisticRegression(X_test, X_train, y_test, y_train)
    svc(X_test, X_train, y_test, y_train, False, ["poly"])
    xgbc(X_test, X_train, y_test, y_train)
    baggingXGBC(X_test, X_train, y_test, y_train)
    rfc(X_test, X_train, y_test, y_train)
    baggingRandomForest(X_test, X_train, y_test, y_train)
    svcLinear(X_test, X_train, y_test, y_train)

    decicionTree(X_test, X_train, y_test, y_train)
    baggingDecicionTree(X_test, X_train, y_test, y_train)


    plotCurves(X_test, X_train, y_test, y_train, [ 'svc'])
    # Cs and gammas MUST BE same length
    compareRbfGamma(X_train, y_train,Cs=[0.1,1,10,1000], gammas=[0.1,1,10,100])
    comparePolyDegree(X_train, y_train,degrees=[2,3,4,10])
    compareDifferentkernels(X_train, y_train, gamma=50, C=50)

    RandomSearchRFC(X_train, y_train)
    BayesianOptimizationRFC(X_train, y_train)

    #X, y = ModifyDatabase(database)
    #X = fixMissingValuesMedian(X)

    # X = EnchanceData(X)
    # X = pd.DataFrame(X.toarray())

    # liersSkewindex = NormalitzeData(X)
    # X = transformutilsColumns(X, liersSkewindex)
    # X = standarise(X)

    sizes = [0.2, 0.3, 0.4, 0.5]
    lr = LogisticRegression()
    svmc = svm.SVC(C=100, kernel="poly", probability=True, random_state=0)
    dt = DecisionTreeClassifier(max_depth=6)
    rf = RandomForestClassifier(max_samples=0.9)
    knn = KNeighborsClassifier(n_neighbors=5)

    models = [lr, dt, rf, knn]
    aprenentatges(X, y, models, sizes)
    kfold(X,y)
    strack_modle(X,y)

    #Cros validation
    scores = cross_val_score(LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001), X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(svm.SVC(C=2, kernel="poly", probability=False, random_state=0, tol=0.0001, max_iter=500), X, y, cv=5,scoring="f1_macro")
    # scores = cross_val_score(RandomForestClassifier(max_leaf_nodes=15,n_estimators=100, ccp_alpha=0.0,bootstrap=True, random_state=0),X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(XGBClassifier(objective='binary:logistic', use_label_encoder =False, n_estimators=5,gamma=0.5, random_state=0),X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(
    #     svm.LinearSVC(C=2,max_iter=500, penalty="l2",random_state=0, tol=1e-4),
    #     X, y, cv=5, scoring="f1_macro")
    # scores = cross_val_score(
    #     KNeighborsClassifier(n_neighbors=2, weights="uniform", p=2),
    #     X, y, cv=5, scoring="f1_macro")
    print(scores)



#names = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RainTomorrow']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()