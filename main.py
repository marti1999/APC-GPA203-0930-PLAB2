from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_curve, average_precision_score, \
    roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



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
    sns.heatmap(databaseChangeRain.corr(), annot=True, linewidths=.5, cmap='rocket');
    plt.show()

    cols_to_drop = ['Date']
    databaseChangeRain.drop(columns=cols_to_drop, inplace=True)

    x = databaseChangeRain.drop(['RainTomorrow'], axis=1)
    y = databaseChangeRain['RainTomorrow']

    plotVariable=['Rainfall', 'Evaporation', 'WindSpeed9am','WindSpeed3pm','MinTemp']
    # creacio d'un plot per veure la distribucio de les dades
    plotVariablesBox(databaseChangeRain, plotVariable)
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
    print("Valores nulos ?(con moda):")
    xModedf = df.copy()
    for i in variables:
        xModedf[i].fillna(xModedf[i].mode()[0], inplace=True)
    print(i, ": ", xModedf[i].isna().sum())
    return xModedf

def fixMissingValuesMedian(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xMediandf = df.copy()
    for i in variables:
        if (np.dtype(xMediandf[i]) == 'object'):
            xMediandf[i].fillna(xMediandf[i].mode()[0], inplace=True)
        else:
            xMediandf[i].fillna(xMediandf[i].median(), inplace=True)
    print(i, ": ", xMediandf[i].isna().sum())
    return xMediandf

def fixMissingValuesMean(df):
    variables = list(df.select_dtypes(include=['float64', 'object']).columns)
    xMeandf = df.copy()
    # toda la base de datos ponemos media menos a los de tipo objeto a esos no se les puede calcular la mediana
    for i in variables:
        if (np.dtype(xMeandf[i]) == 'object'):
            xMeandf[i].fillna(xMeandf[i].mode()[0], inplace=True)
        else:
            xMeandf[i].fillna(xMeandf[i].mean(), inplace=True)
    print(i, ": ", xMeandf[i].isna().sum())
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
    transformer = ColumnTransformer(
                            transformers=[('notCategoric',
                                            OneHotEncoder(sparse='False', drop='first'),
                                           variablesCategoric)],
                            remainder='passthrough')
    return transformer.fit_transform(x)


def standarise(df, with_mean=False):
    # scaler = preprocessing.MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    # df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
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
    logireg = LogisticRegression(max_iter=500)
    logireg.fit(X_train, y_train.values.ravel())  # https://www.geeksforgeeks.org/python-pandas-series-ravel/
    if proba:
        y_pred = logireg.predict_proba(X_test)
        return y_pred

    y_pred = logireg.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))

def svcLinear(X_test, X_train, y_test, y_train):
    # https://scikit-learn.org/stable/modules/svm.html#complexity
    svc = svm.LinearSVC(random_state=0, tol=1e-5)
    svc.fit(X_train, y_train.values.ravel())
    y_pred = svc.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


def svc(X_test, X_train, y_test, y_train, proba=False, kernels=['rbf']):
    for kernel in kernels:
        svc = svm.SVC(C=10.0, kernel=kernel, gamma=0.9, probability=True, random_state=0)
        svc.fit(X_train.head(10000), y_train.head(10000).values.ravel())
        if proba:
            y_pred = svc.predict_proba(X_test.head(10000))
            return y_pred
        y_pred = svc.predict(X_test.head(10000))
        print("Accuracy: ", accuracy_score(y_test.head(10000), y_pred))
        print("f1 score: ", f1_score(y_test.head(10000), y_pred))


def xgbc(X_test, X_train, y_test, y_train, proba=False):
    xgbc = XGBClassifier(objective='binary:logistic', use_label_encoder =False)
    xgbc.fit(X_train,y_train.values.ravel())
    if proba:
        y_pred = xgbc.predict_proba(X_test)
        return y_pred

    y_pred = xgbc.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


def rfc(X_test, X_train, y_test, y_train, proba=False):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train,y_train.values.ravel())
    if proba:
        y_pred = clf.predict_proba(X_test)
        return y_pred

    y_pred = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


def plotCurves(X_test, X_train, y_test, y_train, models):

    for model in models:
        ns_probs = [0 for _ in range(len(y_test))]

        y_probs = None

        if model == 'logistic':
            y_probs = logisticRegression(X_test, X_train, y_test, y_train, proba=True)
        elif model == 'svcLinear':
            continue
        elif model == 'xgbc':
            y_probs = xgbc(X_test, X_train, y_test, y_train, proba=True)
        elif model == 'rfc':
            y_probs = rfc(X_test, X_train, y_test, y_train, proba=True)

        precision = {}
        recall = {}
        average_precision = {}
        plt.figure()
        for i in range(2):
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

        plt.show()
def transformutilsColumns(X,liersSkew):
    pt = PowerTransformer(standardize=False)
    X[liersSkew] = pt.fit_transform(X[liersSkew])
    return X

def main():
    database = pd.read_csv('./weatherAUS.csv')
    analyseData(database)

    X, y = ModifyDatabase(database)
    X = fixMissingValuesMean(X)
    X = EnchanceData(X)

    #X = removeOutliers(X)
    X = pd.DataFrame(X.toarray())
    liersSkewindex = NormalitzeData(X)
    X=transformutilsColumns(X,liersSkewindex)

    X = standarise(X)

    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balanceData(X_train, y_train)

    logisticRegression(X_test, X_train, y_test, y_train)
    svcLinear(X_test, X_train, y_test, y_train)
    xgbc(X_test, X_train, y_test, y_train)
    rfc(X_test, X_train, y_test, y_train)
    #svc(X_test, X_train, y_test, y_train, kernels=['linear', 'rbf', 'sigmoid'])

    plotCurves(X_test, X_train, y_test, y_train, ['logistic', 'xgbc', 'rfc'])







#names = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RainTomorrow']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()