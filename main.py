from sklearn import preprocessing
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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
    print("valors duplicats: ", database.duplicated().sum())
    # Fer proves amb la columna resultat
    databaseChangeRain = database.copy()
    print(database['RainTomorrow'].value_counts())
    databaseChangeRain['RainTomorrow'] = [1 if i == 'Yes' else 0 for i in databaseChangeRain['RainTomorrow']]
    print(databaseChangeRain['RainTomorrow'].value_counts())
    sns.heatmap(database.corr(), annot=True, linewidths=.5, cmap='rocket');
    plt.show()
    cols_to_drop = ['Date']
    databaseChangeRain.drop(columns=cols_to_drop, inplace=True)
    x = databaseChangeRain.drop(['RainTomorrow'], axis=1)
    y = databaseChangeRain['RainTomorrow']
    print("valoren es x:", x.shape)
    print("valoren es y:", y.shape)
    # creacio d'un plot per veure la distribucio de les dades
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharey=False)
    fig.suptitle('Distribution')
    # Rainfall
    sns.boxplot(x="Rainfall", data=databaseChangeRain, palette='Set2', ax=axes[0])
    axes[0].set_title("")
    # Evaporation
    sns.boxplot(x='Evaporation', data=databaseChangeRain, palette='Set2', ax=axes[1])
    axes[1].set_title("")
    # Windspeed (9AM)
    sns.boxplot(x='WindSpeed9am', data=databaseChangeRain, palette='Set2', ax=axes[2])
    axes[2].set_title("")
    # Windspeed (3PM)
    sns.boxplot(x='WindSpeed3pm', data=databaseChangeRain, palette='Set2', ax=axes[3])
    axes[3].set_title("")
    # MinTemp
    sns.boxplot(x='MinTemp', data=databaseChangeRain, palette='Set2', ax=axes[4])
    axes[4].set_title("")
    plt.tight_layout()
    plt.show()
    # hacemos la moda en los Nan a toda la base de datos
    # toda la base de datos ponemos moda
    variables = list(x.select_dtypes(include=['float64', 'object']).columns)
    print("Valores nulos ?(con moda):")
    xModedf = x.copy()
    for i in variables:
        xModedf[i].fillna(xModedf[i].mode()[0], inplace=True)
        print(i, ": ", xModedf[i].isna().sum())
    # toda la base de datos ponemos mediana menos a los de tipo objeto a esos no se les puede calcular la mediana
    print("Valores nulos ?(con mediana):")
    xMediandf = x.copy()
    for i in variables:
        if (np.dtype(xModedf[i]) == 'object'):
            xMediandf[i].fillna(xModedf[i].mode()[0], inplace=True)
        else:
            xMediandf[i].fillna(xMediandf[i].median(), inplace=True)

        print(i, ": ", xMediandf[i].isna().sum())
    print("Valores nulos ?(con media):")
    xMeandf = x.copy()
    # toda la base de datos ponemos media menos a los de tipo objeto a esos no se les puede calcular la mediana
    for i in variables:
        if (np.dtype(xMeandf[i]) == 'object'):
            xMeandf[i].fillna(xModedf[i].mode()[0], inplace=True)
        else:
            xMeandf[i].fillna(xMediandf[i].mean(), inplace=True)
        print(i, ": ", xMeandf[i].isna().sum())
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharey=False)
    fig.suptitle('Distribution Mode')
    # Rainfall
    sns.boxplot(x="Rainfall", data=xModedf, palette='Set2', ax=axes[0])
    axes[0].set_title("")
    # Evaporation
    sns.boxplot(x='Evaporation', data=xModedf, palette='Set2', ax=axes[1])
    axes[1].set_title("")
    # Windspeed (9AM)
    sns.boxplot(x='WindSpeed9am', data=xModedf, palette='Set2', ax=axes[2])
    axes[2].set_title("")
    # Windspeed (3PM)
    sns.boxplot(x='WindSpeed3pm', data=xModedf, palette='Set2', ax=axes[3])
    axes[3].set_title("")
    # MinTemp
    sns.boxplot(x='MinTemp', data=xModedf, palette='Set2', ax=axes[4])
    axes[4].set_title("")
    plt.tight_layout()
    plt.show()


def fixMissingValues(df):
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
    return df

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

def balanceData(X, y):
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    # bàsicament pel fet que un 80% de les Y són 0 i un 20% són 1
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    return X, y




def main():
    database = pd.read_csv('./weatherAUS.csv')
    # analyseData(database)

    # percentatge de nulls abans i després
    print((database.isnull().sum() / len(database)) * 100)
    database = fixMissingValues(database)
    print((database.isnull().sum() / len(database)) * 100)

    database = cleanAndEnchanceData(database)

    y = database[['RainTomorrow']]
    X = database.drop(columns=('RainTomorrow'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_test = balanceData(X_train, y_train)

#names = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RainTomorrow']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()