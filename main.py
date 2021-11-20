from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats

pd.set_option("display.max_columns", None)

#names = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RainTomorrow']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    database = pd.read_csv('./weatherAUS.csv')
    print("Informacio de la base de dades:")
    print(database.info())

    print("---------\ncapçalera:")
    print(database.head())

    sns.heatmap(database.isnull(), yticklabels = False, cbar=False, cmap='viridis')
    plt.show()

    print("---------\npercentatges de nuls en cada variable:")
    missing = pd.DataFrame(database.isnull().sum(), columns=['No. of missing values'])
    missing['% missing_values'] = (missing / len(database)).round(2) * 100
    print(missing)

    print("valors duplicats: ", database.duplicated().sum())

    #Fer proves amb la columna resultat
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

    print("valoren es x:",x.shape)
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

    #MinTemp
    sns.boxplot(x='MinTemp', data=databaseChangeRain, palette='Set2', ax=axes[4])
    axes[4].set_title("")

    plt.tight_layout()
    plt.show()

    #hacemos la moda en los Nan a toda la base de datos
    #toda la base de datos ponemos moda


    variables = list(x.select_dtypes(include=['float64', 'object']).columns)
    print("Valores nulos ?(con moda):")
    xModedf = x.copy()
    for i in variables:
        xModedf[i].fillna(xModedf[i].mode()[0], inplace=True)
        print(i, ": ", xModedf[i].isna().sum())

    #toda la base de datos ponemos mediana menos a los de tipo objeto a esos no se les puede calcular la mediana
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
    #toda la base de datos ponemos media menos a los de tipo objeto a esos no se les puede calcular la mediana
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