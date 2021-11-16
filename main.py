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

    print(database.info())

    print(database.head())

    sns.heatmap(database.isnull(), yticklabels = False, cbar=False, cmap='viridis')

    missing = pd.DataFrame(database.isnull().sum(), columns=['No. of missing values'])
    missing['% missing_values'] = (missing / len(database)).round(2) * 100
    print(missing)


    print("valors duplicats: ", database.duplicated().sum())

    #fer proves
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
    print(x.shape, y.shape)

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharey=False)
    fig.suptitle('Distribution')

    # Rainfall
    sns.boxplot(x=databaseChangeRain["Rainfall"], data=databaseChangeRain, palette='Set2', ax=axes[0])
    axes[0].set_title("")

    # Sunshine
    sns.boxplot(x='Sunshine', data=databaseChangeRain, palette='Set2', ax=axes[1])
    axes[1].set_title("")

    # Evaporation
    sns.boxplot(x='Evaporation', data=databaseChangeRain, palette='Set2', ax=axes[2])
    axes[2].set_title("")

    # Windspeed (9AM)
    sns.boxplot(x='WindSpeed9am', data=databaseChangeRain, palette='Set2', ax=axes[3])
    axes[3].set_title("")

    # Windspeed (3PM)
    sns.boxplot(x='WindSpeed3pm', data=databaseChangeRain, palette='Set2', ax=axes[4])
    axes[4].set_title("")

    plt.tight_layout()
    plt.show()

    #hacemos la moda en los Nan a toda la base de datos
    #toda la base de datos con la moda
    for i in x.head():
        x[i].fillna(x[i].mode()[0], inplace=True)
        print(x[i].isna().sum())