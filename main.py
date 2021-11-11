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


