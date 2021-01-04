import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


from sklearn import ensemble


from processing.learn_model import rename

from processing.learn_model import preprocess_data
from processing.learn_model import learn_model
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedRegressionForecaster,
    TransformedTargetForecaster,
)

from sklearn.linear_model import LinearRegression


# from sklearn.linear_model import RidgeClassifierCV
# from sktime.transformations.panel.rocket import Rocket



def show_plot(df):
    fig, ax1 = plt.subplots()
    lin1 = ax1.plot(df.index, df.temp, label='temp')
    lin2 = ax1.plot(df.index, df.target, label='target')
    ax1.set_ylabel('temp [C]')
    # ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel('valve [%]')
    lin3 = ax2.plot(df.index, df.valve, color='tab:red', label='valve')

    # legend
    lins = lin1 + lin2 + lin3
    labs = [l.get_label() for l in lins]
    ax1.legend(lins, labs, loc=0)

    plt.show()

def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str) -> float:

    temperature = rename(temperature, 'temp', serial_number_for_prediction)
    target_temperature = rename(target_temperature, 'target')
    valve_level = rename(valve_level, 'valve')

    df_combined = pd.concat([temperature, target_temperature, valve_level])
    df_combined = df_combined.drop(columns=['serialNumber'])


    df_combined = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')

    # show_plot(df_combined)

    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value=20)

    # print(df_combined.head(5))
    # print(df_combined.tail(5))
    # print(df_combined.describe())
    # # delete weekend days (Sat & Sun)
    # # df_combined = df_combined[df_combined.index.dayofweek < 5]
    # # show_plot(df_combined)
    #
    to_calulate = df_combined.tail(1).index + pd.DateOffset(minutes=15)
    # # to_calulate = to_calulate.to_period("15T")
    print("to calculate:", to_calulate)



    with Path('data/clf_baseline.p').open('rb') as classifier_file:
        reg_rf_baseline = pickle.load(classifier_file)


    with Path('data/clf.p').open('rb') as classifier_file:
        reg_rf = pickle.load(classifier_file)


    last_sample = df_combined.tail(1)
    last_sample = last_sample[['temp', 'valve']].to_numpy()
    print(last_sample)

    y_pred_baseline = reg_rf_baseline.predict(last_sample)

    last_sample = df_combined.tail(1)
    last_sample = last_sample[['temp', 'valve']].to_numpy()
    y_pred = reg_rf.predict(last_sample)


    print('y_pred_baseline', y_pred_baseline)
    print('y_pred', y_pred)

    # exit()
    print('----------------------------------\n')
    return y_pred_baseline, y_pred
