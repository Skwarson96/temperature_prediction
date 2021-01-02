import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import ensemble

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


def rename(df, col_name, sn = None):
    # print("rename")
    # change col name to more fit
    df = df.rename(columns={'value': col_name})

    # delete useless column
    df = df.drop(columns=['unit'])

    # delete useless values
    if sn is not None:
        df = df[df['serialNumber'] == sn]

    return df

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

    # print(temperature.head(5))
    # print(temperature.tail(5))

    temperature = rename(temperature, 'temp', serial_number_for_prediction)
    target_temperature = rename(target_temperature, 'target')
    valve_level = rename(valve_level, 'valve')

    df_combined = pd.concat([temperature, target_temperature, valve_level])
    df_combined = df_combined.drop(columns=['serialNumber'])

    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')



    # show_plot(df_combined)

    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value=20)

    # print(df_combined.head(5))
    # print(df_combined.tail(5))

    # delete weekend days (Sat & Sun)
    df_combined = df_combined[df_combined.index.dayofweek < 5]
    # show_plot(df_combined)

    to_calulate = df_combined.tail(1).index + pd.DateOffset(minutes=15)
    # to_calulate = to_calulate.to_period("15T")
    # print("to calculate:", to_calulate)
    to_calulate_down = to_calulate - 10 * pd.DateOffset(minutes=15) - pd.DateOffset(days=1)
    to_calulate_up = to_calulate + 10 * pd.DateOffset(minutes=15) - pd.DateOffset(days=1)
    # print("calculate range:", to_calulate_down.values[0], ', ', to_calulate_up.values[0])

    # print(mask)

    days = 6
    mask2 = np.zeros(len(df_combined.index), dtype=bool)
    # mask2 = mask2.transpose()
    # print(mask2)
    for day in range(days):
        day = day + 1
        # print(day)
        to_calulate_down = to_calulate - 10 * pd.DateOffset(minutes=15) - day * pd.DateOffset(days=1)
        to_calulate_up = to_calulate + 10 * pd.DateOffset(minutes=15) - day * pd.DateOffset(days=1)
        # print("calculate range:", to_calulate_down.values[0], ', ', to_calulate_up.values[0])
        mask_pom = (df_combined.index > str(to_calulate_down.values[0])) & (df_combined.index <= str(to_calulate_up.values[0]))
        mask2 = np.add(mask2, mask_pom)

    # df_train = df_combined.tail(150)
    df_train = df_combined.loc[mask2]
    show_plot(df_train)
    X_train = df_train[['temp', 'valve']].to_numpy()[1:-1]
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    # print(X_train)
    # print(y_train)

    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)

    # reg = ensemble.AdaBoostRegressor(random_state=0, n_estimators=100)
    # reg.fit(X_train, y_train)

    last_sample = df_combined.tail(1)
    last_sample = last_sample[['temp', 'valve']].to_numpy()
    # print(last_sample)

    y_pred = reg_rf.predict(last_sample)
    # y_pred = reg.predict(last_sample)


    print('y_pred', y_pred)

    # TODO resamplink na podstawie ilosci probek

    # exit()
    print('----------------------------------\n')
    return y_pred
