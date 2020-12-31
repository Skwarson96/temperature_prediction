import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import ensemble

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster


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



    to_calulate = df_combined.tail(1).index + pd.DateOffset(minutes=15)
    to_calulate = to_calulate.to_period("15T")
    print("to calculate:", to_calulate)

    to_calulate = pd.PeriodIndex(to_calulate)
    fh = ForecastingHorizon(to_calulate, is_relative=False)
    print(fh)

    y_train = df_combined.tail(10)
    y_train.index = y_train.index.to_period("15T")
    print(y_train['temp'])
    forecaster = NaiveForecaster(strategy="last", sp=1)
    forecaster.fit(y_train['temp'])
    print("test")

    y_pred = forecaster.predict(fh)

    print('y_pred', y_pred)


    # exit()
    print('----------------------------------\n')
    return y_pred
