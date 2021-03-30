import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple
from sklearn import ensemble
from processing.learn_model import rename


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
        serial_number_for_prediction: str) -> Tuple[float, float]:

    temperature = rename(temperature, 'temp', serial_number_for_prediction)
    target_temperature = rename(target_temperature, 'target')
    valve_level = rename(valve_level, 'valve')

    df_combined = pd.concat([temperature, target_temperature, valve_level])
    df_combined = df_combined.drop(columns=['serialNumber'])

    df_combined = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')

    # show_plot(df_combined)

    with Path('models/reg_temp_baseline.p').open('rb') as classifier_file:
        reg_rf_temp_baseline = pickle.load(classifier_file)

    with Path('models/reg_valve_baseline.p').open('rb') as classifier_file:
        reg_rf_valve_baseline = pickle.load(classifier_file)


    last_sample = df_combined.tail(1)
    last_sample = last_sample[['temp', 'target', 'valve']].to_numpy()
    y_pred_temp_baseline = reg_rf_temp_baseline.predict(last_sample)


    last_sample = df_combined.tail(1)
    last_sample = last_sample[['temp', 'target',  'valve']].to_numpy()
    y_pred_valve_baseline = reg_rf_valve_baseline.predict(last_sample)


    # print('y_pred_temp_baseline', y_pred_temp_baseline)
    # print('y_pred_valve_baseline', y_pred_valve_baseline)

    return y_pred_temp_baseline, y_pred_valve_baseline
