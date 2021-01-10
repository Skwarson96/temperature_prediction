import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn import ensemble


def rename(df, col_name, sn = None):
    # change col name to more fit
    df = df.rename(columns={'value': col_name})

    # delete useless column
    df = df.drop(columns=['unit'])

    # delete useless values
    if sn is not None:
        df = df[df['serialNumber'] == sn]

    return df

def learn_model_temp_baseline(X_train, y_train):
    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)
    pickle.dump(reg_rf, open('./data/reg_temp_baseline.p', 'wb'))

def learn_model_valve_baseline(X_train, y_train):
    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)
    pickle.dump(reg_rf, open('./data/reg_valve_baseline.p', 'wb'))


def preprocess_data_baseline(
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


    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value=20)

    df_train = df_combined

    X_train = df_train[['temp', 'target', 'valve']].to_numpy()[1:-1]
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    X_train_valve = df_train[['temp', 'target', 'valve']].to_numpy()[1:-1]
    y_train_valve = df_train['valve_gt'].to_numpy()[1:-1]

    return X_train, y_train, X_train_valve, y_train_valve





