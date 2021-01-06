import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm


def rename(df, col_name, sn = None):
    # print("rename")
    # change col name to more fit
    df = df.rename(columns={'value': col_name})

    # delete useless column
    df = df.drop(columns=['unit'])

    # delete useless values
    if sn is not None:
        df = df[df['serialNumber'] == sn]
    #
    return df

def learn_model_baseline(X_train, y_train):
    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    # reg_rf = svm.SVR()
    reg_rf.fit(X_train, y_train)
    pickle.dump(reg_rf, open('./data/clf_baseline.p', 'wb'))

def learn_model(X_train, y_train):
    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    # reg_rf = svm.SVR()
    reg_rf.fit(X_train, y_train)
    pickle.dump(reg_rf, open('./data/clf.p', 'wb'))

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
    print(df_combined)
    df_combined = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')


    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value=20)

    # delete weekend days (Sat & Sun)
    # df_combined = df_combined[df_combined.index.dayofweek < 5]
    # show_plot(df_combined)

    to_calulate = df_combined.tail(1).index + pd.DateOffset(minutes=15)
    # print("to calculate:", to_calulate)


    to_calulate_down = to_calulate - 10 * pd.DateOffset(minutes=15) - pd.DateOffset(days=1)
    to_calulate_up = to_calulate + 10 * pd.DateOffset(minutes=15) - pd.DateOffset(days=1)
    # print("calculate range:", to_calulate_down.values[0], ', ', to_calulate_up.values[0])

    mask = (df_combined.index > str(to_calulate_down.values[0])) & (df_combined.index <= str(to_calulate_up.values[0]))

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
        mask_pom = (df_combined.index > str(to_calulate_down.values[0])) & (
                    df_combined.index <= str(to_calulate_up.values[0]))
        mask2 = np.add(mask2, mask_pom)


    # df_train = df_combined.loc[mask2]

    df_train = df_combined
    # print(df_train.head(5))
    # print(df_train.tail(5))
    # print(df_train.describe())
    # print(df_train.head(21))
    # show_plot(df_train)
    X_train = df_train[['temp', 'valve']].to_numpy()[1:-1]
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    return X_train, y_train

def preprocess_data():

    # temperature = pd.read_csv('.././WZUM_project_2020.12.20/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv', index_col=0, parse_dates=True)
    # target_temperature = pd.read_csv('.././WZUM_project_2020.12.20/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv', index_col=0, parse_dates=True)
    # valve_level = pd.read_csv('.././WZUM_project_2020.12.20/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv', index_col=0, parse_dates=True)

    temperature = pd.read_csv(
        'WZUM_project_2020.12.20/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv', index_col=0,
        parse_dates=True)
    target_temperature = pd.read_csv(
        'WZUM_project_2020.12.20/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv',
        index_col=0, parse_dates=True)
    valve_level = pd.read_csv(
        'WZUM_project_2020.12.20/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv', index_col=0,
        parse_dates=True)


    temperature = rename(temperature, 'temp', '0015BC0035001299')
    target_temperature = rename(target_temperature, 'target')
    valve_level = rename(valve_level, 'valve')

    df_combined = pd.concat([temperature, target_temperature, valve_level])
    df_combined = df_combined.drop(columns=['serialNumber'])

    df_combined = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    print(df_combined)
    df_combined.drop(df_combined.tail(384).index, inplace=True)
    print(df_combined)


    # df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20)

    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20)


    df_train = df_combined

    X_train = df_train[['temp', 'valve']].to_numpy()[1:-1]
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    return X_train, y_train


# preprocess_data()
'''
Zmiany:
-

'''




