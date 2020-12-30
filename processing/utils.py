import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble

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

    # print(df_combined.head(5))
    # print(df_combined.tail(5))
    # print(len(df_combined.index))
    # df_combined.plot()

    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')

    # df_combined['temp_gt'] = df_combined['temp'].shift(-1)
    # df_combined['temp_gt'] = df_combined['temp_gt'].fillna(method='ffill')

    print(df_combined.head(5))
    print(df_combined.tail(5))
    # print(len(df_combined.index))

    to_calulate = df_combined.tail(1).index + pd.DateOffset(minutes=15)
    to_calulate = to_calulate.values
    print("to calculate:", to_calulate[0])

    # show_plot(df_combined)

    # plt.scatter(df_combined.tail(10).index, X_train[:,0])

    X_train = df_combined.tail(10)[['temp', 'target', 'valve']].to_numpy()

    y_train = df_combined.tail(10)[['temp_gt']].to_numpy()


    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)
    y_predicted = reg_rf.predict()
    print('y_predicted', y_predicted)



    # print(y_train)

    # print(type(to_calulate))
    # print(to_calulate[0])
    # print(type(to_calulate[0]))
    # y_train =
    #
    # print(y_train)
    # plt.scatter(df_combined.tail(10).index, y_train[:,0])
    # plt.show()
    # print(valve_level.head(5))
    # print(serial_number_for_prediction)
    # print(X_train)
    # print(y_train)

    #

    # exit()
    print('----------------------------------\n')
    return y_predicted[0][0]
