import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


from processing.utils import perform_processing
from processing.learn_model import preprocess_data
from processing.learn_model import learn_model
from processing.learn_model import preprocess_data_baseline
from processing.learn_model import learn_model_baseline



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    results_file = Path(args.results_file)

    with open(input_file) as f:
        arguments = json.load(f)

    start = pd.Timestamp(arguments['start']).tz_localize('UTC')
    stop = pd.Timestamp(arguments['stop']).tz_localize('UTC')

    df_temperature = pd.read_csv(arguments['file_temperature'], index_col=0, parse_dates=True)
    df_temperature_serial_number = df_temperature[df_temperature['serialNumber'] == arguments['serial_number']]
    df_target_temperature = pd.read_csv(arguments['file_target_temperature'], index_col=0, parse_dates=True)
    df_valve = pd.read_csv(arguments['file_valve_level'], index_col=0, parse_dates=True)

    # print(df_temperature)

    # df_temperature = df_temperature[df_temperature['serialNumber'] == arguments['serial_number']]
    df_combined = pd.concat([
        df_temperature_serial_number.rename(columns={'value': 'temperature'}),
        df_target_temperature.rename(columns={'value': 'target_temperature'}),
        df_valve.rename(columns={'value': 'valve_level'})
    ])

    df_combined_resampled = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')


    df_combined_resampled = df_combined_resampled.loc[start:stop]
    df_combined_resampled['predicted_temperature'] = 0.0
    df_combined_resampled['predicted_valve_level'] = 0.0

    # print(df_temperature_resampled.head(5))
    # print(df_temperature_resampled.tail(5))

    current = start - pd.DateOffset(minutes=15)

    X_train, y_train = preprocess_data_baseline(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            arguments['serial_number']
        )

    learn_model_baseline(X_train, y_train)


    # X_train, y_train = preprocess_data()

    # learn_model(X_train, y_train)


    while current < stop:
        print('current', current)
        print('to calculate:', current + pd.DateOffset(minutes=15))
        predicted_temperature_baseline, predicted_valve_level = perform_processing(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            arguments['serial_number']
        )
        current = current + pd.DateOffset(minutes=15)
        # df_temperature_resampled.at[current, 'predicted_baseline'] = predicted_temperature_baseline
        # df_temperature_resampled.at[current, 'predicted'] = predicted_temperature
        # df_temperature_resampled.at[current, 'predicted_'] = predicted_temperature_

        df_combined_resampled.at[current, 'predicted_temperature'] = predicted_temperature_baseline
        df_combined_resampled.at[current, 'predicted_valve_level'] = predicted_valve_level

    # df_temperature_resampled.to_csv(results_file)
    df_combined_resampled.to_csv(results_file)
    # print(df_temperature_resampled)
    print('\ntemp mae baseline: ',metrics.mean_absolute_error(df_combined_resampled.temperature, df_combined_resampled.predicted_temperature))
    print('\nvalve mae baseline: ',metrics.mean_absolute_error(df_combined_resampled.valve_level, df_combined_resampled.predicted_valve_level))
    # print('\nmae: ',metrics.mean_absolute_error(df_temperature_resampled.value, df_temperature_resampled.predicted))
    # print('\nmae predicted_: ',metrics.mean_absolute_error(df_temperature_resampled.value, df_temperature_resampled.predicted_))

    # plt.plot(df_temperature_resampled.index, df_temperature_resampled.value)
    # plt.plot(df_temperature_resampled.index, df_temperature_resampled.predicted_baseline)
    # plt.plot(df_temperature_resampled.index, df_temperature_resampled.predicted)
    # plt.plot(df_temperature_resampled.index, df_temperature_resampled.predicted_)

    # plt.legend(['value', 'baseline' ,'predicted', 'predicted_'])
    # plt.show()

if __name__ == '__main__':
    main()
