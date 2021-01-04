import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


from processing.utils import perform_processing
from processing.learn_model import preprocess_data
from processing.learn_model import learn_model


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
    df_target_temperature = pd.read_csv(arguments['file_target_temperature'], index_col=0, parse_dates=True)
    df_valve = pd.read_csv(arguments['file_valve_level'], index_col=0, parse_dates=True)

    df_temperature_resampled = df_temperature.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
    df_temperature_resampled = df_temperature_resampled.loc[start:stop]
    df_temperature_resampled['predicted'] = 0.0

    # print(df_temperature_resampled.head(5))
    # print(df_temperature_resampled.tail(5))

    current = start - pd.DateOffset(minutes=15)

    X_train, y_train = preprocess_data(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            arguments['serial_number']
        )

    learn_model(X_train, y_train)

    # exit()
    while current < stop:
        print('current', current)
        print('to calculate:', current + pd.DateOffset(minutes=15))
        predicted_temperature = perform_processing(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            arguments['serial_number']
        )
        current = current + pd.DateOffset(minutes=15)
        df_temperature_resampled.at[current, 'predicted'] = predicted_temperature
    df_temperature_resampled.to_csv(results_file)

    # print(df_temperature_resampled)
    print('\nmae: ',metrics.mean_absolute_error(df_temperature_resampled.value, df_temperature_resampled.predicted))
    plt.plot(df_temperature_resampled.index, df_temperature_resampled.value)
    plt.plot(df_temperature_resampled.index, df_temperature_resampled.predicted)
    plt.legend(['value', 'predicted'])
    plt.show()

if __name__ == '__main__':
    main()
