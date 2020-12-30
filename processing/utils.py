import pandas as pd


def rename(df, col_name, sn = None):
    print("rename")
    # change col name to more fit
    df = df.rename(columns={'value': col_name})

    # delete useless column
    df = df.drop(columns=['unit'])

    # delete useless values
    if sn is not None:
        df = df[df['serialNumber'] == sn]

    return df


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

    # print(target_temperature.head(5))
    # print(valve_level.head(5))
    # print(serial_number_for_prediction)

    # exit()

    return 20
