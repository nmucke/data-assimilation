import pandas as pd
import pdb
import matplotlib.pyplot as plt

def main():

    #sensors_locations = [
    #    '02', '04', '05', '10', '12', '13', '14', '15', '17', '19', '21'
    #]
    sensors_locations = [
        '04', '10', '13', '14', '15', '17', '19', '21'
    ]

    filename = f'data/wave_submerged_bar/observations/a{sensors_locations[0]}.dat'

    #after testing replace StringIO(temp) to filename
    df = pd.read_csv(
        filename, 
        header=None,
        sep='\s+',
    )
    df.rename(
        columns={
            0: 'time',
            1: sensors_locations[0],
        },
        inplace=True
    )

    for sensor in sensors_locations[1:]:

        filename = f'data/wave_submerged_bar/observations/a{sensor}.dat'

        #after testing replace StringIO(temp) to filename
        df_i = pd.read_csv(
            filename, 
            header=None,
            sep='\s+',
        )

        df_i.rename(
            columns={
                0: 'time',
                1: sensor,
            },
            inplace=True
        )
        df_i.drop(columns=['time'], inplace=True)

        df = pd.concat([df, df_i], axis=1)
    
    df.to_csv('data/wave_submerged_bar/observations/observations.csv', index=False)

if __name__ == "__main__":
    main()