import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def main():

    #sensors_locations = [
    #    '02', '04', '05', '10', '12', '13', '14', '15', '17', '19', '21'
    #]
    sensors_locations = [
        '04', '10', '13', '14', '15', '17', '19', '21'
    ]

    df_list = []
    for sensor in sensors_locations:

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

        df_i_time = df_i['time'].values + 0.925

        init_time = np.linspace(
            0, df_i_time.min(), 100
        )
        df_i_time = np.concatenate(
            (init_time, df_i_time), axis=0
        )


        df_i_values = df_i[sensor].values
        init_zeros = np.zeros(
            shape=(100)
        )
        df_i_values = np.concatenate(
            (init_zeros, df_i_values), axis=0
        )

        t_interp = np.linspace(0, 45, 1273)
        t_interp = t_interp[0:1200]

        obs_interp = interp1d(
            df_i_time, df_i_values, kind='linear'
        )
        obs_interp = obs_interp(t_interp)

        df_i = pd.DataFrame(
            data=obs_interp,
            columns=[sensor]
        )
        df_list.append(df_i)

    df = pd.concat(df_list, axis=1)
    #df.insert(0, 'time', t_interp)
    
    df.to_csv('data/wave_submerged_bar/observations/observations.csv', index=False)

if __name__ == "__main__":
    main()