import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from scipy.interpolate import interp1d
from data_assimilation.particle_filter.observation_operator import ObservationOperator

from data_assimilation.true_solution import TrueSolution

TEST_CASE = 'wave_submerged_bar'

MODEL_TYPE = 'latent'
LOW_FIDELITY_NUM_PARTICLES_LIST = [500]

CONFIG_PATH = f'configs/{MODEL_TYPE}/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PLOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

# Initialize observation operator
observation_operator = ObservationOperator(
    **config['observation_operator_args'],
    backend=config['backend'],
)

test_case_index = 0
LOCAL_PATH = f'data/{TEST_CASE}/test'
true_state = np.load(
    f'{LOCAL_PATH}/state/sample_{test_case_index}.npz'
)['data']
true_pars = np.load(
    f'{LOCAL_PATH}/pars/sample_{test_case_index}.npz'
)['data']

true_sol_obs_args = {
  'observation_file': None,
  'noise_variance': 5.0e-12,
  'observation_times': [0, 1200, 1],
  'full_time_points': [0, 45, 1273],
}
true_solution = TrueSolution(
    state=true_state,
    pars=true_pars,
    observation_operator=observation_operator,  
    backend=config['backend'],
    **true_sol_obs_args,
)

def main():

    x = np.linspace(0.00, 25.6, 512)



    t_vec = np.linspace(0, 45, 1273) + 0.925
    t_interp = t_vec[18:-100]

    obs_loc = [4, 10.5, 13.5, 14.5, 15.7, 17.3, 19.0, 21.0]
    
    lal = []
    for i in range(len(obs_loc)):
        lal.append((np.abs(x - obs_loc[i])).argmin())
    
    

    # Load high fidelity results.

    true_observations = pd.read_csv(
        'data/wave_submerged_bar/observations/observations.csv', 
    ).values

    '''

    for i in range(0, true_observations.shape[1]-1):
        obs_interp = interp1d(
           obs_time, true_observations[:, i+1], kind='linear'
        )
        true_observations_new[:, i] = obs_interp(t_interp)
        
    zeroes = np.zeros(
        shape=(18, true_observations_new.shape[1])
    )
    true_observations_new = np.concatenate(
        (zeroes, true_observations_new), axis=0
    )
    true_observations = true_observations_new
    '''


    #t_vec = t_vec[0:pred_observations.shape[-1]]
    #true_observations = true_observations[0:pred_observations.shape[-1]]

    #obs_times = t_vec[obs_times_ids]

    hf_t_vec = np.linspace(0, 45, 1273)
    hf_t_vec = hf_t_vec[0:1200]
    
    sensors_locations = [
        '04', '10', '13', '14', '15', '17', '19', '21'
    ]
    plt.figure(figsize=(20,10))
    for i in range(8):

        plt.subplot(4, 2, i + 1)
        plt.title(f'Observation {obs_loc[i]} m')
        #plt.plot(obs_t_vec, true_observations[:, i+1], '--', color='black', label='True', linewidth=2, markersize=20)
        plt.plot(hf_t_vec, true_observations[:, i], '--', color='black', label='True', linewidth=2, markersize=20)
        #plt.plot(obs_times, true_observations[obs_times_ids, i], '.', color='tab:red', markersize=15)
        plt.plot(hf_t_vec, true_solution.observations[i, :], '--', color='tab:red')        
        plt.xlim([31, 41])
    plt.xlabel('Time [s]')
    plt.ylabel('Wave Height')
    plt.legend()
    plt.grid()
    plt.show()
    












    return 0


if __name__ == '__main__':
    main()