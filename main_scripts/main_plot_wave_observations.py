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
PARTICLE_FILTER_TYPE_LIST = ['bootstrap']
LOW_FIDELITY_NUM_PARTICLES_LIST = [50, 100, 250, 500, 1000, 2500, 5000]

CONFIG_PATH = f'configs/{MODEL_TYPE}/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PLOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

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

    x = np.linspace(0.05, 25.6, 512)
    t_vec = np.linspace(0, 45, 1273)

    obs_loc = [4, 10.5, 13.5, 14.5, 15.7, 17.3, 19.0, 21.0]
    '''
    lal = []
    for i in range(len(obs_loc)):
        lal.append((np.abs(x - obs_loc[i])).argmin())
    pdb.set_trace()
    '''

    # Load high fidelity results.

    true_observations = pd.read_csv(
        'data/wave_submerged_bar/observations/observations.csv', 
    ).values

    obs_times_ids = range(
        config['observation_args']['observation_times'][0],
        config['observation_args']['observation_times'][1],
        config['observation_args']['observation_times'][2],
    )

    pred_observations_dict = {}

    for particle_filter_type in PARTICLE_FILTER_TYPE_LIST:
        pred_observations_dict[particle_filter_type] = {}
        for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:
            pred_observations_dict[particle_filter_type][num_particles] = {}

            LOW_FIDELITY_LOAD_PATH = f'results/{MODEL_TYPE}/{TEST_CASE}_{num_particles}_test_case_0'
            pred_observations = np.load(f'{LOW_FIDELITY_LOAD_PATH}/observations.npz')
            pred_observations = pred_observations['data']

            true_observations_i = true_observations[0:pred_observations.shape[-1]].T

            pred_observations_dict[particle_filter_type][num_particles]['observations'] = pred_observations
            pred_observations_dict[particle_filter_type][num_particles]['observations_mean'] = pred_observations.mean(axis=0)
            pred_observations_dict[particle_filter_type][num_particles]['observations_std'] = pred_observations.std(axis=0)

            observation_RRMSE = np.sqrt(np.mean((pred_observations.mean(axis=0) - true_observations_i)**2))/np.sqrt(np.mean(true_observations_i**2))
            pred_observations_dict[particle_filter_type][num_particles]['observation_RRMSE'] = observation_RRMSE

            pred_observations_dict[particle_filter_type][num_particles]['0.025_quantile'] = np.quantile(pred_observations, 0.025, axis=0)
            pred_observations_dict[particle_filter_type][num_particles]['0.975_quantile'] = np.quantile(pred_observations, 0.975, axis=0)

            # prediction interval coverage probability
            pred_observations_dict[particle_filter_type][num_particles]['coverage'] = np.mean(
                (pred_observations_dict[particle_filter_type][num_particles]['0.025_quantile'] < true_observations_i) * \
                    (true_observations_i < pred_observations_dict[particle_filter_type][num_particles]['0.975_quantile'])
            )



            #pred_observations_dict[particle_filter_type][num_particles]['coverage'] = np.mean(
            #    (pred_observations.mean(axis=0) - 1.96 * pred_observations.std(axis=0) < true_observations_i) * (true_observations_i < pred_observations.mean(axis=0) + 1.96 * pred_observations.std(axis=0))
            #)
            pred_observations_dict[particle_filter_type][num_particles]['picp'] = pred_observations_dict[particle_filter_type][num_particles]['coverage']

            # standard deviation of state   
            pred_observations_dict[particle_filter_type][num_particles]['state_std'] = pred_observations.std(axis=0).mean()/np.sqrt(np.mean(true_observations_i))

            # get pars
            pars = np.load(f'{LOW_FIDELITY_LOAD_PATH}/pars.npz')
            pars = pars['data']

            # only take pars that within 3 std of the mean
            pars = pars[:, np.abs(pars.mean(axis=0) - true_pars) < 1*pars.std(axis=0)] 

            pred_observations_dict[particle_filter_type][num_particles]['pars'] = pars





    t_vec = t_vec[0:pred_observations.shape[-1]]
    true_observations = true_observations[0:pred_observations.shape[-1]]

    obs_times = t_vec[obs_times_ids]

    RRMSE_list = []
    PICP_list = []
    std_list = []
    for j, num_particles in enumerate(LOW_FIDELITY_NUM_PARTICLES_LIST):
        RRMSE_list.append(pred_observations_dict['bootstrap'][num_particles]['observation_RRMSE'])
        PICP_list.append(pred_observations_dict['bootstrap'][num_particles]['picp'])
        std_list.append(pred_observations_dict['bootstrap'][num_particles]['state_std'])


    plt.figure()
    plt.subplot(1, 4, 1)
    plt.loglog(
        LOW_FIDELITY_NUM_PARTICLES_LIST,
        RRMSE_list, 
        '.-', markersize=20, linewidth=3,
    )
    plt.xlabel('Number of particles')
    plt.ylabel('RRMSE')
    

    plt.subplot(1, 4, 2)
    plt.loglog(
        LOW_FIDELITY_NUM_PARTICLES_LIST,
        PICP_list,
        '.-', markersize=20, linewidth=3,
    )
    plt.xlabel('Number of particles')
    plt.ylabel('PICP')

    plt.subplot(1, 4, 3)
    plt.loglog(
        LOW_FIDELITY_NUM_PARTICLES_LIST,
        std_list,
        '.-', markersize=20, linewidth=3,
    )
    plt.xlabel('Number of particles')
    plt.ylabel('STD')

    plt.subplot(1, 4, 4)
    for i, num_particles in enumerate(LOW_FIDELITY_NUM_PARTICLES_LIST):
        plt.hist(pred_observations_dict['bootstrap'][num_particles]['pars'], bins=200, alpha=0.2, label=f'{num_particles} particles', density=True, color=PLOT_COLORS[i])
        plt.axvline(x=pred_observations_dict['bootstrap'][num_particles]['pars'].mean(), color=PLOT_COLORS[i], linestyle='-', linewidth=3)
    plt.legend()
    plt.axvline(x=0.1, color='k', linestyle='-')
    plt.xlim([0.07, 0.13])
    plt.ylim([0, 100])

    plt.legend()
    plt.xlabel('Number of particles')
    plt.ylabel('PICP')
    plt.show()

    plt.figure(figsize=(20,10))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        for k, particle_filter_type in enumerate(PARTICLE_FILTER_TYPE_LIST):
            for j, num_particles in enumerate(LOW_FIDELITY_NUM_PARTICLES_LIST):
                plt.plot(t_vec, pred_observations_dict[particle_filter_type][num_particles]['observations_mean'][i], '-', linewidth=2, markersize=20, color=PLOT_COLORS[j], label=f'{num_particles} particles, {particle_filter_type}')
                plt.fill_between(
                    t_vec,
                    #pred_observations_dict[particle_filter_type][num_particles]['observations_mean'][i] - 2 * pred_observations_dict[particle_filter_type][num_particles]['observations_std'][i],
                    #pred_observations_dict[particle_filter_type][num_particles]['observations_mean'][i] + 2 * pred_observations_dict[particle_filter_type][num_particles]['observations_std'][i],
                    pred_observations_dict[particle_filter_type][num_particles]['0.025_quantile'][i],
                    pred_observations_dict[particle_filter_type][num_particles]['0.975_quantile'][i],
                    alpha=0.1,
                    color=PLOT_COLORS[j],
                )
        plt.title(f'Observation {obs_loc[i]} m')
        plt.plot(t_vec, true_observations[:, i], '--', color='black', label='True', linewidth=3, markersize=20)
        plt.plot(obs_times, true_observations[obs_times_ids, i], '.', color='tab:red', markersize=15)
        #plt.plot(t_vec, true_solution.observations[i, 0:len(t_vec)], '--', color='tab:red')        
        plt.xlim([20, 37])
    plt.xlabel('Time [s]')
    plt.ylabel('Wave Height')
    plt.legend()
    plt.grid()
    plt.show()
    












    return 0


if __name__ == '__main__':
    main()