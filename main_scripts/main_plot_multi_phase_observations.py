import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

TEST_CASE = 'multi_phase_pipeflow_with_leak'

MODEL_TYPE = 'latent'
LOW_FIDELITY_NUM_PARTICLES_LIST = [1000, 2500, 5000]

CONFIG_PATH = f'configs/{MODEL_TYPE}/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PLOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

def main():

    # Load high fidelity results.

    true_observations = pd.read_csv(
        'data/wave_submerged_bar/observations/observations.csv', 
    )
    
    zeroes = np.zeros(
        shape=(17, true_observations.shape[1])
    )
    true_observations = np.concatenate(
        (zeroes, true_observations), axis=0
    )
    t_vec = true_observations[:, 0]
    t_vec[0:17] = np.linspace(0, 0.615573, 17)
    true_observations = true_observations[:, 1:]

    obs_times_ids = range(
        config['observation_args']['observation_times'][0],
        config['observation_args']['observation_times'][1],
        config['observation_args']['observation_times'][2],
    )

    pred_observations_dict = {}
    for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:
        pred_observations_dict[num_particles] = {}

        LOW_FIDELITY_LOAD_PATH = f'results/{TEST_CASE}_{MODEL_TYPE}'
        pred_observations = np.load(f'{LOW_FIDELITY_LOAD_PATH}_{num_particles}/observations.npz')
        pred_observations = pred_observations['data']

        pred_observations_dict[num_particles]['observations'] = pred_observations
        pred_observations_dict[num_particles]['observations_mean'] = pred_observations.mean(axis=0)
        pred_observations_dict[num_particles]['observations_std'] = pred_observations.std(axis=0)



    t_vec = t_vec[0:pred_observations.shape[-1]]
    true_observations = true_observations[0:pred_observations.shape[-1]]

    obs_times = t_vec[obs_times_ids]

    plt.figure(figsize=(20,10))
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        for j, num_particles in enumerate(LOW_FIDELITY_NUM_PARTICLES_LIST):
            plt.plot(t_vec, pred_observations_dict[num_particles]['observations_mean'][i], '-', linewidth=2, markersize=20, color=PLOT_COLORS[j], label=f'{num_particles} particles')
            plt.fill_between(
                t_vec,
                pred_observations_dict[num_particles]['observations_mean'][i] - 2 * pred_observations_dict[num_particles]['observations_std'][i],
                pred_observations_dict[num_particles]['observations_mean'][i] + 2 * pred_observations_dict[num_particles]['observations_std'][i],
                alpha=0.1,
                color=PLOT_COLORS[j],
            )
        plt.plot(t_vec+0.925, true_observations[:, i], '--', color='black', label='True', linewidth=2, markersize=20)
        plt.plot(obs_times+0.925, true_observations[obs_times_ids, i], '.', color='tab:red', markersize=15)
    plt.xlabel('Time [s]')
    plt.ylabel('Wave Height')
    plt.legend()
    plt.grid()
    plt.show()
    












    return 0


if __name__ == '__main__':
    main()