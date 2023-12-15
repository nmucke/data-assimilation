import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import torch
import yaml
from scipy.stats import wasserstein_distance, entropy
from geomloss import SamplesLoss
import torch

from data_assimilation.oracle import ObjectStorageClientWrapper
Loss =  SamplesLoss("sinkhorn", blur=0.05,)

TEST_CASE = 'multi_phase_pipeflow_with_leak'

TEST_NUM_LIST = [4]

MODEL_TYPE_LIST = ['latent', 'latent_AE_no_reg', 'latent_WAE_conv']
LOW_FIDELITY_NUM_PARTICLES_LIST = [100, 500, 1000, 5000, 10000]

PLOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

ORACLE_PATH = f'PDE/multi_phase_pipeflow_with_leak_3000_test_case'
BUCKET_NAME = 'data_assimilation_results'


def main():

    # Load high fidelity results.
    #true_state = np.load(f'results/PDE/{TEST_CASE}_10000_test_case_{TEST_NUM}/states.npz')
    #true_state = true_state['data']

    #true_pars = np.load(f'results/PDE/{TEST_CASE}_10000_test_case_{TEST_NUM}/pars.npz')
    #true_pars = true_pars['data']
    #true_pars = torch.tensor(true_pars, dtype=torch.float32)

    # prepare result dictionaries
    pred_dict = {}
    for model_type in MODEL_TYPE_LIST:
        pred_dict[model_type] = {}
        for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:
            pred_dict[model_type][num_particles] = {
                'wasserstein_pars_0': 0,
                'wasserstein_pars_1': 0,
                'RRMSE_pars_0': 0,
                'RMSE_state': 0,
            }
    #plt.figure()

    percentiles_to_test = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]

    hf_pars_dict = {}
    true_pars_dict = {}
    for test_num in TEST_NUM_LIST:

        bucket_name = "bucket-20230222-1753"
        object_storage_client = ObjectStorageClientWrapper(bucket_name)
        true_pars = object_storage_client.get_numpy_object(
            source_path=f'multi_phase/raw_data/test/pars/sample_{test_num}.npz'
        )
        true_pars = torch.tensor(true_pars, dtype=torch.float32)
        true_pars_dict[test_num] = {
            'leak_location': true_pars[0],
        }

        object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)
        hf_pars = object_storage_client.get_numpy_object(
            source_path=f'{ORACLE_PATH}_{test_num}/pars.npz'
        )
        hf_pars = torch.tensor(hf_pars, dtype=torch.float32)
        hf_pars_dict[test_num] = {
            'leak_location': hf_pars[:, 0, 0],
        }

        hf_state = object_storage_client.get_numpy_object(
            source_path=f'{ORACLE_PATH}_{test_num}/states.npz'
        )
        hf_state = torch.tensor(hf_state, dtype=torch.float32)

        hf_state_percentiles = np.percentile(hf_state, percentiles_to_test, axis=0)    
        hf_state_percentiles = torch.tensor(hf_state_percentiles, dtype=torch.float32)
        

        for model_type in MODEL_TYPE_LIST:
            for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:

                LOW_FIDELITY_LOAD_PATH = f'results/{model_type}/{TEST_CASE}_{num_particles}_test_case_{test_num}'

                pred_state = np.load(f'{LOW_FIDELITY_LOAD_PATH}/states.npz')
                pred_state = pred_state['data']
                #pred_state = torch.tensor(pred_state, dtype=torch.float32)
                
                pred_pars = np.load(f'{LOW_FIDELITY_LOAD_PATH}/pars.npz')
                pred_pars = pred_pars['data']
                pred_pars = torch.tensor(pred_pars, dtype=torch.float32)

                # remove samples that are more than 5 standard deviations away from the median
                pred_pars = pred_pars[:, 0, 0][torch.abs(pred_pars[:, 0, 0] - pred_pars[:, 0, 0].median()) < 5 * pred_pars[:, 0, 0].std()]
                pred_pars = pred_pars.unsqueeze(1)
                
                #pred_state = np.load(f'{LOW_FIDELITY_LOAD_PATH}/states.npz')
                #pred_state = pred_state['data']

                pred_dict[model_type][num_particles]['leak_location'] = pred_pars[:,0]             

                #pred_dict[model_type][num_particles]['wasserstein_pars_0'] += \
                #    1/len(TEST_NUM_LIST) * Loss(hf_pars[:, 0, 0:1], pred_pars)
                pred_dict[model_type][num_particles]['wasserstein_pars_0'] += \
                    1/len(TEST_NUM_LIST) * wasserstein_distance(hf_pars[:, 0, 0], pred_pars[:,0])

                #pred_dict[model_type][num_particles]['wasserstein_pars_1'] += \
                #    1/len(TEST_NUM_LIST) * Loss(hf_pars[:, 1, 0:1], pred_pars[:, 1, 0:1])
                
                pred_dict[model_type][num_particles]['RRMSE_pars_0'] += \
                    1/len(TEST_NUM_LIST) * torch.sqrt((true_pars[0] - pred_pars.mean())**2) / true_pars[0]
                
                #pred_dict[model_type][num_particles]['RRMSE_pars_1'] += \
                #    1/len(TEST_NUM_LIST) * torch.sqrt((true_pars[1] - pred_pars[:, 1, 0:1].mean())**2) / true_pars[1]

                # Get the percentiles of the states
                pred_state = np.percentile(pred_state, percentiles_to_test, axis=0)    
                pred_state = torch.tensor(pred_state, dtype=torch.float32)

                # Calculate the RMSE distance between the percentiles of the states
                pred_dict[model_type][num_particles]['RMSE_state'] += \
                    1/len(TEST_NUM_LIST) * torch.sqrt(torch.mean((hf_state_percentiles - pred_state)**2))

                #plt.hist(pred_pars[:, 0, 0], bins=100, alpha=0.5, label=f'{model_type} {num_particles} particles', density=True)
    #plt.hist(true_pars[:, 0, 0], bins=100, alpha=0.5, label='True', density=True)
    #plt.legend()
    #plt.show()

            
    
    wasserstein_distance_0_dict = {}
    wasserstein_distance_1_dict = {}
    state_RMSE_dict = {}
    pars_RMSE_dict = {}
    for model_type in MODEL_TYPE_LIST:
        wasserstein_distance_0_dict[model_type] = []
        wasserstein_distance_1_dict[model_type] = []
        state_RMSE_dict[model_type] = []
        pars_RMSE_dict[model_type] = []
        for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:
            wasserstein_distance_0_dict[model_type].append(pred_dict[model_type][num_particles]['wasserstein_pars_0'])
            #wasserstein_distance_1_dict[model_type].append(pred_dict[model_type][num_particles]['wasserstein_pars_1'])
            state_RMSE_dict[model_type].append(pred_dict[model_type][num_particles]['RMSE_state'])
            pars_RMSE_dict[model_type].append(pred_dict[model_type][num_particles]['RRMSE_pars_0'])
    
    plt.figure(figsize=(30,10))
    plt.subplot(1, 3, 1)
    for i, model_type in enumerate(MODEL_TYPE_LIST):
        plt.loglog(LOW_FIDELITY_NUM_PARTICLES_LIST, wasserstein_distance_0_dict[model_type], '.-', linewidth=3, markersize=20, color=PLOT_COLORS[i], label=f'{model_type}')
    plt.xlabel('Number of Particles')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    for i, model_type in enumerate(MODEL_TYPE_LIST):
        plt.loglog(LOW_FIDELITY_NUM_PARTICLES_LIST, state_RMSE_dict[model_type], '.-', linewidth=3, markersize=20, color=PLOT_COLORS[i], label=f'{model_type}')
    plt.xlabel('Number of Particles')
    plt.ylabel('RMSE Percentile Distance')
    plt.grid()

    plt.subplot(1, 3, 3)
    for i, model_type in enumerate(MODEL_TYPE_LIST):
        plt.loglog(LOW_FIDELITY_NUM_PARTICLES_LIST, pars_RMSE_dict[model_type], '.-', linewidth=3, markersize=20, color=PLOT_COLORS[i], label=f'{model_type}')
    plt.xlabel('Number of Particles')
    plt.ylabel('RMSE Pars')
    plt.grid()

    plt.savefig('results/figures/multi_phase_leak_convergence.png')
    plt.show()



    '''
    for i, model_type in enumerate(MODEL_TYPE_LIST):
        plt.loglog(LOW_FIDELITY_NUM_PARTICLES_LIST, wasserstein_distance_1_dict[model_type], '.-', linewidth=3, markersize=20, color=PLOT_COLORS[i], label=f'{model_type}')
    plt.xlabel('Number of Particles')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid()
    plt.savefig('results/figures/multi_phase_size_convergence.png')

    plt.show()
    '''

    plt.figure(figsize=(20,10))
    for i, model_type in enumerate(MODEL_TYPE_LIST):
        for j, num_particles in enumerate(LOW_FIDELITY_NUM_PARTICLES_LIST):
            plt.hist(pred_dict[model_type][num_particles]['leak_location'], bins=100, alpha=0.1, label=f'{model_type} {num_particles} particles', density=True)

            kde = stats.gaussian_kde(pred_dict[model_type][num_particles]['leak_location'])
            x = np.linspace(
                pred_dict[model_type][num_particles]['leak_location'].min(), 
                pred_dict[model_type][num_particles]['leak_location'].max(), 
                1000
            )
            plt.plot(x, kde(x), linewidth=3, color=PLOT_COLORS[j])

            plt.axvline(x=pred_dict[model_type][num_particles]['leak_location'].mean(), color=PLOT_COLORS[j], linestyle='--')



    plt.hist(hf_pars_dict[8]['leak_location'], bins=100, alpha=0.1, label='HF 10000 particles', density=True, color=PLOT_COLORS[-1])
    kde = stats.gaussian_kde(hf_pars_dict[8]['leak_location'])
    x = np.linspace(
        hf_pars_dict[8]['leak_location'].min(), 
        hf_pars_dict[8]['leak_location'].max(), 
        1000
    )
    plt.plot(x, kde(x), linewidth=3, color=PLOT_COLORS[-1])
    plt.axvline(x=hf_pars_dict[8]['leak_location'].mean(), color=PLOT_COLORS[-1], linestyle='--')
    
    plt.axvline(x=true_pars_dict[8]['leak_location'], color='k', linestyle='--', label='True')
    plt.xlim([true_pars_dict[8]['leak_location']-200, true_pars_dict[8]['leak_location']+200])
    plt.legend()
    plt.show()






    '''
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
    '''
    












    return 0


if __name__ == '__main__':
    main()