import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    entropy,
    wasserstein_distance
)
from geomloss import SamplesLoss
import torch
Loss =  SamplesLoss("sinkhorn", blur=0.05,)

TEST_CASE = 'multi_phase_pipeflow_with_leak'

HIGH_FIDELITY_MODEL_TYPE = 'PDE'
HIGH_FIDELITY_NUM_PARTICLES = 10000
HIGH_FIDELITY_LOAD_PATH = f'results/{TEST_CASE}_{HIGH_FIDELITY_MODEL_TYPE}_{HIGH_FIDELITY_NUM_PARTICLES}'

LOW_FIDELITY_MODEL_TYPES = ['latent']
LOW_FIDELITY_NUM_PARTICLES_LIST = [50, 100, 1000, 5000, 10000]

TEST_CASE_INDEX_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def main():

    # Load high fidelity results.
    high_fidelity_state = np.load(f'{HIGH_FIDELITY_LOAD_PATH}/states.npz')
    high_fidelity_state = high_fidelity_state['data']
    high_fidelity_state = torch.from_numpy(high_fidelity_state)
    high_fidelity_state = high_fidelity_state.type(torch.float32) 

    high_fidelity_pars = np.load(f'{HIGH_FIDELITY_LOAD_PATH}/pars.npz')
    high_fidelity_pars = high_fidelity_pars['data']

    high_fidelity_pars = torch.from_numpy(high_fidelity_pars)
    high_fidelity_pars = high_fidelity_pars.type(torch.float32)

    divergence_state_results = {} 
    divergence_pars_results = {} 
    for model_type in LOW_FIDELITY_MODEL_TYPES: 
        divergence_state_results[model_type] = []
        divergence_pars_results[model_type] = []
        for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:

            divergence_state = 0.
            divergence_pars = 0.
            for test_case_index in TEST_CASE_INDEX_LIST:

                LOW_FIDELITY_LOAD_PATH = f'results/{model_type}/{TEST_CASE}_{num_particles}_test_case_{test_case_index}'
                low_fidelity_state = np.load(f'{LOW_FIDELITY_LOAD_PATH}/states.npz')
                low_fidelity_state = low_fidelity_state['data']
                low_fidelity_state = torch.from_numpy(low_fidelity_state)
                low_fidelity_state = low_fidelity_state.type(torch.float32)

                divergence_state += Loss(high_fidelity_state[:, -1, :, -1], low_fidelity_state[:, -1, :, -1])


                low_fidelity_pars = np.load(f'{LOW_FIDELITY_LOAD_PATH}/pars.npz')
                low_fidelity_pars = low_fidelity_pars['data']
                low_fidelity_pars = torch.from_numpy(low_fidelity_pars)
                low_fidelity_pars = low_fidelity_pars.type(torch.float32)

                divergence_pars += wasserstein_distance(high_fidelity_pars[:, 0, -1], low_fidelity_pars[:, 0, -1])
                #divergence_pars = Loss(high_fidelity_pars[:, 0, -1:], low_fidelity_pars[:, 0, -1:])

            divergence_state_results[model_type].append(divergence_state/len(TEST_CASE_INDEX_LIST))
            divergence_pars_results[model_type].append(divergence_pars/len(TEST_CASE_INDEX_LIST))


    
    plt.figure()
    for model_type in LOW_FIDELITY_MODEL_TYPES:
        plt.loglog(LOW_FIDELITY_NUM_PARTICLES_LIST, divergence_state_results[model_type], '.-', label=model_type, linewidth=2, markersize=20)
    plt.xlabel('Number of Particles')
    plt.ylabel('Sinkhorn Distance')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure()
    for model_type in LOW_FIDELITY_MODEL_TYPES:
        plt.loglog(LOW_FIDELITY_NUM_PARTICLES_LIST, divergence_pars_results[model_type], '.-', label=model_type, linewidth=2, markersize=20)
    plt.xlabel('Number of Particles')
    plt.ylabel('Sinkhorn Distance')
    plt.legend()
    plt.grid()
    plt.show()












    return 0


if __name__ == '__main__':
    main()