import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    entropy,
    wasserstein_distance
)

TEST_CASE = 'multi_phase_pipeflow_with_leak'

HIGH_FIDELITY_MODEL_TYPE = 'PDE'
HIGH_FIDELITY_NUM_PARTICLES = 100
HIGH_FIDELITY_LOAD_PATH = f'results/{TEST_CASE}_{HIGH_FIDELITY_MODEL_TYPE}_{HIGH_FIDELITY_NUM_PARTICLES}'

LOW_FIDELITY_MODEL_TYPES = ['latent']
LOW_FIDELITY_NUM_PARTICLES_LIST = [100, 1000]


def main():

    # Load high fidelity results.
    high_fidelity_state = np.load(f'{HIGH_FIDELITY_LOAD_PATH}/states.npz')
    high_fidelity_state = high_fidelity_state['data']

    high_fidelity_pars = np.load(f'{HIGH_FIDELITY_LOAD_PATH}/pars.npz')
    high_fidelity_pars = high_fidelity_pars['data']

    divergence_results = {} 
    for model_type in LOW_FIDELITY_MODEL_TYPES: 
        divergence_results[model_type] = []
        for num_particles in LOW_FIDELITY_NUM_PARTICLES_LIST:
            LOW_FIDELITY_LOAD_PATH = f'results/{TEST_CASE}_{model_type}'
            low_fidelity_state = np.load(f'{LOW_FIDELITY_LOAD_PATH}_{num_particles}/states.npz')
            low_fidelity_state = low_fidelity_state['data']

            low_fidelity_pars = np.load(f'{LOW_FIDELITY_LOAD_PATH}_{num_particles}/pars.npz')
            low_fidelity_pars = low_fidelity_pars['data']

            divergence = wasserstein_distance(high_fidelity_pars[:, 0, -1], low_fidelity_pars[:, 0, -1])

            divergence_results[model_type].append(divergence)


    
    
    plt.figure()
    for model_type in LOW_FIDELITY_MODEL_TYPES:
        divergence_list = divergence_results[model_type]
        plt.plot(LOW_FIDELITY_NUM_PARTICLES_LIST, divergence_results[model_type], '.-', label=model_type, linewidth=2, markersize=20)
    plt.xlabel('Number of Particles')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid()
    plt.show()












    return 0


if __name__ == '__main__':
    main()