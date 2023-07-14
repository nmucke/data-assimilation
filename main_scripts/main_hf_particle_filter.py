import importlib
import ray
import yaml
import pdb
import numpy as np
from matplotlib import pyplot as plt

from data_assimilation.particle_filter.particle_filter import VanillaParticleFilter
from data_assimilation.particle_filter.model_error import PDEModelError
from data_assimilation.particle_filter.likelihood import PDELikelihood
from data_assimilation.particle_filter.observation_operator import PDEObservationOperator
from data_assimilation.true_solution import TrueSolution

TEST_CASE = 'multi_phase_pipeflow_with_leak'
TRUE_SOLUTION_PATH = f'data/{TEST_CASE}/test'

DISTRIBUTED = True

PDEForwardModel = importlib.import_module(
    f'data_assimilation.test_cases.{TEST_CASE}.forward_model'
).PDEForwardModel

# Load config file.
with open(f'configs/{TEST_CASE}.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def main():

    # Initialize observation operator.
    observation_operator = PDEObservationOperator(
        **config['observation_operator_args'],
    )

    # Initialize true solution.
    true_state = np.load(f'{TRUE_SOLUTION_PATH}/state/sample_7.npy')
    true_pars = np.load(f'{TRUE_SOLUTION_PATH}/pars/sample_7.npy')

    true_solution = TrueSolution(
        true_state=true_state,
        true_pars=true_pars,
        observation_operator=observation_operator,  
        **config['observation_args'],
    )

    if config['model_type'] == 'PDE':

        # Initialize forward model.
        forward_model = PDEForwardModel(
            **config['forward_model_args'],
            distributed=DISTRIBUTED,
        )
    
        # Initialize likelihood.
        likelihood = PDELikelihood(
            observation_operator=observation_operator,
            **config['likelihood_args'],
        )

        # Initialize model error.
        model_error = PDEModelError(
            **config['model_error_args'],
        )

    # Initialize particle filter.
    particle_filter = VanillaParticleFilter(
        particle_filter_args=config['particle_filter_args'],
        forward_model=forward_model,
        observation_operator=observation_operator,
        likelihood=likelihood,
        model_error=model_error,
    )

    init_pars = np.random.uniform(
        low=config['prior_pars']['lower_bound'],
        high=config['prior_pars']['upper_bound'],
        size=(particle_filter.num_particles, config['prior_pars']['num_pars']),
    )
    state_ensemble, pars_ensemble = particle_filter.compute_filtered_solution(
        true_solution=true_solution,
        init_pars=init_pars,
        transform_state=True,
        save_level=1,
    )
        
    state_ensemble = forward_model.transform_state(
        state_ensemble[:, :, :, -1],
        x_points=observation_operator.full_space_points,
        )
    #state_ensemble = np.mean(state_ensemble, axis=0)

    plt.figure()
    for i in range(state_ensemble.shape[0]):
        plt.plot(state_ensemble[i, 1], linewidth=1., color='tab:blue')
    plt.plot(true_state[1, :, 100], linewidth=3., color='black')
    plt.show()


if __name__ == '__main__':

    if DISTRIBUTED:
        ray.shutdown()
        ray.init(num_cpus=30)
    main()

    if DISTRIBUTED:
        ray.shutdown()