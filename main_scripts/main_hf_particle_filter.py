import importlib
import ray
import yaml
import pdb
import numpy as np
from matplotlib import pyplot as plt

from data_assimilation.plotting_utils import (
    plot_parameter_results, 
    plot_state_results
)
from data_assimilation.oracle import ObjectStorageClientWrapper
from data_assimilation.particle_filter.particle_filter import BootstrapFilter
from data_assimilation.particle_filter.model_error import (
    PDEModelError,
    NeuralNetworkModelError,
)
from data_assimilation.particle_filter.likelihood import (
    NeuralNetworkLikelihood, 
    PDELikelihood
)
from data_assimilation.particle_filter.observation_operator import (
    PDEObservationOperator,
    LatentObservationOperator,
)
from data_assimilation.true_solution import TrueSolution
from data_assimilation.utils import create_directory

PHASE = 'single'

MODEL_TYPE = 'PDE'
TEST_CASE = f'{PHASE}_phase_pipeflow_with_leak'
TRUE_SOLUTION_PATH = f'data/{TEST_CASE}/test'

DISTRIBUTED = True
NUM_WORKERS = 64

ORACLE_PATH = f'{PHASE}_phase/raw_data/test'

DEVICE = 'cpu'

SAVE_LOCAL_OR_ORACLE = 'local'
BUCKET_NAME = 'data_assimilation_results'

SAVE_LEVEL = 0

if MODEL_TYPE == 'neural_network':
    NNForwardModel = importlib.import_module(
        f'data_assimilation.test_cases.{TEST_CASE}.NN_forward_model'
    ).NNForwardModel
elif MODEL_TYPE == 'PDE':
    PDEForwardModel = importlib.import_module(
        f'data_assimilation.test_cases.{TEST_CASE}.PDE_forward_model'
    ).PDEForwardModel

# Load config file.
CONFIG_PATH = f'configs/{MODEL_TYPE}s/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if SAVE_LOCAL_OR_ORACLE == 'local':
    LOCAL_SAVE_PATH = f'results/{TEST_CASE}_{config["particle_filter_args"]["num_particles"]}'
    create_directory(LOCAL_SAVE_PATH)
    
elif SAVE_LOCAL_OR_ORACLE == 'oracle':
    ORACLE_SAVE_PATH = f'{TEST_CASE}_{config["particle_filter_args"]["num_particles"]}'


def main():

    # Initialize observation operator.
    if config['model_type'] == 'neural_network':
        observation_operator = LatentObservationOperator(
            **config['observation_operator_args'],
        )
    elif config['model_type'] == 'PDE':
        observation_operator = PDEObservationOperator(
            **config['observation_operator_args'],
        )
    

    bucket_name = "bucket-20230222-1753"

    object_storage_client = ObjectStorageClientWrapper(bucket_name)

    test_case_index = 0
    # Initialize true solution.
    true_state = object_storage_client.get_numpy_object(
        source_path=f'{ORACLE_PATH}/state/sample_{test_case_index}.npz'
    )
    
    true_pars = object_storage_client.get_numpy_object(
        source_path=f'{ORACLE_PATH}/pars/sample_{test_case_index}.npz'
    )

    true_solution = TrueSolution(
        state=true_state,
        pars=true_pars,
        observation_operator=observation_operator,  
        **config['observation_args'],
    )

    if config['model_type'] == 'neural_network':

        # Initialize forward model.
        forward_model = NNForwardModel(
            **config['forward_model_args'],
            distributed=DISTRIBUTED,
            device=DEVICE,
            num_particles=config['particle_filter_args']['num_particles'],
        )
        # Initialize likelihood.
        likelihood = NeuralNetworkLikelihood(
            observation_operator=observation_operator,
            **config['likelihood_args'],
        )
        # Initialize model error.
        model_error = NeuralNetworkModelError(
            **config['model_error_args'],
        )
    elif config['model_type'] == 'PDE':

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
    particle_filter = BootstrapFilter(
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
    num_previous_steps = 1 
    if config['model_type'] == 'neural_network':
        num_previous_steps =\
              config['forward_model_args']['model_args']['num_previous_steps']
        
    state_ensemble, pars_ensemble = particle_filter.compute_filtered_solution(
        true_solution=true_solution,
        init_pars=init_pars,
        transform_state=True,
        num_previous_steps=num_previous_steps,
        save_level=SAVE_LEVEL,
        model_type=config['model_type'],
    )
    
    state_ensemble_save = np.zeros(
        (
            state_ensemble.shape[0], 
            state_ensemble.shape[1], 
            config['observation_operator_args']['full_space_points'][-1],
            state_ensemble.shape[-1], 
        )
    )
    for time_idx in range(state_ensemble.shape[-1]):
        state_ensemble_save[:, :, :, time_idx] = forward_model.transform_state(
            state_ensemble[:, :, :, time_idx],
            x_points=observation_operator.full_space_points,
            pars=pars_ensemble[:, :, -1],
        )

    if SAVE_LOCAL_OR_ORACLE == 'oracle':

        object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)

        object_storage_client.put_numpy_object(
            data=state_ensemble_save,
            destination_path=f'{ORACLE_SAVE_PATH}/states.npz',
        )
        object_storage_client.put_numpy_object(
            data=pars_ensemble,
            destination_path=f'{ORACLE_SAVE_PATH}/pars.npz',
        )

        plot_state_results(
            state_ensemble=state_ensemble_save,
            true_solution=true_solution,
            save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
            object_storage_client=object_storage_client,
            num_states_to_plot = 3 if PHASE == 'multi' else 2,
        )

        plot_parameter_results(
            pars_ensemble=pars_ensemble,
            true_solution=true_solution,
            save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
            object_storage_client=object_storage_client,
        )

    elif SAVE_LOCAL_OR_ORACLE == 'local':
        np.savez_compressed(f'{LOCAL_SAVE_PATH}/states.npz', data=state_ensemble_save)
        np.savez_compressed(f'{LOCAL_SAVE_PATH}/pars.npz', data=pars_ensemble)

        plot_state_results(
            state_ensemble=state_ensemble_save,
            true_solution=true_solution,
            save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
            num_states_to_plot = 3 if PHASE == 'multi' else 2,
        )

        plot_parameter_results(
            pars_ensemble=pars_ensemble,
            true_solution=true_solution,
            save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
        )


    
if __name__ == '__main__':

    if DISTRIBUTED:
        ray.shutdown()
        ray.init(num_cpus=NUM_WORKERS)
    main()

    if DISTRIBUTED:
        ray.shutdown()