import importlib
import ray
from tqdm import tqdm
import yaml
import pdb
import numpy as np
from matplotlib import pyplot as plt
import torch
from data_assimilation.particle_filter.NN_forward_model import NNForwardModel
from data_assimilation.particle_filter.stein_filter import SteinFilter
from data_assimilation.particle_filter.ml_bootstrap_filter import MLBootstrapFilter

from data_assimilation.plotting_utils import (
    plot_observation_results,
    plot_parameter_results, 
    plot_state_results
)
from data_assimilation.oracle import ObjectStorageClientWrapper
from data_assimilation.particle_filter.bootstrap_filter import BootstrapFilter
from data_assimilation.particle_filter.model_error import (
    PDEModelError,
    LatentModelError,
)
from data_assimilation.particle_filter.likelihood import (
    Likelihood
)
from data_assimilation.particle_filter.observation_operator import (
    ObservationOperator
)
from data_assimilation.true_solution import TrueSolution
from data_assimilation.utils import create_directory
from data_assimilation.particle_filter.latent_forward_model import LatentForwardModel

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

particle_filter_type = 'bootstrap'

PHASE = 'multi'
TEST_CASE = 'multi_phase_pipeflow_with_leak'#'wave_submerged_bar'#''lorenz_96'#

MODEL_TYPE = 'PDE'

DISTRIBUTED = True
NUM_WORKERS = 25

TEST_DATA_FROM_ORACLE_OR_LOCAL = 'oracle'
ORACLE_PATH = f'{PHASE}_phase/raw_data/test'
LOCAL_PATH = f'data/{TEST_CASE}/test'

DEVICE = 'cuda'

SAVE_LOCAL_OR_ORACLE = 'oracle'
BUCKET_NAME = 'data_assimilation_results'

SAVE_LEVEL = 0

if MODEL_TYPE == 'PDE':
    PDEForwardModel = importlib.import_module(
        f'data_assimilation.test_cases.{TEST_CASE}.PDE_forward_model'
    ).PDEForwardModel

# Load config file.
CONFIG_PATH = f'configs/{MODEL_TYPE}/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if SAVE_LOCAL_OR_ORACLE == 'local':
    LOCAL_SAVE_PATH = f'results/{TEST_CASE}_{MODEL_TYPE}_{config["particle_filter_args"]["num_particles"]}'
    create_directory(LOCAL_SAVE_PATH)
    
elif SAVE_LOCAL_OR_ORACLE == 'oracle':
    ORACLE_SAVE_PATH = f'{TEST_CASE}_{MODEL_TYPE}_{config["particle_filter_args"]["num_particles"]}'


def main():

    # Initialize observation operator
    observation_operator = ObservationOperator(
            **config['observation_operator_args'],
            backend=config['backend'],
        )
    
    test_case_index = 4

    # Initialize true solution.
    if TEST_DATA_FROM_ORACLE_OR_LOCAL == 'oracle':

        bucket_name = "bucket-20230222-1753"
        object_storage_client = ObjectStorageClientWrapper(bucket_name)
        true_state = object_storage_client.get_numpy_object(
            source_path=f'{ORACLE_PATH}/state/sample_{test_case_index}.npz'
        )
        true_pars = object_storage_client.get_numpy_object(
            source_path=f'{ORACLE_PATH}/pars/sample_{test_case_index}.npz'
        )
    else:
        true_state = np.load(
            f'{LOCAL_PATH}/state/sample_{test_case_index}.npz'
        )['data']
        true_pars = np.load(
            f'{LOCAL_PATH}/pars/sample_{test_case_index}.npz'
        )['data']

    true_solution = TrueSolution(
        state=true_state,
        pars=true_pars,
        observation_operator=observation_operator,  
        backend=config['backend'],
        **config['observation_args'],
    )

    if config['model_type'] == 'latent':
        
        # Initialize forward model.
        forward_model = LatentForwardModel(
            **config['forward_model_args'],
            distributed=DISTRIBUTED,
            device=DEVICE,
            num_particles=config['particle_filter_args']['num_particles'],
        )
        
        # Initialize model error.
        model_error = LatentModelError(
            **config['model_error_args'],
            latent_dim=forward_model.latent_dim,
        )

    elif config['model_type'] == 'FNO':

        # Initialize forward model.
        forward_model = NNForwardModel(
            **config['forward_model_args'],
            distributed=DISTRIBUTED,
            device=DEVICE,
            num_particles=config['particle_filter_args']['num_particles'],
        )
        
        # Initialize model error.
        model_error = PDEModelError(
            **config['model_error_args'],
            space_dim=config['forward_model_args']['space_dim']
        )

    elif config['model_type'] == 'PDE':

        # Initialize forward model.
        forward_model = PDEForwardModel(
            **config['forward_model_args'],
            distributed=DISTRIBUTED,
        )

        # Initialize model error.
        model_error = PDEModelError(
            **config['model_error_args'],
            space_dim=\
                config['forward_model_args']['model_args']['basic_args']['num_elements']*\
                (config['forward_model_args']['model_args']['basic_args']['polynomial_order']+1),
        )
    

    # Initialize likelihood.
    likelihood = Likelihood(
        observation_operator=observation_operator,
        backend=config['backend'],
        **config['likelihood_args'],
    )

    particle_filter_input = {
        'particle_filter_args': config['particle_filter_args'],
        'forward_model': forward_model,
        'observation_operator': observation_operator,
        'likelihood': likelihood,
        'model_error': model_error,
        'backend': config['backend'],
    }

    # Initialize particle filter.
    if particle_filter_type == 'bootstrap':
        particle_filter = BootstrapFilter(
            **particle_filter_input,
        )
    elif particle_filter_type == 'stein':
        particle_filter = SteinFilter(
            **particle_filter_input,
        )
    elif particle_filter_type == 'ml_bootstrap':
        particle_filter = MLBootstrapFilter(
            **particle_filter_input,
        )
        
    init_pars = np.random.uniform(
        low=config['prior_pars']['lower_bound'],
        high=config['prior_pars']['upper_bound'],
        size=(particle_filter.num_particles, config['prior_pars']['num_pars']),
    )
    
    state_ensemble, pars_ensemble, state_observations = particle_filter.compute_filtered_solution(
        true_solution=true_solution,
        init_pars=init_pars,
        save_level=SAVE_LEVEL,
    )
    
    state_ensemble_save = np.zeros(
        (
            state_ensemble.shape[0], 
            config['forward_model_args']['num_PDE_states'], 
            config['observation_operator_args']['full_space_points'][-1],
            state_ensemble.shape[-1], 
        )
    )
    pars_ensemble_save = np.zeros(
        (
            pars_ensemble.shape[0], 
            pars_ensemble.shape[1], 
            pars_ensemble.shape[-1], 
        )
    )

    print('Transforming state ensemble...')
    with torch.no_grad():
        pbar = tqdm(
            range(state_ensemble.shape[-1]),
            total=state_ensemble.shape[-1],
            bar_format = "{desc}: {percentage:.2f}%|{bar:20}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"#
            )
        for time_idx in pbar:
            state_ensemble_save[:, :, :, time_idx] = \
                forward_model.transform_state(
                        state_ensemble[:, :, :, time_idx] if config['model_type'] in ['PDE', 'FNO'] else\
                        state_ensemble[:, :, time_idx].unsqueeze(-1),
                    x_points=observation_operator.full_space_points,
                    pars=pars_ensemble[:, :, time_idx]
                )

    print('Transforming pars ensemble...')
    pbar = tqdm(
        range(pars_ensemble.shape[-1]),
        total=pars_ensemble.shape[-1],
        bar_format = "{desc}: {percentage:.2f}%|{bar:20}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"#
        )
    for time_idx in pbar:
        if config['model_type'] == 'latent':
            pars_ensemble_save[:, :, time_idx] = forward_model.transform_pars(
                pars_ensemble[:, :, time_idx]                                                                                                                      
            )
        else:
            pars_ensemble_save[:, :, time_idx] = pars_ensemble[:, :, time_idx]

    RMSE = np.sqrt(np.mean((state_ensemble_save[:, :, :, -1] - true_solution.state[:, :, -1])**2))
    print(f'RMSE: {RMSE}')

    print('Saving results...')
    if SAVE_LOCAL_OR_ORACLE == 'oracle':

        object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)

        object_storage_client.put_numpy_object(
            data=state_ensemble_save,
            destination_path=f'{ORACLE_SAVE_PATH}/states.npz',
        )
        object_storage_client.put_numpy_object(
            data=pars_ensemble_save,
            destination_path=f'{ORACLE_SAVE_PATH}/pars.npz',
        )
        object_storage_client.put_numpy_object(
            data=state_observations,
            destination_path=f'{ORACLE_SAVE_PATH}/state_observations.npz',
        )

    elif SAVE_LOCAL_OR_ORACLE == 'local':
        np.savez_compressed(f'{LOCAL_SAVE_PATH}/states.npz', data=state_ensemble_save)
        np.savez_compressed(f'{LOCAL_SAVE_PATH}/pars.npz', data=pars_ensemble_save)
        np.savez_compressed(f'{LOCAL_SAVE_PATH}/state_observations.npz', data=state_observations)

    print('Plotting results...')
    plot_state_results(
        state_ensemble=state_ensemble_save,
        true_solution=true_solution,
        save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
        object_storage_client=object_storage_client if SAVE_LOCAL_OR_ORACLE == 'oracle' else None,
        plotting_args=config['plotting_args'],
    )

    plot_parameter_results(
        pars_ensemble=pars_ensemble_save,
        true_solution=true_solution,
        save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
        object_storage_client=object_storage_client if SAVE_LOCAL_OR_ORACLE == 'oracle' else None,
        plotting_args=config['plotting_args'],
    )
    plot_observation_results(
        obs_ensemble=state_observations,
        true_solution= true_solution,
        save_path=ORACLE_SAVE_PATH if SAVE_LOCAL_OR_ORACLE == 'oracle' else LOCAL_SAVE_PATH,
        object_storage_client=object_storage_client if SAVE_LOCAL_OR_ORACLE == 'oracle' else None,
        plotting_args=config['plotting_args'],
    )


    
if __name__ == '__main__':

    if DISTRIBUTED:
        ray.shutdown()
        ray.init(num_cpus=NUM_WORKERS)
    main()

    if DISTRIBUTED:
        ray.shutdown()