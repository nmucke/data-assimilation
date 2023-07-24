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

MODEL_TYPE = 'neural_network'
TEST_CASE = 'single_phase_pipeflow_with_leak'
TRUE_SOLUTION_PATH = f'data/{TEST_CASE}/test'

DISTRIBUTED = True
NUM_WORKERS = 10

SAVE_LEVEL = 1

PDEForwardModel = importlib.import_module(
    f'data_assimilation.test_cases.{TEST_CASE}.PDE_forward_model'
).PDEForwardModel

# Load config file.
CONFIG_PATH = f'configs/PDEs/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

LOCAL_SAVE_PATH = f'data/{TEST_CASE}/initial_conditions'
create_directory(LOCAL_SAVE_PATH)
    
def main():

    # Initialize forward model.
    forward_model = PDEForwardModel(
        **config['forward_model_args'],
        distributed=DISTRIBUTED,
    )

    init_pars = np.random.uniform(
        low=config['prior_pars']['lower_bound'],
        high=config['prior_pars']['upper_bound'],
        size=(500, config['prior_pars']['num_pars']),
    )

    init_states, _ = forward_model.initialize_state(
        pars=init_pars
    )
    t_range = [0, 16*5*0.01]


    state, _ = forward_model.compute_forward_model(
        state_ensemble=init_states[:, :, :, 0],
        pars_ensemble=init_pars,
        t_range=t_range,
    )

    full_x_points = np.linspace(0, 2000, 256)
    state = state[:, :, : , ::5]

    state_transformed = np.zeros((state.shape[0], state.shape[1], len(full_x_points), state.shape[-1]))
    for t_idx in range(state.shape[-1]):
        state_transformed[:, :, :, t_idx] = \
            forward_model.transform_state(state[:, :, :, t_idx], full_x_points)

    np.savez_compressed(f'{LOCAL_SAVE_PATH}/states.npz', data=state_transformed)
    np.savez_compressed(f'{LOCAL_SAVE_PATH}/pars.npz', data=init_pars)


    
if __name__ == '__main__':

    if DISTRIBUTED:
        ray.shutdown()
        ray.init(num_cpus=NUM_WORKERS)
    main()

    if DISTRIBUTED:
        ray.shutdown()