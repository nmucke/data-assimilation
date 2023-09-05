import importlib
import ray
import yaml
import pdb
import numpy as np
from matplotlib import pyplot as plt

from data_assimilation.utils import create_directory

TEST_CASE = 'multi_phase_pipeflow_with_leak'
TRUE_SOLUTION_PATH = f'data/{TEST_CASE}/test'

DISTRIBUTED = True
NUM_WORKERS = 25

SAVE_LEVEL = 1

PDEForwardModel = importlib.import_module(
    f'data_assimilation.test_cases.{TEST_CASE}.PDE_forward_model'
).PDEForwardModel

# Load config file.
CONFIG_PATH = f'configs/PDEs/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load config file.
CONFIG_PATH = f'configs/neural_networks/{TEST_CASE}.yml'
with open(CONFIG_PATH, 'r') as f:
    NN_config = yaml.load(f, Loader=yaml.FullLoader)

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
        size=(25000, config['prior_pars']['num_pars']),
    )

    init_states, _ = forward_model.initialize_state(
        pars=init_pars
    )

    t_end = NN_config['forward_model_args']['model_args']['num_previous_steps'] * \
        NN_config['forward_model_args']['num_skip_steps'] * \
        NN_config['forward_model_args']['PDE_step_size']
    
    t_range = [0, t_end]


    state, _ = forward_model.compute_forward_model(
        state_ensemble=init_states[:, :, :, 0],
        pars_ensemble=init_pars,
        t_range=t_range,
    )

    full_x_points = np.linspace(
        0, 
        NN_config['forward_model_args']['x_max'], 
        NN_config['forward_model_args']['space_dim']
    )
    state = state[:, :, : , ::NN_config['forward_model_args']['num_skip_steps']]

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