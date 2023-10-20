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
BATCH_SIZE = 25

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

LOCAL_SAVE_PATH_STATE = f'data/{TEST_CASE}/initial_conditions/state'
LOCAL_SAVE_PATH_PARS = f'data/{TEST_CASE}/initial_conditions/pars'

create_directory(LOCAL_SAVE_PATH_STATE)
create_directory(LOCAL_SAVE_PATH_PARS)
    
def main():

    num_samples = 25000

    # Initialize forward model.
    forward_model = PDEForwardModel(
        **config['forward_model_args'],
        distributed=DISTRIBUTED,
    )

    init_pars = np.random.uniform(
        low=config['prior_pars']['lower_bound'],
        high=config['prior_pars']['upper_bound'],
        size=(num_samples, config['prior_pars']['num_pars']),
    )

    init_states, _ = forward_model.initialize_state(
        pars=init_pars
    )

    t_end = NN_config['forward_model_args']['model_args']['num_previous_steps'] * \
        NN_config['forward_model_args']['num_skip_steps'] * \
        NN_config['forward_model_args']['PDE_step_size']
    
    t_range = [0, t_end]

    full_x_points = np.linspace(
        0, 
        NN_config['forward_model_args']['x_max'], 
        NN_config['forward_model_args']['space_dim']
    )

    for i in range(0, num_samples, BATCH_SIZE): 
        
        batch_ids = range(i, i + BATCH_SIZE)
                
        state, _ = forward_model.compute_forward_model(
            state_ensemble=init_states[batch_ids, :, :, 0],
            pars_ensemble=init_pars[batch_ids],
            t_range=t_range,
        )

        state = state[:, :, : , ::NN_config['forward_model_args']['num_skip_steps']]

        state_transformed = np.zeros((state.shape[0], state.shape[1], len(full_x_points), state.shape[-1]))
        for t_idx in range(state.shape[-1]):
            state_transformed[:, :, :, t_idx] = \
                forward_model.transform_state(state[:, :, :, t_idx], full_x_points)

        for save_i in batch_ids:
            np.savez_compressed(f'{LOCAL_SAVE_PATH_STATE}/sample_{save_i}.npz', data=state_transformed[save_i-i])
            np.savez_compressed(f'{LOCAL_SAVE_PATH_PARS}/sample_{save_i}.npz', data=init_pars[save_i-i])


    
if __name__ == '__main__':

    if DISTRIBUTED:
        ray.shutdown()
        ray.init(num_cpus=NUM_WORKERS)
    main()

    if DISTRIBUTED:
        ray.shutdown()