import numpy as np
import pdb
import matplotlib.pyplot as plt

from data_assimilation.true_solution import TrueSolution
from data_assimilation.oracle import ObjectStorageClientWrapper

def plot_state_variable(
    x_vec: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    true_state: np.ndarray,
    save_path: str,
    x_obs_vec: np.ndarray = None,
    state_obs: np.ndarray = None,
):
    
    plt.figure()
    plt.plot(x_vec, true_state, linewidth=3., color='black')
    plt.plot(x_vec, state_mean, linewidth=2., color='tab:blue')
    plt.fill_between(
        x_vec,
        state_mean - 2*state_std,
        state_mean + 2*state_std,
        alpha=0.25,
        color='tab:blue',
    )
    if state_obs is not None:
        plt.scatter(x_obs_vec, state_obs, color='black', s=40)
    plt.grid()

def plot_state_results(
    state_ensemble: np.ndarray,
    true_solution: TrueSolution,
    save_path: str,
    object_storage_client: ObjectStorageClientWrapper = None,
    num_states_to_plot: int = 3,
):
    
    x_vec = true_solution.observation_operator.full_space_points
    x_obs_vec = true_solution.observation_operator.x_observation_points
    
    state_mean = np.mean(state_ensemble, axis=0)
    state_std = np.std(state_ensemble, axis=0)

    if num_states_to_plot == 3:
        liquid_hold_up_path = f'{save_path}/liquid_hold_up.png'
        plot_state_variable(
            x_vec=x_vec,
            state_mean=state_mean[0, :, -1],
            state_std=state_std[0, :, -1],
            true_state=true_solution.state[0, :, true_solution.observation_times[-1]],
            save_path=liquid_hold_up_path,
        )

        if object_storage_client is not None:
            with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path}/liquid_hold_up.png', 'wb') as f:
                plt.savefig(f)
        else:
            plt.savefig(liquid_hold_up_path)
        plt.close()


    pressure_path = f'{save_path}/pressure.png'
    plot_state_variable(
        x_vec=x_vec,
        state_mean=state_mean[1, :, -1] if num_states_to_plot == 3 else state_mean[0, :, -1],
        state_std=state_std[1, :, -1] if num_states_to_plot == 3 else state_std[0, :, -1],
        true_state=true_solution.state[1, :, true_solution.observation_times[-1]] if num_states_to_plot == 3 else true_solution.state[0, :, true_solution.observation_times[-1]],
        save_path=pressure_path,
        x_obs_vec=x_obs_vec,
        state_obs=true_solution.observations[:, -1] if 0 in true_solution.observation_operator.observation_state_ids else None,
    )
    if object_storage_client is not None:
        with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path}/pressure.png', 'wb') as f:
            plt.savefig(f)
    else:
        plt.savefig(pressure_path)
    plt.close()

    velocity_path = f'{save_path}/velocity.png'
    plot_state_variable(
        x_vec=x_vec,
        state_mean=state_mean[-1, :, -1],
        state_std=state_std[-1, :, -1],
        true_state=true_solution.state[-1, :, true_solution.observation_times[-1]] if num_states_to_plot == 3 else true_solution.state[1, :, true_solution.observation_times[-1]],
        save_path=velocity_path,
        x_obs_vec=x_obs_vec,
        state_obs=true_solution.observations[:, -1] if 1 in true_solution.observation_operator.observation_state_ids else None,
    )
    if object_storage_client is not None:
        with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path}/velocity.png', 'wb') as f:
            plt.savefig(f)
    else:
        plt.savefig(velocity_path)
    plt.close()

def plot_parameter_results(
    pars_ensemble: np.ndarray,
    true_solution: TrueSolution,
    save_path: str,
    object_storage_client: ObjectStorageClientWrapper = None 
):

    plt.figure()
    plt.hist(pars_ensemble[:, 0, -1], bins=50)
    plt.axvline(x=true_solution.pars[0], color='black', linewidth=3.)
    if object_storage_client is not None:
        with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path}/leak_location.png', 'wb') as f:
            plt.savefig(f)
    else:
        plt.savefig(f'{save_path}/leak_location.png')
    plt.close()

    plt.figure()
    plt.hist(pars_ensemble[:, 1, -1], bins=50)
    plt.axvline(x=true_solution.pars[1], color='black', linewidth=3.)
    if object_storage_client is not None:
        with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path}/leak_size.png', 'wb') as f:
            plt.savefig(f)
    else:
        plt.savefig(f'{save_path}/leak_size.png')
    plt.close()