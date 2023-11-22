import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
        plt.plot(x_obs_vec, state_obs, '.', color='tab:red', markersize=15)#, s=40)
    plt.grid()

def plot_state_results(
    state_ensemble: np.ndarray,
    true_solution: TrueSolution,
    save_path: str,
    object_storage_client: ObjectStorageClientWrapper = None,
    plotting_args: dict = None,
):
    
    obs = true_solution.observations[:, -1]

    x_vec = true_solution.observation_operator.full_space_points
    x_obs_vec = true_solution.observation_operator.x_observation_points
    state_obs_ids = true_solution.observation_operator.observation_state_ids
    
    state_mean = np.mean(state_ensemble, axis=0)
    state_std = np.std(state_ensemble, axis=0)

    for state_idx in plotting_args['states_to_plot']:
        save_path_i = f'{save_path}/{plotting_args["state_names"][state_idx]}.png'
        plot_state_variable(
            x_vec=x_vec,
            state_mean=state_mean[state_idx, :, -1],
            state_std=state_std[state_idx, :, -1],
            true_state=true_solution.state[state_idx, :, true_solution.observation_times[-1]],
            save_path=save_path_i,
            x_obs_vec=x_obs_vec,
            state_obs=obs if state_idx in state_obs_ids else None,
        )

        if object_storage_client is not None:
            with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path_i}', 'wb') as f:
                plt.savefig(f)
        else:
            plt.savefig(save_path_i)
        plt.close()



def plot_observation_results(
    obs_ensemble: np.ndarray,
    true_solution: TrueSolution,
    save_path: str,
    object_storage_client: ObjectStorageClientWrapper = None,
    plotting_args: dict = None,
):

    save_path_i = f'{save_path}/observations.png'

    obs_mean = np.mean(obs_ensemble, axis=0)
    obs_std = np.std(obs_ensemble, axis=0)

    plt.figure()
    for i in range(obs_mean.shape[0]):
        plt.plot(
            np.linspace(0, true_solution.observation_t_vec[-1], obs_mean.shape[-1]),
            obs_mean[i], 
            linewidth=2., color='tab:blue'
        )
        plt.fill_between(
            np.linspace(0, true_solution.observation_t_vec[-1], obs_mean.shape[-1]),
            obs_mean[i] - 2*obs_std[i],
            obs_mean[i] + 2*obs_std[i],
            alpha=0.25,
            color='tab:blue',
        )
        plt.plot(
            true_solution.observation_t_vec, 
            true_solution.observations[i]
            , '.', color='tab:red', markersize=15
        )
    plt.ylabel(plotting_args['observations_names'][0])
    plt.xlabel('Time')
    plt.grid()
    
    if object_storage_client is not None:
        with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path_i}', 'wb') as f:
            plt.savefig(f)
    else:
        plt.savefig(save_path_i)
    plt.close()


def plot_parameter_results(
    pars_ensemble: np.ndarray,
    true_solution: TrueSolution,
    save_path: str,
    object_storage_client: ObjectStorageClientWrapper = None,
    plotting_args: dict = None,
):
    
    for pars_idx in plotting_args['pars_to_plot']:
        save_path_i = f'{save_path}/{plotting_args["par_names"][pars_idx]}.png'

        pars = pars_ensemble[:, pars_idx, -1]

        # get KDE approximations for plotting
        kde_pars = gaussian_kde(pars, bw_method=0.3).pdf

        x_vec = np.linspace(
            pars.min()-pars.std()*5, 
            pars.max()+pars.std()*5, 
            1000
        )

        plt.figure()
        plt.hist(pars, bins=50, density=True, color='tab:blue', alpha=0.5)
        plt.plot(x_vec, kde_pars(x_vec), color='tab:blue', linewidth=2.)
        plt.axvline(x=true_solution.pars[pars_idx], color='black', linewidth=3.)

        if object_storage_client is not None:
            with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path_i}', 'wb') as f:
                plt.savefig(f)
        else:
            plt.savefig(save_path_i)
        plt.close()

    if plotting_args.get('plot_bar') is not None:

        x_vec = np.linspace(0, 25.6, 1000)
        iM1 = np.where(x_vec <= 6.0)
        iM2 = np.where((x_vec > 6.0) & (x_vec <= 12.0))
        iM3 = np.where((x_vec > 12.0) & (x_vec <= 14.0))
        iM4 = np.where((x_vec > 14.0) & (x_vec <= 17.0))
        iM5 = np.where(x_vec > 17.0)


        bar_height_ensemble = np.zeros((pars_ensemble.shape[0], x_vec.shape[0]), dtype=float)
        for i in range(0, pars.shape[0]):
            bar_height = pars[i]

            hx = np.zeros_like(x_vec, dtype=float)

            hx[iM2] = -(0.4-bar_height)/6
            hx[iM4] = (0.4-bar_height)/3

            h = np.zeros_like(x_vec, dtype=float)

            h[iM2] = 0.4 + hx[iM2]*(x_vec[iM2]-6.0)
            h[iM3] = bar_height
            h[iM4] = bar_height + hx[iM4]*(x_vec[iM4]-14.0)

            h[iM1] = 0.4
            h[iM5] = 0.4

            h = -h

            bar_height_ensemble[i, :] = h

        bar_height_mean = np.mean(bar_height_ensemble, axis=0)
        bar_height_std = np.std(bar_height_ensemble, axis=0)

        true_bar_height = true_solution.pars[0]

        hx = np.zeros_like(x_vec, dtype=float)

        hx[iM2] = -(0.4-true_bar_height)/6
        hx[iM4] = (0.4-true_bar_height)/3

        h = np.zeros_like(x_vec, dtype=float)

        h[iM2] = 0.4 + hx[iM2]*(x_vec[iM2]-6.0)
        h[iM3] = true_bar_height
        h[iM4] = true_bar_height + hx[iM4]*(x_vec[iM4]-14.0)

        h[iM1] = 0.4
        h[iM5] = 0.4

        h = -h


        plt.plot(x_vec, bar_height_mean, '-', linewidth=2., color='tab:blue', label='Prediction')
        plt.fill_between(
            x_vec,
            bar_height_mean - 2*bar_height_std,
            bar_height_mean + 2*bar_height_std,
            alpha=0.25,
            color='tab:blue',
        )
        plt.plot(x_vec, h, '-', linewidth=3., color='black', label='True')
        
        plt.ylim(-0.41, 0.0)
        plt.legend()

        save_path_i = f'{save_path}/bar.png'
        if object_storage_client is not None:
            with object_storage_client.fs.open(f'{object_storage_client.bucket_name}@{object_storage_client.namespace}/{save_path_i}', 'wb') as f:
                plt.savefig(f)
        else:
            plt.savefig(save_path_i)
        plt.close()
        #plt.show()
    