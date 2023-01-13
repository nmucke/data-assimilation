from attr import dataclass
import ray
from data_assimilation.PDE_models import AdvectionEquation
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.animation import FuncAnimation
from data_assimilation.particle_filter import ParticleFilter
from data_assimilation.aux_particle_filter import AuxParticleFilter
from data_assimilation.test_cases.advection_equation import (
    TrueSolution,
    ObservationOperator,
    AdvectionForwardModel,
    ModelError,
)

xmin=0.
xmax=2.*np.pi

BC_params = {
    'type':'dirichlet',
    'treatment': 'naive',
    'numerical_flux': 'lax_friedrichs',
}

numerical_flux_type = 'lax_friedrichs'
numerical_flux_params = {
    'alpha': 0.0,
}
stabilizer_type = 'filter'
stabilizer_params = {
    'num_modes_to_filter': 10,
    'filter_order': 6,
}

step_size = 0.025
time_integrator_type = 'implicit_euler'
time_integrator_params = {
    'step_size': step_size,
    'newton_params':{
        'solver': 'krylov',
        'max_newton_iter': 200,
        'newton_tol': 1e-5
        }
    }

polynomial_type='legendre'
num_states=1

polynomial_order=7
num_elements=6

adv_DG = AdvectionEquation(
    xmin=xmin,
    xmax=xmax,
    num_elements=num_elements,
    polynomial_order=polynomial_order,
    polynomial_type=polynomial_type,
    num_states=num_states,
    BC_params=BC_params,
    stabilizer_type=stabilizer_type, 
    stabilizer_params=stabilizer_params,
    time_integrator_type=time_integrator_type,
    time_integrator_params=time_integrator_params, 
    numerical_flux_type=numerical_flux_type,
    numerical_flux_params=numerical_flux_params,
    )


def get_true_sol(
    true_state_init, 
    true_pars, 
    observation_operator, 
    t_vec, 
    obs_times_idx, 
    observation_params
    ):
    """Get the true solution."""

    adv_DG.model_params['advection_velocity'] = true_pars[0]
    #adv_DG.model_params['inflow_frequence'] = true_pars[1]

    true_sol, t_vec = adv_DG.solve(
        t=t_vec[0],
        t_final=t_vec[-1],
        q_init=true_state_init,
        print_progress=False,
        )

    true_sol = TrueSolution(
        x_vec=adv_DG.DG_vars.x.flatten('F'),
        t_vec=np.array(t_vec),
        sol=true_sol,
        pars=true_pars,
        obs_times_idx=obs_times_idx,
        observation_operator=observation_operator,
        observation_noise_params=observation_params,
        )

    return true_sol

observation_operator_params = {
    'observation_index': np.arange(0, adv_DG.DG_vars.Np * adv_DG.DG_vars.K, 5)
}
model_error_params = {
    'state_std': 0.1,
    'pars_std': 0.1,
    'smoothing_factor': 0.4,
}
particle_filter_params = {
    'num_particles': 900,
}
observation_params = {
    'std': 0.1,
}
likelihood_params = {
    'std': 0.1,
}


def main():

    observation_operator = ObservationOperator(
        params=observation_operator_params
        )

    true_state_init = adv_DG.initial_condition(adv_DG.DG_vars.x.flatten('F'))
    true_pars = np.array([1*np.pi, 2*np.pi])
    t_range = [0, 5.]
    t_vec = np.arange(t_range[0], t_range[1], step_size)
    obs_times_idx = np.arange(0, len(t_vec), 20)

    true_sol = get_true_sol(
        true_state_init=true_state_init,
        true_pars=true_pars,
        observation_operator=observation_operator,
        t_vec=t_vec,
        obs_times_idx=obs_times_idx,
        observation_params=observation_params,
        )    

    forward_model = AdvectionForwardModel(
        model=adv_DG,
        model_error_params=model_error_params,
        )

    particle_filter = ParticleFilter(
    #particle_filter = AuxParticleFilter(
        params=particle_filter_params,
        forward_model=forward_model,
        observation_operator=observation_operator,
        likelihood_params=likelihood_params,
        )
    
    state_init = true_state_init
    pars_init = np.array([1.2*np.pi, 1.8*np.pi])

    state, pars = particle_filter.compute_filtered_solution(
        true_sol=true_sol,
        state_init=state_init,
        pars_init=pars_init,
        )

    mean_state = np.mean(state, axis=0)
    mean_pars = np.mean(pars, axis=0)

    std_state = np.std(state, axis=0)
    std_pars = np.std(pars, axis=0)
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(adv_DG.DG_vars.x.flatten('F'), mean_state[0, :, -1], label='Mean sol', linewidth=3)
    plt.fill_between(
        adv_DG.DG_vars.x.flatten('F'),
        mean_state[0, :, -1] - std_state[0, :, -1],
        mean_state[0, :, -1] + std_state[0, :, -1],
        alpha=0.25,
        )
    plt.plot(
        adv_DG.DG_vars.x.flatten('F'), 
        true_sol.sol[0, :, true_sol.obs_times_idx[-1]], 
        label='True sol', linewidth=2
        )
    plt.plot(
        true_sol.obs_x,
        true_sol.observations[0, :, -1],
        '.',
        label='observations', markersize=20)
    plt.grid()
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.hist(pars[:, 0, -1], bins=30, density='Particle filter', label='advection_velocity')
    plt.axvline(true_pars[0], color='k', label='True value', linewidth=3)

    plt.subplot(1, 4, 3)
    plt.hist(pars[:, 1, -1], bins=30, density='Particle filter', label='advection_velocity')
    plt.axvline(true_pars[1], color='k', label='True value', linewidth=3)

    plt.subplot(1, 4, 4)
    plt.plot(range(len(std_pars[0])), mean_pars[0], label='advection_velocity', color='tab:blue', linewidth=3, linestyle='--')
    plt.fill_between(
        range(len(std_pars[0])),
        mean_pars[0] - std_pars[0],
        mean_pars[0] + std_pars[0],
        alpha=0.25,
        )
    plt.plot(range(len(std_pars[0])), true_pars[0] * np.ones(len(std_pars[0])), label='True value', color='tab:blue', linewidth=3)

    plt.plot(range(len(std_pars[0])), mean_pars[1], label='advection_velocity', color='tab:orange', linewidth=3, linestyle='--')
    plt.fill_between(
        range(len(std_pars[0])),
        mean_pars[1] - std_pars[1],
        mean_pars[1] + std_pars[1],
        alpha=0.25,
        )
    plt.plot(range(len(std_pars[0])), true_pars[1] * np.ones(len(std_pars[0])), label='True value', color='tab:orange', linewidth=3)
    plt.show()

    '''
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.hist([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, 2000)
        ax.set_ylim(2, 6)
        return ln,

    def update(frame):
        ln.set_data(x, u[:, frame])
        #plt.axvline(true_pars[0], color='k', label='True value', linewidth=3)
        return ln,

    ani = FuncAnimation(
        fig,
        update,
        frames=t_vec,
        init_func=init, 
        blit=True,
        interval=10,
        )
    ani.save('pipeflow.gif', fps=30)
    plt.show()
    '''



if __name__ == '__main__':

    ray.shutdown()
    ray.init(num_cpus=30)
    main()
    ray.shutdown()