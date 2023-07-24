import pdb

import numpy as np
import matplotlib.pyplot as plt
import ray

from data_assimilation.particle_filter.base import BaseForwardModel
from data_assimilation.test_cases.single_phase_pipeflow_with_leak.PDE_model import PipeflowEquations


def compute_forward_model_PDE(
    forward_model,
    state: np.ndarray, 
    pars: np.ndarray,
    t_range: list,
    ):

    forward_model.update_parameters(pars)

    sol, t_vec = forward_model.solve(
        t=t_range[0], 
        q_init=state, 
        t_final=t_range[-1], 
        print_progress=False
        )
    
    return sol

@ray.remote(num_returns=1)
def compute_forward_model_PDE_ray(
    forward_model,
    state: np.ndarray, 
    pars: np.ndarray,
    t_range: list,
    ):

    forward_model.update_parameters(pars)

    sol, t_vec = forward_model.solve(
        t=t_range[0], 
        q_init=state, 
        t_final=t_range[-1], 
        print_progress=False
        )
    
    return sol



class PDEForwardModel(BaseForwardModel):

    def __init__(
        self,
        model_args: dict,
        distributed: bool = False,
        **kwargs
        ):

        super().__init__(**kwargs)

        self.distributed = distributed

        self.steady_state_args = model_args['steady_state']
        model_args.pop('steady_state')

        self.model_parameters = {
            'L': 2000,
            'd': 0.508,
            'A': np.pi*0.508**2/4,
            'c': 308.,
            'rho_ref': 52.67,
            'p_amb': 101325.,
            'p_ref': 52.67*308**2,
            'e': 1e-2,
            'mu': 1.2e-5,
            'Cd': 5e-4,
            'leak_location': 500,
        }
        self.model = PipeflowEquations(
            model_parameters=self.model_parameters,
            **model_args
            )
        
        if self.distributed:
            self._compute_forward_model = ray.remote(compute_forward_model_PDE)
        else:
            self._compute_forward_model = compute_forward_model_PDE

    def update_params(self, params):
        self.model.update_parameters(params)
    
    def transform_state(self, state, x_points, pars):

        state_out = np.zeros((state.shape[0], 2, x_points.shape[0]))
        
        for i in range(state.shape[0]):
            state_out[i, 0] = self.model.evaluate_solution(
                x=x_points,
                sol_nodal=state[i, 0],
            )

            state_out[i, 1] = self.model.evaluate_solution(
                x=x_points,
                sol_nodal=state[i, 1],
            )

            state_out[i, 1] = state_out[i, 1]/state_out[i, 0]
            state_out[i, 0] = state_out[i, 0]/self.model.A 
            #    self.model.density_to_pressure(state_out[i, 1]/self.model_parameters['A'])

        return state_out
            
    def initialize_state(self, pars):

        init = self.model.initial_condition(self.model.DG_vars.x.flatten('F'))

        state, t_vec = self.model.solve(
            t=0, 
            q_init=init, 
            t_final=0, 
            steady_state_args=self.steady_state_args,
            print_progress=False
            )
            
        state = np.expand_dims(state, axis=0)
        state = np.repeat(state, pars.shape[0], axis=0)

        pars = np.expand_dims(pars, axis=-1)

        return state, pars

    def compute_forward_model(self, state_ensemble, pars_ensemble, t_range):

        num_particles = state_ensemble.shape[0]

        sol = []
        
        for i in range(num_particles):
            
            if self.distributed:
                sol_i = compute_forward_model_PDE_ray.remote(
                    forward_model=self.model,
                    state=state_ensemble[i],
                    pars=pars_ensemble[i],
                    t_range=t_range
                )
            else:
                sol_i = self._compute_forward_model(
                    forward_model=self.model,
                    state=state_ensemble[i],
                    pars=pars_ensemble[i],
                    t_range=t_range
                )

            sol.append(sol_i)

        if self.distributed:
            sol = ray.get(sol)

        sol = np.array(sol)
                
        return sol, 0
        
        

