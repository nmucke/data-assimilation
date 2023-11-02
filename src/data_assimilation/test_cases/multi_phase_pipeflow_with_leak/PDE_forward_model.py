import pdb

import numpy as np
import matplotlib.pyplot as plt
import ray

from data_assimilation.particle_filter.base import BaseForwardModel
from data_assimilation.test_cases.multi_phase_pipeflow_with_leak.PDE_model import PipeflowEquations


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



class PDEForwardModel(BaseForwardModel):

    def __init__(
        self,
        model_args: dict,
        distributed: bool = False,
        **kwargs
        ):

        super().__init__()

        self.distributed = distributed
        self.model_type = 'PDE'

        self.steady_state_args = model_args['steady_state']
        model_args.pop('steady_state')

        self.model_parameters = {
            'L': 5000, # meters
            'd': 0.2, # meters
            'A': np.pi*0.2**2/4, # meters^2
            'c': 308., # m/s
            'rho_g_norm': 1.26, # kg/m^3
            'rho_l': 1003., # kg/m^3
            'p_amb': 1.01325e5, # Pa
            'p_norm': 1.0e5, # Pa
            'p_outlet': 1.0e6, # Pa
            'e': 1e-8, # meters
            'mu_g': 1.8e-5, # Pa*s
            'mu_l': 1.516e-3, # Pa*s
            'T_norm': 278, # Kelvin
            'T': 278, # Kelvin
            'Cd': 0.1,
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
    
    def transform_state(self, state, x_points, pars=None, **kwargs):

        state_out = np.zeros((state.shape[0], 3, x_points.shape[0]))
        
        for i in range(state.shape[0]):
            state_out[i, 0] = self.model.evaluate_solution(
                x=x_points,
                sol_nodal=state[i, 0],
            )/self.model.A

            state_out[i, 1] = self.model.evaluate_solution(
                x=x_points,
                sol_nodal=state[i, 1],
            )

            state_out[i, 2] = self.model.evaluate_solution(
                x=x_points,
                sol_nodal=state[i, 2],
            )

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
                sol_i = self._compute_forward_model.remote(
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
        
        

