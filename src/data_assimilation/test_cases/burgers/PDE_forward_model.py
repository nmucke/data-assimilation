import pdb

import numpy as np
import matplotlib.pyplot as plt
import ray

from data_assimilation.particle_filter.base import BaseForwardModel
from data_assimilation.test_cases.burgers.PDE_model import Burgers
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

@ray.remote(num_returns=1, num_cpus=1)
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
        
        super().__init__()

        self.distributed = distributed
        self.model_type = 'PDE'

        self.model = Burgers(
            **model_args
        )
        
    def update_params(self, params):
        self.model.update_parameters(params)
    
    def transform_state(self, state, x_points, pars, **kwargs):
        return state
            
    def initialize_state(self, pars):
        state, pars = self.model.initialize_state(pars)
        state = np.expand_dims(state, axis=1)
        state = np.expand_dims(state, axis=-1)
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
                sol_i = compute_forward_model_PDE(
                    forward_model=self.model,
                    state=state_ensemble[i],
                    pars=pars_ensemble[i],
                    t_range=t_range
                )

            sol.append(sol_i)

        if self.distributed:
            sol = ray.get(sol)


        sol = np.array(sol)

        sol = np.expand_dims(sol, axis=1)
        sol = np.expand_dims(sol, axis=-1)
                
        return sol, 0
        
        

