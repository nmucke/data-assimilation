

import numpy as np
import torch
from data_assimilation.particle_filter.base import BaseModelError
import pdb
from scipy.stats import norm
import matplotlib.pyplot as plt

class PDEModelError(BaseModelError):
    def __init__(
        self, 
        noise_variance: float,
        **kwargs
        ):
        super().__init__(**kwargs)

        self.noise_variance = noise_variance

        self.num_states = len(self.noise_variance)

        self.model_error_distributions = []
        for i in range(self.num_states):
            self.model_error_distributions.append(
                norm(loc=0, scale=self.noise_variance[i])
            )
            
    def add_model_error(self, state_ensemble, pars_ensemble):

        for i in range(self.num_states):
            state_ensemble[:, i, :] = state_ensemble[:, i, :] + \
                self.model_error_distributions[i].rvs(
                    size=(state_ensemble.shape[0], state_ensemble.shape[2])
                )
        
        return state_ensemble, pars_ensemble 
    
    def update(self, state_ensemble, pars_ensemble):
        pass    
        
class NeuralNetworkModelError(BaseModelError):
    def __init__(
        self, 
        noise_variance: float,
        **kwargs
        ):
        super().__init__(**kwargs)

        self.noise_variance = noise_variance
        
        self.model_error_distribution = \
            torch.distributions.normal.Normal(
                loc=0, scale=self.noise_variance
                )

    def add_model_error(self, state_ensemble, pars_ensemble):
        return state_ensemble, pars_ensemble 
    
    def update(self, state_ensemble, pars_ensemble):
        pass    
        
        