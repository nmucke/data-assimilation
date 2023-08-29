

import numpy as np
import torch
from data_assimilation.particle_filter.base import BaseModelError
import pdb
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from scipy.sparse import diags

class PDEModelError(BaseModelError):
    def __init__(
        self, 
        noise_variance: float,
        space_dim: int,
        **kwargs
        ):
        super().__init__(**kwargs)

        self.noise_variance = noise_variance

        self.space_dim = space_dim

        self.num_states = len(self.noise_variance)

        band_width = 8

        var_multipliers = np.linspace(0.1, 1, band_width, endpoint=False)
        var_multipliers = np.concatenate((var_multipliers, np.array([1.]), var_multipliers[::-1]))
        var_multipliers = var_multipliers/var_multipliers.sum()

        diag_indices = np.arange(-band_width, band_width+1)

        diag_var = []
        for i in range(self.num_states):
            diag_var.append(var_multipliers*self.noise_variance[i])

        self.covariance_matrices = [
            diags(
                diag_var[i],
                diag_indices, 
                shape=(space_dim,space_dim)
            ).toarray() for i in range(self.num_states)
        ]

        self.model_error_distributions = []
        for i in range(self.num_states):
            self.model_error_distributions.append(
                multivariate_normal(mean=np.zeros(space_dim), cov=self.covariance_matrices[i])
                #norm(loc=0, scale=self.noise_variance[i])
            )
            
    def add_model_error(self, state_ensemble, pars_ensemble):

        state_ensemble = state_ensemble.copy()

        '''
        for i in range(self.num_states):

            noise = self.model_error_distributions[i].rvs(
                size=state_ensemble.shape[0]
            )
            #noise = self.model_error_distributions[i].rvs(
            #        size=(state_ensemble.shape[0], state_ensemble.shape[2])
            #    )

            state_ensemble[:, i, :] = state_ensemble[:, i, :] + noise#\
                #self.model_error_distributions[i].rvs(
                #    size=(state_ensemble.shape[0], state_ensemble.shape[2])
                #)
        '''
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
        
        