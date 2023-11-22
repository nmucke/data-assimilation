

import numpy as np
import torch
from data_assimilation.particle_filter.base import BaseModelError
import pdb
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from scipy.sparse import diags
from torch.distributions.multivariate_normal import MultivariateNormal

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

        '''
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
        '''
        self.covariance_matrices = [var * np.eye(self.space_dim) for var in self.noise_variance]
        self.model_error_distributions = []
        for i in range(self.num_states):
            self.model_error_distributions.append(
                multivariate_normal(mean=np.zeros(space_dim), cov=self.covariance_matrices[i])
            )

            
    def add_model_error(self, state_ensemble, pars_ensemble):

        if isinstance(state_ensemble, torch.Tensor):
            state_ensemble = state_ensemble.detach().numpy()
            
        state_ensemble = state_ensemble.copy()

        for i in range(self.num_states):

            noise = self.model_error_distributions[i].rvs(
                size=state_ensemble.shape[0]
            )

            state_ensemble[:, i, :] = state_ensemble[:, i, :] + noise

        return state_ensemble, pars_ensemble 
    
    def update(self, state_ensemble, pars_ensemble):

        for i in range(self.num_states):

            C = np.cov(state_ensemble[:, i].T)

            self.covariance_matrices[i] = self.noise_variance[i]*C

            self.model_error_distributions[i] = (
                multivariate_normal(
                    mean=np.zeros(self.space_dim), 
                    cov=self.covariance_matrices[i], 
                    allow_singular=True
                )
            )
        
class LatentModelError(BaseModelError):
    def __init__(
        self, 
        state_noise_variance: float,
        parameter_noise_variance: float,
        latent_dim: int,
        **kwargs
        ):
        super().__init__(**kwargs)

        self.state_noise_variance = state_noise_variance
        self.parameter_noise_variance = torch.tensor(parameter_noise_variance)

        self.counter = 0

        self.num_params = len(self.parameter_noise_variance)

        self.latent_dim = latent_dim
        
        self.state_error_distribution = \
            MultivariateNormal(
                loc=torch.zeros(self.latent_dim), 
                covariance_matrix=self.state_noise_variance*torch.eye(self.latent_dim)
            )
        
        self.parameter_covariance = torch.diag(self.parameter_noise_variance)        
        self.parameter_error_distribution = \
            MultivariateNormal(
                loc=torch.zeros(self.num_params), 
                covariance_matrix=self.parameter_covariance
            )

    def add_model_error(self, state_ensemble, pars_ensemble):

        state_ensemble = state_ensemble.clone()
        for i in range(state_ensemble.shape[-1]):
            state_noise = self.state_error_distribution.sample((state_ensemble.shape[0],))
            state_ensemble[:, :, -i] = state_ensemble[:, :, -i] + state_noise.to(state_ensemble.device)

        pars_ensemble = pars_ensemble.clone()
        parameter_noise = self.parameter_error_distribution.sample((pars_ensemble.shape[0],))
        pars_ensemble[:, : , -1] = pars_ensemble[:, : , -1] + parameter_noise.to(pars_ensemble.device)

        return state_ensemble, pars_ensemble 
    
    def update(self, state_ensemble, pars_ensemble):
        self.counter += 1
        '''
        
        C = torch.cov(state_ensemble[:, :, -1].T)

        self.state_covariance = self.state_noise_variance*C

        self.state_error_distribution = \
            MultivariateNormal(
                loc=torch.zeros(self.latent_dim), covariance_matrix=self.state_covariance
            )
        
        
        self.parameter_covariance = 1/np.sqrt(self.counter) * self.parameter_covariance#torch.cov(pars_ensemble[:, :].T) + 1e-12*torch.eye(self.num_params)
        self.parameter_error_distribution = \
            MultivariateNormal(
                loc=torch.zeros(self.num_params), covariance_matrix=self.parameter_covariance
            )
        '''
        
        