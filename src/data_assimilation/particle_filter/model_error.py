

import torch
from data_assimilation.base import BaseModelError

from scipy.stats import norm

class PDEModelError(BaseModelError):
    def __init__(
        self, 
        noise_variance: float,
        **kwargs
        ):
        super().__init__(**kwargs)

        self.noise_variance = noise_variance

        self.model_error_distribution = \
            norm(loc=0, scale=self.noise_variance)
            

    def add_model_error(self, state_ensemble, pars_ensemble):
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
        
        