

import pdb
import numpy as np
from data_assimilation.particle_filter.base import BaseParticleFilter



class BootstrapFilter(BaseParticleFilter):

    def __init__(
        self,
        **kwargs
    ) -> None:
        
        super().__init__(**kwargs)

    def _update_weights(
        self, 
        likelihood=None, 
        weights=None, 
        restart=False
    ):
        if restart:
            return np.ones(self.num_particles) / self.num_particles
        
        weights = weights * likelihood
        weights = weights / (weights.sum() + 1e-12)

        ESS = 1 / (np.sum(weights**2) + 1e-12)

        return weights, ESS
    
    def _initialize_particles(self, pars):
        
        state_ensemble, pars_ensemble = self.forward_model.initialize_state(
            pars=pars
        )
        
        return state_ensemble, pars_ensemble


