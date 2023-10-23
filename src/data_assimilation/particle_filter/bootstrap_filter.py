

import pdb
import numpy as np
from data_assimilation.particle_filter.base import BaseParticleFilter



class BootstrapFilter(BaseParticleFilter):

    def __init__(
        self,
        **kwargs
    ) -> None:
        
        super().__init__(**kwargs)


        self._update_weights(
            likelihood=None,
            restart=True
        )

        self.ESS_threshold = self.num_particles / 2

    def _update_weights(
        self, 
        likelihood=None, 
        restart=False
    ):
        if restart:
            self.weights = np.ones(self.num_particles) / self.num_particles    
            return
        
        self.weights = self.weights * likelihood
        self.weights = self.weights / (self.weights.sum() + 1e-12)

        self.ESS = 1 / (np.sum(self.weights**2) + 1e-12)
    

    def _resample(
        self, 
        state_ensemble, 
        pars_ensemble, 
        weights
    ):
        
        resampled_ids = np.random.multinomial(
            n=self.num_particles,
            pvals=weights,
        )
        indeces = np.repeat(
            np.arange(self.num_particles),
            resampled_ids
        )

        return state_ensemble[indeces], pars_ensemble[indeces]
    
    def _get_posterior(
        self, 
        prior_state_ensemble, 
        prior_pars_ensemble, 
        likelihood_function,
        transform_state_function,
        observations,
    ):
        
        if transform_state_function is not None:
            prior_state_ensemble_transformed = transform_state_function(
                state=\
                    prior_state_ensemble[:, :, :, -1] if self.backend == 'numpy' else \
                    prior_state_ensemble[:, :, -1:],
                x_points=self.observation_operator.full_space_points,
                pars=prior_pars_ensemble[:, :, -1],
                numpy=True if self.backend == 'numpy' else False,
            )
        else:
            prior_state_ensemble_transformed = \
                prior_state_ensemble[:, :, :, -1] if self.backend == 'numpy' else \
                prior_state_ensemble[:, :, -1],
        
        # Compute the likelihood    
        likelihood = likelihood_function(
            state=prior_state_ensemble_transformed, 
            observations=observations,
        )
    
        if self.backend == 'torch':
            likelihood = likelihood.detach().numpy()

        # Update the particle weights
        self._update_weights(
            likelihood=likelihood,
        )
        
        #print(f'ESS: {self.ESS:0.2f}, threshold: {self.ESS_threshold}')
        if self.ESS < self.ESS_threshold:
            posterior_state_ensemble, posterior_pars_ensemble = \
                self._resample(
                    state_ensemble=prior_state_ensemble,
                    pars_ensemble=prior_pars_ensemble,
                    weights=self.weights,
                )
            self._update_weights(restart=True)

            print('Resampling')
        else:
            posterior_state_ensemble = prior_state_ensemble
            posterior_pars_ensemble = prior_pars_ensemble
            
        return posterior_state_ensemble, posterior_pars_ensemble
    

