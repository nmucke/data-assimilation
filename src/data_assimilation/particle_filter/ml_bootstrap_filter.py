

import pdb
import numpy as np
import torch
from data_assimilation.particle_filter.base import BaseParticleFilter



class MLBootstrapFilter(BaseParticleFilter):

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
        
    def _compute_prior_particles(
        self, 
        state_ensemble, 
        pars_ensemble,
        t_range
    ):

        state_ensemble = self.forward_model.compute_forward_model(
            state_ensemble=state_ensemble,
            pars_ensemble=pars_ensemble,
            t_range=t_range,
        )

        return state_ensemble
    
    def _get_posterior(
        self, 
        state_ensemble, 
        pars_ensemble, 
        observations,
        t_range,
    ):
        
        self.model_error.update(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1],
        )

        state_ensemble, pars_ensemble = self.model_error.add_model_error(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1:],
        )

        if self.model_type in ['FNO']:
            state_ensemble = torch.tensor(state_ensemble).unsqueeze(-1)
            pars_ensemble = pars_ensemble.clone().detach()
                
        state_ensemble, t_vec = self._compute_prior_particles(
            state_ensemble=\
                state_ensemble if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1],
            t_range=t_range,
        )

        if self.forward_model.transform_state is not None:
            state_ensemble_transformed = self.forward_model.transform_state(
                state=\
                    state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                    state_ensemble[:, :, -1:],
                x_points=self.observation_operator.full_space_points,
                pars=pars_ensemble[:, :, -1],
                numpy=True if self.backend == 'numpy' else False,
            )
        else:
            state_ensemble_transformed = \
                state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -1],
        
        # Compute the likelihood    
        likelihood = self.likelihood.compute_likelihood(
            state=state_ensemble_transformed, 
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
            state_ensemble, pars_ensemble = \
                self._resample(
                    state_ensemble=state_ensemble,
                    pars_ensemble=pars_ensemble,
                    weights=self.weights,
                )
            self._update_weights(restart=True)

            print('Resampling')
        
        if self.backend == 'torch':
            state_ensemble = state_ensemble.cpu().detach()
            pars_ensemble = pars_ensemble.cpu().detach()
        
        return state_ensemble, pars_ensemble
    

