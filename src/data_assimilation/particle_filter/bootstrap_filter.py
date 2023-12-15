

import pdb
import numpy as np
import torch
from data_assimilation.particle_filter.base import BaseParticleFilter
import matplotlib.pyplot as plt


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
        state_ensemble, 
        pars_ensemble, 
        observations,
        t_range,
    ):
        
        #self.model_error.update(
        #    state_ensemble=\
        #        state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
        #        state_ensemble[:, :, -self.num_previous_steps:],
        #    pars_ensemble=pars_ensemble[:, :, -1],
        #)
        state_ensemble, pars_ensemble = self.model_error.add_model_error(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1:],
        )

        if self.model_type in ['FNO']:
            state_ensemble = torch.tensor(state_ensemble).unsqueeze(-1)
            pars_ensemble = pars_ensemble.clone().detach()
                
        state_ensemble_new, t_vec = self.forward_model.compute_forward_model(
            state_ensemble=\
                state_ensemble if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1],
            t_range=t_range,
        )
        if self.model_type in ['latent']:
            state_ensemble_new, t_vec = self.forward_model.compute_forward_model(
                state_ensemble=state_ensemble[:, :, -self.num_previous_steps:],
                pars_ensemble=pars_ensemble[:, :, -1],
                t_range=t_range,
            )
            if state_ensemble_new.shape[-1] < self.num_previous_steps:
                state_ensemble = torch.cat(
                    [
                        state_ensemble[:, :, -(self.num_previous_steps - state_ensemble_new.shape[-1]):],
                        state_ensemble_new,
                    ],
                    axis=-1
                )
            else:
                state_ensemble = state_ensemble_new
        else:
            state_ensemble, t_vec = self.forward_model.compute_forward_model(
                state_ensemble=state_ensemble,
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

            self.model_error.update(
                state_ensemble=\
                    state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                    state_ensemble[:, :, -self.num_previous_steps:],
                pars_ensemble=pars_ensemble[:, :, -1],
            )


            print('Resampling')
            
        if self.save_observations:
            observations_to_save = []
            for i in range(self.num_particles):
                state_ensemble_transformed = self.forward_model.transform_state(
                    state=state_ensemble[i:i+1],
                    x_points=self.observation_operator.full_space_points,
                    pars=pars_ensemble[i:i+1, :, -1],
                    numpy=True if self.backend == 'numpy' else False,
                )
                if self.backend == 'torch':
                    state_ensemble_transformed = state_ensemble_transformed.detach().cpu()
                observations_time_to_save = []
                for j in range(state_ensemble_transformed.shape[-1]):
                    obs = self.observation_operator.get_observations(
                        state=state_ensemble_transformed[0, :, :, j],
                        ensemble=False
                    )

                    if self.backend == 'torch':
                        obs.detach().cpu().numpy()
                    observations_time_to_save.append(obs)
                observations_time_to_save = np.stack(observations_time_to_save, axis=-1)
                observations_to_save.append(observations_time_to_save)

            observations_to_save = np.stack(observations_to_save, axis=0)
            self.pred_observations.append(observations_to_save)
        
        
        if self.backend == 'torch':
            state_ensemble = state_ensemble.cpu().detach()
            pars_ensemble = pars_ensemble.cpu().detach()
        
        return state_ensemble, pars_ensemble
    

