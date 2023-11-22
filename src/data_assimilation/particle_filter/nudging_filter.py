import pdb
import numpy as np

import torch
from data_assimilation.particle_filter.base import BaseParticleFilter

import matplotlib.pyplot as plt


class NudgingFilter(BaseParticleFilter):

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

    def _get_mean_state_prior(self, state_ensemble):
        return state_ensemble.mean(axis=0)


    def _update_model_error(self, state_ensemble, pars_ensemble):
        """Update the model error."""
        
        # Update the model error distribution
        self.model_error.update(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type == 'PDE' else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble,
        )
                
    def _add_model_error(self, state_ensemble, pars_ensemble):
        """Add the model error."""

        # Add model error to the particles
        prior_state_ensemble, prior_pars_ensemble = self.model_error.add_model_error(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type == 'PDE' else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1:],
        )

        return prior_state_ensemble, prior_pars_ensemble
    
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
    
    def _get_likelihood(
        self, 
        ensemble, 
        observations, 
        t_range,
    ):
        
        state_old = ensemble[:, :-self.forward_model.num_pars]
        state_old = state_old.reshape(state_old.shape[0], self.forward_model.latent_dim, -1)

        pars_old = ensemble[:, -self.forward_model.num_pars:]
        pars_old = pars_old.reshape(pars_old.shape[0], self.forward_model.num_pars, -1)

        state_with_noise, pars_with_noise = self.model_error.add_model_error(
            state_ensemble=state_old,
            pars_ensemble=pars_old,
        )

        # Compute the prior particles
        state_with_noise_pred, _ = self.forward_model.compute_forward_model(
            state_ensemble=state_with_noise,
            pars_ensemble=pars_with_noise[:, :, -1],
            output_seq_len=1,
            with_grad=True,
        )
        state_with_noise = torch.cat([state_with_noise, state_with_noise_pred], dim=-1)

        state_old_pred, _ = self.forward_model.compute_forward_model(
            state_ensemble=state_old,
            pars_ensemble=pars_old[:, :, -1],
            output_seq_len=1,
            with_grad=True,
        )
        state_old = torch.cat([state_old, state_old_pred], dim=-1)
        
        # Transform the state
        state_transformed = self.forward_model.transform_state(
            state=state_old[:, :, -1:],
            x_points=self.observation_operator.full_space_points,
            pars=pars_old[:, :, -1],
            numpy=False,
            with_grad=True,
        )

        # Compute the likelihood
        likelihood = self.likelihood.compute_likelihood(
            state=state_transformed,
            observations=observations,
        )
        
        state_residual = state_with_noise - state_old
        pars_residual = pars_old - pars_with_noise

        state_prior = 0
        for i in range(state_residual.shape[-1]):
            state_prior = state_prior + self.model_error.state_error_distribution.log_prob(state_residual[:, :, i])
        #state_prior = torch.exp(state_prior)

        pars_prior = self.model_error.parameter_error_distribution.log_prob(pars_residual[:, :, 0])
        #pars_prior = torch.exp(pars_prior)

        likelihood = torch.log(likelihood)# + state_prior + pars_prior
        likelihood = likelihood.sum()

        return likelihood


    def _get_posterior(
        self,
        state_ensemble,
        pars_ensemble,
        observations,
        t_range,
    ):        
        
        num_nudged_particles = int(state_ensemble.shape[0]*0.5)#
        learning_rate = 1e-1

        # Compute the prior particles
        with torch.no_grad():
            state_ensemble, t_vec = self._compute_prior_particles(
                state_ensemble=state_ensemble[:, :, -self.num_previous_steps:],
                pars_ensemble=pars_ensemble[:, :, -1],
                t_range=t_range,
            )

        nudged_particle_ids = np.random.choice(
            state_ensemble.shape[0],
            size=num_nudged_particles,
            replace=False,
        )
        
        nudged_ensemble = state_ensemble[nudged_particle_ids, :, -(self.num_previous_steps+1):-1].clone()
        nudged_ensemble = nudged_ensemble.reshape(num_nudged_particles, -1)
        nudged_pars = pars_ensemble[nudged_particle_ids, :, -1].clone().to(nudged_ensemble.device)

        nudged_ensemble = torch.cat([nudged_ensemble, nudged_pars], dim=1) 

        nudged_ensemble.requires_grad = True

        optim = torch.optim.Adam([nudged_ensemble], lr=learning_rate, weight_decay=1e-1)

        for i in range(1):
            likelihood = -self._get_likelihood(
                ensemble=nudged_ensemble,
                observations=observations,
                t_range=t_range,
            )

            '''
            likelihood_grad = torch.autograd.grad(
                outputs=likelihood,
                inputs=nudged_ensemble,
            )[0]

            nudged_ensemble = nudged_ensemble + learning_rate*likelihood_grad
            '''

            optim.zero_grad()
            likelihood.backward()
            optim.step()

            '''
            # Compute the gradient of the likelihood
            likelihood_gradient = torch.autograd.grad(
                outputs=likelihood,
                inputs=nudged_ensemble,
            )[0]

            likelihood_gradient = likelihood_gradient.detach()
            print(likelihood.sum())

            nudged_ensemble = nudged_ensemble + learning_rate*likelihood_gradient
            '''
        

        nudged_state = nudged_ensemble[:, :-self.forward_model.num_pars]
        nudged_state = nudged_state.reshape(num_nudged_particles, self.forward_model.latent_dim, -1)
        if self.num_previous_steps < nudged_state.shape[-1]:
            state_ensemble[nudged_particle_ids, :, -self.num_previous_steps:] = nudged_state[:, :, -self.num_previous_steps:]
        else:
            state_ensemble[nudged_particle_ids, :, -nudged_state.shape[-1]:] = nudged_state

        pars_ensemble[nudged_particle_ids, :, -1] = nudged_ensemble[:, -self.forward_model.num_pars:]


        if self.forward_model.transform_state is not None:
            state_ensemble_transformed = self.forward_model.transform_state(
                state=state_ensemble[:, :, -1:],
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
        
        if self.save_observations:
            observations_to_save = []
            for i in range(0, self.num_particles, self.forward_model.batch_size):

                batch_ids = np.arange(i, min(i+self.forward_model.batch_size, self.num_particles)) 

                state_ensemble_transformed = self.forward_model.transform_state(
                    state=state_ensemble[batch_ids],
                    x_points=self.observation_operator.full_space_points,
                    pars=pars_ensemble[batch_ids, :, -1],
                    numpy=True if self.backend == 'numpy' else False,
                ).detach().cpu()
                observation_batch_to_save = []
                for j in range(state_ensemble_transformed.shape[0]):
                    observations_time_to_save = []
                    for k in range(state_ensemble_transformed.shape[-1]):
                        observations_time_to_save.append(
                            self.observation_operator.get_observations(
                                state=state_ensemble_transformed[j, :, :, k],
                                ensemble=False
                            ).detach().cpu().numpy()
                        )
                    observations_time_to_save = np.stack(observations_time_to_save, axis=-1)

                    observation_batch_to_save.append(observations_time_to_save)
                    
                observation_batch_to_save = np.stack(observation_batch_to_save, axis=0)
                observations_to_save.append(observation_batch_to_save)
            observations_to_save = np.concatenate(observations_to_save, axis=0)
            self.pred_observations.append(observations_to_save)
        
        if self.backend == 'torch':
            state_ensemble = state_ensemble.cpu().detach()
            pars_ensemble = pars_ensemble.cpu().detach()
        
        
        return state_ensemble, pars_ensemble
    

