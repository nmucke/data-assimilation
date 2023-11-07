

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
        
    def _maximum_likelihood_pars_estimate(
        self,
        state_ensemble,
        pars_ensemble,
        observations,
        t_range
    ):
        pars_input = pars_ensemble[:, :, -1].clone().to(self.forward_model.device)
        pars_input.requires_grad = True

        optimizer = torch.optim.Adam([pars_input], lr=0.01)

        state_ensemble=state_ensemble[:, :, -self.num_previous_steps:]

        output_seq_len = int((t_range[-1] - t_range[0]) // self.forward_model.step_size)

        for i in range(100):
            likelihood = 0
            for batch_idx in range(0, self.num_particles, self.forward_model.batch_size):

                batch_state = state_ensemble[batch_idx:batch_idx+self.forward_model.batch_size].to(self.forward_model.device)
                batch_pars = pars_input[batch_idx:batch_idx+self.forward_model.batch_size].to(self.forward_model.device)

                batch_state = self.forward_model.time_stepping_model.multistep_prediction(
                    input=batch_state, 
                    pars=batch_pars, 
                    output_seq_len=output_seq_len,
                )

                batch_state = self.forward_model.AE_model.decode(batch_state[:, :, -1:], batch_pars)
                batch_state = self.forward_model.preprocesssor.inverse_transform_state(batch_state, ensemble=True)
                batch_state = batch_state.squeeze(-1)
                
                likelihood -= self.likelihood.compute_likelihood(
                    state=batch_state, 
                    observations=observations.to(self.forward_model.device),
                )

            likelihood = likelihood.sum()

            optimizer.zero_grad()
            likelihood.backward()
            optimizer.step()


        return pars_input.cpu().detach()        


    def _get_posterior(
        self, 
        state_ensemble, 
        pars_ensemble, 
        observations,
        t_range,
    ):
        
        ml_pars = self._maximum_likelihood_pars_estimate(
            state_ensemble=state_ensemble,
            pars_ensemble=pars_ensemble,
            observations=observations,
            t_range=t_range,
        )
        
        self.model_error.update(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=ml_pars#pars_ensemble[:, :, -1],
        )

        state_ensemble, pars_ensemble = self.model_error.add_model_error(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type in ['PDE', 'FNO'] else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=ml_pars#pars_ensemble[:, :, -1:],
        )

        state_ensemble, _ = self.forward_model.compute_forward_model(
            state_ensemble=state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble,#pars_ensemble[:, :, -1],
            t_range=t_range,
        )

        if self.model_type in ['FNO']:
            state_ensemble = torch.tensor(state_ensemble).unsqueeze(-1)
            pars_ensemble = pars_ensemble.clone().detach()
                

        if self.forward_model.transform_state is not None:
            state_ensemble_transformed = self.forward_model.transform_state(
                state=state_ensemble[:, :, -1:],
                x_points=self.observation_operator.full_space_points,
                pars=pars_ensemble,#pars_ensemble[:, :, -1],
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
        
        return state_ensemble, pars_ensemble.unsqueeze(-1)
    

