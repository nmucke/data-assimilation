import pdb

import torch
from data_assimilation.particle_filter.base import BaseParticleFilter

from ksddescent import ksdd_lbfgs, ksdd_gradient
from ksddescent.contenders import svgd, mmd_lbfgs


class SteinFilter(BaseParticleFilter):

    def __init__(
        self,
        **kwargs
    ) -> None:
        
        super().__init__(**kwargs)

    def _get_mean_state_prior(self, state_ensemble):
        return state_ensemble.mean(axis=0)
    
    def _score_function(
        self, 
        state, 
        pars,
        observations, 
        likelihood_function,
        transform_state_function
        ):

        state = state.unsqueeze(-1)

        # Transform the state
        state_transform = transform_state_function(
            state=state,
            x_points=self.observation_operator.full_space_points,
            pars=pars[:, :, -1],
            numpy=False,
        )

        # Compute the likelihood
        likelihood = likelihood_function(
            observations=observations,
            state=state_transform,
        )
        likelihood = torch.log(likelihood).sum()

        # Compute the gradient of the likelihood
        likelihood_gradient = torch.autograd.grad(
            outputs=likelihood,
            inputs=[state],
        )[0].squeeze(-1)

        return likelihood_gradient

    def _get_posterior(
        self, 
        prior_state_ensemble, 
        prior_pars_ensemble, 
        likelihood_function,
        transform_state_function,
        observations,
    ):        
        

        prior_state_mean = prior_state_ensemble[:, :, -1].mean(axis=0)
        prior_pars_mean = prior_pars_ensemble[:, :, -1].mean(axis=0)

        score = lambda x: self._score_function(
            state=x*prior_state_mean, 
            pars=prior_pars_ensemble,
            observations=observations, 
            likelihood_function=likelihood_function,
            transform_state_function=transform_state_function
        )

        state = prior_state_ensemble[:, :, -1].clone()
        state.requires_grad = True

        posterior_state_ensemble= ksdd_lbfgs(state, score, bw=0.1, store=False, max_iter=25, tol=1e-5,)
        #posterior_state_ensemble= ksdd_gradient(state, score, step=0.0003, bw=0.1, store=False, max_iter=2)
        #posterior_state_ensemble= svgd(state, score, step=0.003, bw=0.1, store=False, max_iter=10)


        posterior_state_ensemble = posterior_state_ensemble.detach()
        posterior_state_ensemble = posterior_state_ensemble.unsqueeze(-1)
        

        return posterior_state_ensemble, prior_pars_ensemble
    

