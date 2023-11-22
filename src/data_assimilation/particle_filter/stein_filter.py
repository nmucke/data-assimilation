import pdb
import numpy as np

import torch
from data_assimilation.particle_filter.base import BaseParticleFilter

from ksddescent import ksdd_lbfgs, ksdd_gradient
from ksddescent.contenders import svgd, mmd_lbfgs

import matplotlib.pyplot as plt

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

class SVGD:
    '''
    https://github.com/activatedgeek/svgd
    '''
    def __init__(self, log_prob, K, optimizer):
        self.log_prob = log_prob
        self.K = K
        self.optim = optimizer

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob = self.log_prob(X)
        score_func = torch.autograd.grad(log_prob.sum(), X, retain_graph=True)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return phi

    def step(self, X):
        self.optim.zero_grad()
        X.grad = -self.phi(X)
        self.optim.step()


class SteinFilter(BaseParticleFilter):

    def __init__(
        self,
        **kwargs
    ) -> None:
        
        super().__init__(**kwargs)

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
    
    '''
    def _get_likelihood_gradient(
        self, 
        ensemble, 
        observations, 
        t_range,
    ):
        
        state = ensemble[:, :-self.forward_model.num_pars]
        state = state.reshape(state.shape[0], self.forward_model.latent_dim, -1)

        pars = ensemble[:, -self.forward_model.num_pars:]

        # Compute the prior particles

        state, _ = self.forward_model.compute_forward_model(
            state_ensemble=state,
            pars_ensemble=pars,
            output_seq_len=1
        )
        
        # Transform the state
        state_transformed = self.forward_model.transform_state(
            state=state[:, :, -1:],
            x_points=self.observation_operator.full_space_points,
            pars=pars,
            numpy=False,
        )

        # Compute the likelihood
        likelihood = self.likelihood.compute_likelihood(
            state=state_transformed,
            observations=observations,
        )
        likelihood = torch.log(likelihood)#.sum()

        # Compute the gradient of the likelihood
        likelihood_gradient = torch.autograd.grad(
            outputs=likelihood,
            inputs=ensemble,
        )[0]

        # normal distribution prior
        dist = torch.distributions.normal.Normal(0.5, 0.75)


        prior = torch.zeros_like(likelihood)

        for i in range(pars.shape[1]):
            prior += dist.log_prob(pars[:, i])
            
            #if (pars[:, i] < 0).sum().item()>1 or (pars[:, i] > 1).sum().item()>1:
            #    pdb.set_trace()


        return likelihood + prior
    '''

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
        likelihood = torch.exp(likelihood)

        return likelihood
    
    def _get_likelihood_gradient(
        self,
        ensemble,
        observations,
        t_range,
    ):
        
        ensemble.requires_grad = True

        likelihood = self._get_likelihood(
            ensemble=ensemble,
            observations=observations,
            t_range=t_range,
        )

        likelihood_gradient = torch.autograd.grad(
            outputs=likelihood.sum(),
            inputs=ensemble,
            retain_graph=True,
        )[0]

        return likelihood_gradient
        


    def _get_posterior(
        self,
        state_ensemble,
        pars_ensemble,
        observations,
        t_range,
    ):        
        

        # Compute the prior particles
        with torch.no_grad():
            state_ensemble, t_vec = self._compute_prior_particles(
                state_ensemble=state_ensemble[:, :, -self.num_previous_steps:],
                pars_ensemble=pars_ensemble[:, :, -1],
                t_range=t_range,
            )

        ensemble = state_ensemble[:, :, -(self.num_previous_steps+1):-1].clone()
        ensemble = ensemble.reshape(self.num_particles, -1)
        pars = pars_ensemble[:, :, -1].clone().to(ensemble.device)

        ensemble = torch.cat([ensemble, pars], dim=1) 

        ensemble.requires_grad = True

        score_function = lambda x: self._get_likelihood_gradient(
            ensemble=x,
            observations=observations,
            t_range=t_range,
        )

        K = RBF()
        svgd = SVGD(
            log_prob=score_function,
            K=K,
            optimizer=torch.optim.Adam([ensemble], lr=1e-1),
        )

        for i in range(250):
            svgd.step(ensemble)
            
            #if i % 100 == 0:
            #    print(svgd.phi(ensemble).mean())



        #posterior_ensemble = ksdd_lbfgs(ensemble, score_function, bw=0.1, store=False, max_iter=25, tol=1e-5,)
        #posterior_ensemble = torch.zeros_like(ensemble)
        '''
        ensemble = ksdd_lbfgs(
            ensemble, 
            score_function, 
            bw=0.01, 
            store=False, 
            max_iter=25, 
            tol=1e-5,
            verbose=True,
            )

        ensemble = ksdd_gradient(
            ensemble, 
            score_function, 
            bw=1.0, 
            store=False, 
            max_iter=100, 
            step=0.1,
            verbose=True,
            )
        ensemble = ensemble.cpu().detach()
        '''
        '''
        for i in range(0, ensemble.shape[0], self.forward_model.batch_size):
            posterior_ensemble[i:i+self.forward_model.batch_size] = ksdd_gradient(
                ensemble[i:i+self.forward_model.batch_size], 
                score_function, 
                bw=1.0, 
                store=False, 
                max_iter=25, 
                step=0.1
            ).cpu().detach()
        '''
        
        
        state = ensemble[:, :-self.forward_model.num_pars]
        state = state.reshape(self.num_particles, self.forward_model.latent_dim, -1)
        if self.num_previous_steps < state.shape[-1]:
            state_ensemble[:, :, -self.num_previous_steps:] = state[:, :, -self.num_previous_steps:]
        else:
            state_ensemble[:, :, -state.shape[-1]:] = state


        pars_ensemble = ensemble[:, -self.forward_model.num_pars:]
        pars_ensemble = pars_ensemble.unsqueeze(-1)


        '''
        self._update_model_error(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type == 'PDE' else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1],
        )

        # Add model error to the particles
        state_ensemble, pars_ensemble = self._add_model_error(
            state_ensemble=\
                state_ensemble[:, :, :, -1] if self.model_type == 'PDE' else \
                state_ensemble[:, :, -self.num_previous_steps:],
            pars_ensemble=pars_ensemble[:, :, -1:],
        )
        '''

        '''
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

        '''

        #posterior_state_ensemble = posterior_state_ensemble.detach()
        #posterior_state_ensemble = posterior_state_ensemble.unsqueeze(-1)
        
        return state_ensemble, pars_ensemble
    

