from abc import abstractmethod
from discontinuous_galerkin.base.base_model import BaseModel
import numpy as np
from attr import dataclass
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import ray
from scipy.stats import norm

from data_assimilation.model_error import BaseModelError
from data_assimilation.forward_model import BaseForwardModel
from data_assimilation.observation_operator import BaseObservationOperator


def compute_next_state(
    forward_model, 
    t_range,
    state, 
    pars
    ):
    """Compute the prior."""

    state, t_vec = \
        forward_model.compute_forward_model(
            t_range=t_range, 
            state=state
            )
    
    return state, pars, t_vec

@ray.remote(num_returns=3)
def compute_next_state_ray(
    forward_model, 
    t_range,
    state, 
    pars
    ):
    """Compute the prior."""

    return compute_next_state(
        forward_model=forward_model, 
        t_range=t_range,
        state=state, 
        pars=pars
        )

def compute_prior_particle(
    forward_model, 
    t_range,
    state, 
    pars
    ):
    """Compute the prior."""

    state, pars = \
        forward_model.model_error.add_model_error(
            state=state, 
            pars=pars
            )

    forward_model.update_params(pars=pars)

    state, t_vec = \
        forward_model.compute_forward_model(
            t_range=t_range, 
            state=state
            )
    
    return state, pars, t_vec

@ray.remote(num_returns=3)
def compute_prior_particle_ray(
    forward_model, 
    t_range,
    state, 
    pars
    ):
    """Compute the prior."""

    return compute_prior_particle(
        forward_model=forward_model, 
        t_range=t_range,
        state=state, 
        pars=pars
        )

def compute_likelihood(
    state: np.ndarray,
    observation: np.ndarray,
    observation_operator: BaseObservationOperator,
    likelihood_params: dict,
    ) -> np.ndarray:
    """Compute the likelihood."""

    model_observations = observation_operator.get_observations(state=state)

    residual = observation - model_observations

    lol = norm(loc=0, scale=likelihood_params['std'])

    likelihood = lol.pdf(residual).sum()
    '''
    likelihood = np.exp(-0.5/likelihood_params['std']/likelihood_params['std'] * residual_norm)
    likelihood = np.exp(-0.5/10/10 * residual_norm)
    likelihood = likelihood / np.sqrt(2*np.pi*likelihood_params['std']**2)/residual.shape[1]
    '''


    return likelihood

@ray.remote
def compute_likelihood_ray(
    state: np.ndarray,
    observation: np.ndarray,
    observation_operator: BaseObservationOperator,
    likelihood_params: dict,
    ) -> np.ndarray:
    """Compute the likelihood."""
    
    return compute_likelihood(
        state=state,
        observation=observation,
        observation_operator=observation_operator,
        likelihood_params=likelihood_params,
    )
class AuxParticleFilter():

    def __init__(
        self,
        params: dict,
        forward_model: BaseForwardModel,
        observation_operator: BaseObservationOperator,
        likelihood_params: dict,
        ) -> None:

        self.params = params
        self.forward_model = forward_model
        self.observation_operator = observation_operator
        self.likelihood_params = likelihood_params

        self.ESS_threshold = self.params['num_particles'] / 2
    

    def _update_weights_prior(self, likelihood, weights):
        """Update the weights of the particles."""
        
        weights = weights * likelihood
        weights = weights / weights.sum()

        return weights

    def _update_weights(self, prior_likelihood, likelihood, weights):
        """Update the weights of the particles."""
        
        weights = likelihood/prior_likelihood
        weights = weights / weights.sum()

        ESS = 1 / np.sum(weights**2)

        return weights, ESS

    def _restart_weights(self, ):
        """Restart the weights of the particles."""
        
        return np.ones(self.params['num_particles']) / self.params['num_particles']
    
    def _compute_mean_state(self, t_range, state, pars):

        if ray.is_initialized():
            state_ensemble = []
            for i in range(self.params['num_particles']):
                particle_state, _, _ = compute_next_state_ray.remote(
                        forward_model=self.forward_model,
                        t_range=t_range,
                        state=state[i], 
                        pars=pars[i]
                        )
                state_ensemble.append(particle_state)
                
            state = ray.get(state_ensemble)
            state = np.asarray(state)
        else:
            for i in range(self.params['num_particles']):
                state[i], _, _ = compute_next_state(
                        forward_model=self.forward_model,
                        t_range=t_range,
                        state=state[i], 
                        pars=pars[i]
                        )

        return state

    def _compute_initial_ensemble(self, state_init, pars_init):
        """Compute the initial ensemble."""
        
        state_init_ensemble = np.repeat(
            np.expand_dims(state_init, 0), 
            self.params['num_particles'], 
            axis=0
            )
        
        pars_init_ensemble = np.repeat(
            np.expand_dims(pars_init, 0), 
            self.params['num_particles'], 
            axis=0
            )
        
        for i in range(self.params['num_particles']):
            state_init_ensemble[i], pars_init_ensemble[i] = \
                self.forward_model.model_error.add_model_error(
                    state=state_init_ensemble[i], 
                    pars=pars_init_ensemble[i]
                    )

        return state_init_ensemble, pars_init_ensemble   

    def _compute_prior_particles(self, t_range, state, pars):
        """Compute the prior particles."""

        if ray.is_initialized():
            state_ensemble = []
            pars_ensemble = []
            for i in range(self.params['num_particles']):
                particle_state, particle_pars, t_vec = \
                    compute_prior_particle_ray.remote(
                        forward_model=self.forward_model, 
                        t_range=t_range,
                        state=state[i], 
                        pars=pars[i]
                        )
                state_ensemble.append(particle_state)
                pars_ensemble.append(particle_pars)
                
            state_ensemble = ray.get(state_ensemble)
            pars_ensemble = ray.get(pars_ensemble)

            state_ensemble = np.asarray(state_ensemble)
            pars_ensemble = np.asarray(pars_ensemble)
        else:
            state_ensemble = np.zeros((self.params['num_particles'], *state.shape))
            pars_ensemble = np.zeros((self.params['num_particles'], *pars.shape))
            for i in range(self.params['num_particles']):
                particle_state, particle_pars, t_vec = compute_prior_particle(
                        forward_model=self.forward_model, 
                        t_range=t_range,
                        state=state[i], 
                        pars=pars[i]
                        )
                state_ensemble[i] = particle_state
                pars_ensemble[i] = particle_pars

        return state_ensemble, pars_ensemble

    def _compute_likelihood(self, state_ensemble, observation):
        """Compute the model likelihood."""
        if False:#ray.is_initialized():
            likelihood_list = []
            for i in range(self.params['num_particles']):
                likelihood = compute_likelihood_ray.remote(
                    state=state_ensemble[i], 
                    observation=observation,
                    observation_operator=self.observation_operator,
                    likelihood_params=self.likelihood_params,
                    )
                likelihood_list.append(likelihood)
            likelihood = ray.get(likelihood_list)
        else:
            likelihood = np.zeros(self.params['num_particles'])
            for i in range(self.params['num_particles']):
                likelihood[i] = compute_likelihood(
                    state=state_ensemble[i], 
                    observation=observation,
                    observation_operator=self.observation_operator,
                    likelihood_params=self.likelihood_params,
                    )
        
        return likelihood
    

    def _get_resampled_particles(self, state_ensemble, pars_ensemble, weights):
        """Compute the posterior."""
        
        resampled_ids = np.random.multinomial(
            n=self.params['num_particles'],
            pvals=weights,
        )
        indeces = np.repeat(
            np.arange(self.params['num_particles']),
            resampled_ids
        )

        return state_ensemble[indeces], pars_ensemble[indeces]
    
    def compute_filtered_solution(
        self,
        true_sol,
        state_init,
        pars_init
    ):
        """Compute the filtered solution."""

        weights = self._restart_weights()
        
        self.forward_model.model_error.initialize_model_error_distribution(
            state=state_init,
            pars=pars_init
            )
        state_ensemble, pars_ensemble = self._compute_initial_ensemble(
                state_init=state_init,
                pars_init=pars_init
            )

        state_ensemble = np.expand_dims(state_ensemble, -1)
        pars_ensemble = np.expand_dims(pars_ensemble, -1)
        
        p_bar = tqdm(
            enumerate(zip(true_sol.obs_t[:-1], true_sol.obs_t[1:])),
            total=len(true_sol.obs_t[1:]),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )

        state_ensemble_filtered = state_ensemble.copy()
        pars_ensemble_filtered = pars_ensemble.copy()
        for i, (t_old, t_new) in p_bar:

            # Compute state mean
            state_mean = self._compute_mean_state(
                t_range=[t_old, t_new], 
                state=state_ensemble[:, :, :, -1],
                pars=pars_ensemble[:, :, -1]
                )

            prior_likelihood = self._compute_likelihood(
                state_ensemble=state_mean[:, :, :, -1],
                observation=true_sol.observations[:, :, i+1]
                )

            weights = self._update_weights_prior(
                likelihood=prior_likelihood,
                weights=weights
                )

            state_ensemble, pars_ensemble = \
                self._get_resampled_particles(
                    state_ensemble=state_ensemble,
                    pars_ensemble=pars_ensemble,
                    weights=weights
                )

            self.forward_model.model_error.update_model_error_distribution(
                state=state_ensemble[:, :, :, -1],
                pars=pars_ensemble[:, :, -1],
                weights=weights
                )
            
            # Compute the prior particles
            state_ensemble, pars_ensemble = self._compute_prior_particles(
                t_range=[t_old, t_new],
                state=state_ensemble[:, :, :, -1],
                pars=pars_ensemble[:, :, -1]
            )
            pars_ensemble = np.expand_dims(pars_ensemble, axis=-1)

            # Compute the likelihood
            likelihood = self._compute_likelihood(
                state_ensemble=state_ensemble[:, :, :, -1], 
                observation=true_sol.observations[:, :, i+1]
                )
            
            weights, ESS = self._update_weights(
                prior_likelihood=prior_likelihood,
                likelihood=likelihood,
                weights=weights
            )

            if ESS < self.ESS_threshold:
                state_ensemble, pars_ensemble = \
                    self._get_resampled_particles(
                        state_ensemble=state_ensemble,
                        pars_ensemble=pars_ensemble,
                        weights=weights
                    )
                weights = self._restart_weights()

                resample = True
            else:
                resample = False
            
            state_ensemble_filtered = np.concatenate(
                (state_ensemble_filtered, state_ensemble), 
                axis=-1
                )
            pars_ensemble_filtered = np.concatenate(
                (pars_ensemble_filtered, pars_ensemble), 
                axis=-1
                )
            
            p_bar.set_postfix({'Resample': resample})
            
        return state_ensemble_filtered, pars_ensemble_filtered
    