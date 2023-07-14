    
from abc import abstractmethod, ABC
import pdb

import numpy as np
import tqdm

import matplotlib.pyplot as plt


class BaseForwardModel(ABC):
    
    @abstractmethod
    def update_params(self, params):
        
        raise NotImplementedError

    @abstractmethod
    def compute_forward_model(
        self,
        state_ensemble,
        pars_ensemble,
        t_range
        ):
        
        raise NotImplementedError
    
    @abstractmethod
    def initialize_state(self, pars):
            
        raise NotImplementedError

class BaseModelError(ABC):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        
        super().__init__()


    @abstractmethod
    def add_model_error(self, state_ensemble, pars_ensemble):

        raise NotImplementedError
    

    @abstractmethod
    def update(self, state_ensemble, pars_ensemble):

        raise NotImplementedError
        
class BaseObservationOperator(ABC):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        
        super().__init__()

    @abstractmethod
    def get_observations(self, state):

        raise NotImplementedError

class BaseLikelihood(ABC):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        
        super().__init__()

    @abstractmethod
    def compute_log_likelihood(
        self, 
        observations, 
        state, 
    ):

        raise NotImplementedError

class BaseParticleFilter(ABC):

    def __init__(
        self,
        particle_filter_args: dict,
        forward_model: BaseForwardModel,
        observation_operator: BaseObservationOperator,
        likelihood: BaseLikelihood,
        model_error: BaseModelError,
        ) -> None:
        
        super().__init__()

        self.num_particles = particle_filter_args['num_particles']
        self.ESS_threshold = particle_filter_args['ESS_threshold']

        self.forward_model = forward_model
        self.observation_operator = observation_operator
        self.likelihood = likelihood
        self.model_error = model_error

    @abstractmethod
    def _update_weights(self, **kwargs):
        """Update the weights of the particles."""
        raise NotImplementedError
    
    @abstractmethod
    def _initialize_particles(self, **kwargs):
        """Initialize the particles."""
        raise NotImplementedError

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
    

    def compute_filtered_solution(
        self, 
        true_solution,
        init_pars,
        transform_state = False,
        save_level = 2,
    ):
        """Compute the filtered solution."""

        self.ESS_threshold = self.num_particles / 2

        weights = self._update_weights(restart=True)

        state_ensemble, pars_ensemble = \
            self._initialize_particles(pars=init_pars)
        
        t_old = 0      

        pbar = tqdm.tqdm(
            enumerate(true_solution.observation_t_vec),
            total=true_solution.observation_t_vec.shape[0],
            bar_format = "{desc}: {percentage:.2f}%|{bar:20}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"#
            )                         
        for i, t_new in pbar:
                
                # Update the model error distribution
                self.model_error.update(
                    state_ensemble=state_ensemble[:, :, :, -1],
                    pars_ensemble=pars_ensemble[:, :, -1],
                )

                # Add model error to the particles
                prior_state_ensemble, prior_pars_ensemble = \
                    self.model_error.add_model_error(
                        state_ensemble=state_ensemble[:, :, :, -1:],
                        pars_ensemble=pars_ensemble[:, :, -1:],
                    )
                
                # Compute the prior particles
                prior_state_ensemble, t_vec = self._compute_prior_particles(
                    state_ensemble=prior_state_ensemble[:, :, :, -1],
                    pars_ensemble=prior_pars_ensemble[:, :, -1],
                    t_range=[t_old, t_new],
                )

                if transform_state:
                    prior_state_ensemble_transformed = \
                        self.forward_model.transform_state(
                            state=prior_state_ensemble[:, :, :, -1],
                            x_points=self.observation_operator.full_space_points,
                        )
                else:
                    prior_state_ensemble_transformed = prior_state_ensemble[:, :, :, -1]

                # Compute the likelihood    
                likelihood = self.likelihood.compute_log_likelihood(
                    state=prior_state_ensemble_transformed, 
                    observations=true_solution.observations[:, i],
                )

                # Update the particle weights
                weights, ESS = self._update_weights(
                    likelihood=likelihood,
                    weights=weights,
                )

                print(f'ESS: {ESS}')
                print(f'ESS threshold: {self.ESS_threshold}')
                if ESS < self.ESS_threshold:
                    posterior_state_ensemble, posterior_pars_ensemble = \
                        self._resample(
                            state_ensemble=prior_state_ensemble,
                            pars_ensemble=prior_pars_ensemble,
                            weights=weights,
                        )
                    weights = self._update_weights(restart=True)

                    print('Resampling')
                else:
                    posterior_state_ensemble = prior_state_ensemble
                    posterior_pars_ensemble = prior_pars_ensemble
                
                lol = self.forward_model.transform_state(
                    posterior_state_ensemble[:, :, :, -1],
                    x_points=self.observation_operator.full_space_points,
                    )
                plt.figure()
                for j in range(posterior_state_ensemble.shape[0]):
                    
                    plt.plot(
                        np.linspace(0, 5000, 512),
                        lol[j, 2],
                        )
                    plt.plot(
                        np.linspace(0, 5000, 512),
                        true_solution.true_state[2, :, true_solution.observation_times[i]], 
                        '--', linewidth=3., color='black'
                        )    
                plt.show()

                if save_level == 0:
                    state_ensemble = posterior_state_ensemble[:, :, :, -1:]
                    pars_ensemble = posterior_pars_ensemble[:, :, -1:]
                elif save_level == 1:
                    state_ensemble = np.concatenate(
                        (state_ensemble, posterior_state_ensemble[:, :, :, -1:]), 
                        axis=-1
                        )
                    pars_ensemble = np.concatenate(
                        (pars_ensemble, posterior_pars_ensemble[:, :, -1:]), 
                        axis=-1
                        )
                elif save_level == 2:
                    state_ensemble = np.concatenate(
                        (state_ensemble, posterior_state_ensemble), 
                        axis=-1
                        )
                    pars_ensemble = np.concatenate(
                        (pars_ensemble, posterior_pars_ensemble), 
                        axis=-1
                        )
                    
                t_old = t_new

        return state_ensemble, pars_ensemble



        

