
import numpy as np
from data_assimilation.forward_model import BaseForwardModel
from data_assimilation.model_error import BaseModelError
from data_assimilation.observation_operator import BaseObservationOperator
import pdb

class ModelError(BaseModelError):
    def __init__(self, params, state_dims=None, pars_dims=None):
        
        self.params = params
        self.state_dims = state_dims
        self.pars_dims = pars_dims

        self.smoothing_factor = self.params['smoothing_factor']
        self.a = np.sqrt(1 - self.smoothing_factor)

    def get_initial_ensemble(self, state, pars):
        """Get the initial model error."""

        pars = np.random.uniform(
            low=pars - 10*self.params['pars_std'],
            high=pars + 10*self.params['pars_std'],
            size=(self.pars_dims),
            )

        state_noise = np.random.normal(
            loc=0.,
            scale=self.params['state_std'],
            size=self.state_dims,
            )
        state = state + state_noise

        return state, pars
        


    def initialize_model_error_distribution(self, state, pars):
        """Initialize the model error."""

        self.pars_weighted_mean = pars
        self.pars_covariance = self.params['pars_std'] * np.eye(self.pars_dims)


    def update_model_error_distribution(self, state, pars, weights):
        """Update the model error distribution."""

        self.pars_weighted_mean = np.dot(pars.T, weights)

        pars_mean_shifted = pars - self.pars_weighted_mean

        self.pars_covariance = np.dot(
            pars_mean_shifted.T, 
            np.dot(np.diag(weights), pars_mean_shifted)
            )

    def _sample_noise(self, state, pars):
        """Sample noise."""

        state_noise = np.random.normal(
            loc=0.,
            scale=self.params['state_std'],
            size=self.state_dims,
            )

        pars_mean = self.a*pars + (1 - self.a)*self.pars_weighted_mean
        
        pars_noise = np.random.multivariate_normal(
            mean=pars_mean,
            cov=self.smoothing_factor*self.pars_covariance,
            )
        
        return state_noise, pars_noise

    def get_model_error(self, state, pars):
        """Compute the model error."""

        state_noise, pars_noise = self._sample_noise(state, pars)

        return state_noise, pars_noise
    
    def add_model_error(self, state, pars):
        """Add the model error to the state and parameters."""

        state_noise, pars_noise = self.get_model_error(state, pars)

        state = state + state_noise
        #pars = pars + pars_noise
        pars = pars_noise

        return state, pars

class AdvectionForwardModel(BaseForwardModel):
    def __init__(
        self, 
        model,
        model_error_params=None,
        ):
        super().__init__()

        self.model = model
        if model_error_params is not None:
            self.model_error_params = model_error_params
            self.model_error = ModelError(
                self.model_error_params,
                state_dims=self.model.DG_vars.num_states*self.model.DG_vars.Np*self.model.DG_vars.K,
                pars_dims=2,
                )

    def initialize_forward_model(self, state, pars):
        """Initialize the forward model."""

        self.pars = pars
        self.model.model_params['advection_velocity'] = pars[0]
        self.model.model_params['inflow_frequence'] = pars[1]

    def update_params(self, pars): 
        self.pars = pars
        self.model.model_params['advection_velocity'] = pars[0]
        self.model.model_params['inflow_frequence'] = pars[1]

    def compute_forward_model(self, t_range, state):
        """Compute the forward model."""

        # Compute the forward model.
        state, t_vec = self.model.solve(
            t=t_range[0],
            t_final=t_range[-1],
            q_init=state,
            print_progress=False,
            )
        t_vec = np.array(t_vec)

        return state, t_vec


class ObservationOperator(BaseObservationOperator):
    def __init__(self, params):
        
        self.params = params

    def get_observations(self, state):
        """Compute the observations."""

        if len(state.shape) == 4:
            return state[:, :, self.params['observation_index']]
        else:
            return state[:, self.params['observation_index']]


class TrueSolution():
    """True solution class."""

    def __init__(
        self, 
        x_vec: np.ndarray,
        t_vec: np.ndarray,
        sol: np.ndarray,
        pars: np.ndarray,
        observation_operator: BaseObservationOperator,
        obs_times_idx: np.ndarray,
        observation_noise_params: dict,
        ) -> None:

        self.sol = sol
        self.pars = pars

        self.x_vec = x_vec 
        self.t_vec = t_vec

        self.obs_times_idx = obs_times_idx
        self.observation_operator = observation_operator
        self.observation_noise_params = observation_noise_params

        self.observations = self.get_observations(with_noise=True)
        self.obs_x = self.x_vec[self.observation_operator.params['observation_index']]
        self.obs_t = self.t_vec[obs_times_idx]
    
    def get_observations(self, with_noise=True):
        """Compute the observations."""
    
        observations = self.observation_operator.get_observations(
            self.sol[:, :, self.obs_times_idx]
            )

        if not with_noise:
            return observations
        else:
            observations_noise = np.random.normal(
                loc=0.,
                scale=self.observation_noise_params['std'],
                size=observations.shape,
                )

            return observations + observations_noise


