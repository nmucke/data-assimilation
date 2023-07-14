import pdb
import numpy as np

from data_assimilation.base import BaseObservationOperator

class TrueSolution():
    def __init__(
        self,
        true_state: np.ndarray,
        true_pars: np.ndarray,
        observation_operator: BaseObservationOperator,
        noise_variance: float,
        observation_times: list,
    ) -> None:

        self.true_state = true_state
        self.true_pars = true_pars
        self.observation_operator = observation_operator
        self.noise_variance = noise_variance
        self.observation_times = observation_times

        self.num_observation_times = len(self.observation_times)

        self.full_t_vec = np.linspace(
            0, 250, 25000
        )
        self.observation_t_vec = self.full_t_vec[self.observation_times]

        self.observations = np.zeros(
            shape=(self.observation_operator.num_observations, self.num_observation_times)
        )
        for idx, time_idx in enumerate(self.observation_times):
            self.observations[:, idx] = self.observation_operator.get_observations(
                state=self.true_state[:, :, time_idx]
            )

        self.noise = np.random.normal(
            loc=0, scale=np.sqrt(self.noise_variance), size=self.observations.shape
        )

        self.observations += self.noise