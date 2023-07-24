import pdb
import numpy as np

from data_assimilation.particle_filter.base import BaseObservationOperator

class TrueSolution():
    def __init__(
        self,
        state: np.ndarray,
        pars: np.ndarray,
        observation_operator: BaseObservationOperator,
        noise_variance: float,
        observation_times: list,
        full_time_points: list,
    ) -> None:

        self.state = state
        self.pars = pars
        self.observation_operator = observation_operator
        self.noise_variance = noise_variance
        self.observation_times = observation_times

        self.num_observation_times = len(self.observation_times)

        self.full_t_vec = np.linspace(
            full_time_points[0], full_time_points[1], full_time_points[2]
        )
        self.observation_t_vec = self.full_t_vec[self.observation_times]

        self.observations = np.zeros(
            shape=(self.observation_operator.num_observations, self.num_observation_times)
        )
        for idx, time_idx in enumerate(self.observation_times):
            self.observations[:, idx] = self.observation_operator.get_observations(
                state=self.state[:, :, time_idx]
            )

        self.noise = np.random.normal(
            loc=0, scale=np.sqrt(self.noise_variance), size=self.observations.shape
        )

        self.observations += self.noise