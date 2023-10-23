import pdb
import numpy as np
import pandas as pd
import torch

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
        observation_file: str = None,
        backend: str = 'numpy',
    ) -> None:

        self.state = state
        self.pars = pars
        self.observation_operator = observation_operator
        self.noise_variance = noise_variance
        self.observation_times = observation_times
        if len(self.observation_times) == 3:
            self.observation_times = range(
                self.observation_times[0],
                self.observation_times[1],
                self.observation_times[2],
            )

        if observation_file is not None:
            observations_file = pd.read_csv(observation_file, index_col=0).values

        self.num_observation_times = len(self.observation_times)

        self.full_t_vec = np.linspace(
            full_time_points[0], full_time_points[1], full_time_points[2]
        )
        self.observation_t_vec = self.full_t_vec[self.observation_times]

        self.observations = np.zeros(
            shape=(self.observation_operator.num_observations, self.num_observation_times)
        )
        for idx, time_idx in enumerate(self.observation_times):
            if observation_file is None:
                self.observations[:, idx] = self.observation_operator.get_observations(
                    state=self.state[:, :, time_idx]
                )
            else:
                self.observations[:, idx] = observations_file[time_idx]


        if observation_file is None:
            self.noise = np.random.normal(
                loc=0, scale=np.sqrt(self.noise_variance), size=self.observations.shape
            )

            self.observations += self.noise
        
        if backend == 'torch':
            self.observations = torch.tensor(self.observations)
            