        
import pdb

import numpy as np
from data_assimilation.particle_filter.base import BaseObservationOperator


class PDEObservationOperator(BaseObservationOperator):

    def __init__(
        self, 
        observation_state_ids: list,
        observation_space_ids: list,
        full_space_points: list = None,
        **kwargs
        ) -> None:
            
        super().__init__(**kwargs)

        self.observation_space_ids = observation_space_ids
        self.observation_state_ids = observation_state_ids

        self.num_observations = len(self.observation_space_ids)

        if full_space_points is not None:
            self.full_space_points = np.linspace(
                full_space_points[0], 
                full_space_points[1], 
                full_space_points[2]
                )
            
            self.x_observation_points = self.full_space_points[self.observation_space_ids]            

    def get_observations(self, state, ensemble=False):
        """Compute the observations."""

        if ensemble:
            return state[:, 
                self.observation_state_ids*self.num_observations, 
                self.observation_space_ids
                ]
        else:
            return state[
                self.observation_state_ids*self.num_observations, 
                self.observation_space_ids
                ]


class LatentObservationOperator(BaseObservationOperator):

    def __init__(
        self,
        observation_state_ids: list,
        observation_space_ids: list,
        full_space_points: list = None,
        **kwargs
        ) -> None:
            
        super().__init__(**kwargs)

        self.observation_space_ids = observation_space_ids
        self.observation_state_ids = observation_state_ids

        self.num_observations = len(self.observation_space_ids)

        if full_space_points is not None:
            self.full_space_points = np.linspace(
                full_space_points[0], 
                full_space_points[1], 
                full_space_points[2]
                )
            
            self.x_observation_points = self.full_space_points[self.observation_space_ids]            

    def get_observations(self, state, ensemble=False):
        """Compute the observations."""

        if ensemble:
            return state[:, 
                self.observation_state_ids*self.num_observations, 
                self.observation_space_ids
                ]
        else:
            return state[
                self.observation_state_ids*self.num_observations, 
                self.observation_space_ids
                ]