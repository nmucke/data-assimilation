

import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_assimilation.particle_filter.base import BaseLikelihood, BaseObservationOperator

from scipy.stats import norm

class NeuralNetworkLikelihood(BaseLikelihood):
    def __init__(
        self,
        observation_operator: BaseObservationOperator,
        noise_variance: float,
        **kwargs,
        ):
        
        super().__init__(**kwargs)

        self.observation_operator = observation_operator
        self.noise_variance = noise_variance
        
        #self.likelihood_distribution = torch.distributions.norma#l.Normal(
        #    loc=0, scale=np.sqrt(self.noise_variance)
        #    )                
        self.likelihood_distribution = norm(loc=0, scale=np.sqrt(self.noise_variance))   

    def compute_log_likelihood(
        self, 
        state, 
        observations
        ):

        model_observations = self.observation_operator.get_observations(
            state=state,
            ensemble=True
            )

        residual = observations - model_observations.detach().numpy()

        #log_likelihood = self.likelihood_distribution.log_prob(residual).sum()
        log_likelihood = self.likelihood_distribution.pdf(residual).sum(axis=1)

        return log_likelihood

class PDELikelihood(BaseLikelihood):
    def __init__(
        self,
        observation_operator: BaseObservationOperator,
        noise_variance: float,
        **kwargs,
        ):
        
        super().__init__(**kwargs)

        self.observation_operator = observation_operator
        self.noise_variance = noise_variance

        self.likelihood_distribution = norm(loc=0, scale=np.sqrt(self.noise_variance))            

    def compute_log_likelihood(
        self, 
        state, 
        observations
        ):
        
        model_observations = self.observation_operator.get_observations(
            state=state,
            ensemble=True
            )
        
        residual = observations - model_observations

        log_likelihood = self.likelihood_distribution.pdf(residual).sum(axis=1)

        return log_likelihood