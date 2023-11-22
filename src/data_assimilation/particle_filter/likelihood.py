

import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_assimilation.particle_filter.base import BaseLikelihood, BaseObservationOperator

from scipy.stats import norm, multivariate_normal
'''
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

        residual = observations - model_observations

        #log_likelihood = self.likelihood_distribution.log_prob(residual).sum()
        log_likelihood = self.likelihood_distribution.pdf(residual).sum(axis=1)

        return log_likelihood
'''

class Likelihood(BaseLikelihood):
    def __init__(
        self,
        observation_operator: BaseObservationOperator,
        noise_variance: float,
        multivariate: bool = False,
        backend: str = 'numpy',
        **kwargs,
        ):
        
        super().__init__(**kwargs)

        self.backend = backend

        self.observation_operator = observation_operator
        self.noise_variance = noise_variance
        self.multivariate = multivariate


        if self.backend == 'torch':
            if self.multivariate:
                self.likelihood_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=torch.zeros(observation_operator.num_observations), 
                    covariance_matrix=self.noise_variance*torch.eye(observation_operator.num_observations)
                )
            else:
                self.likelihood_distribution = torch.distributions.normal.Normal(
                    loc=0, scale=np.sqrt(self.noise_variance)
                )
        else:
            if self.multivariate:
                self.likelihood_distribution = multivariate_normal(
                    mean=np.zeros(observation_operator.num_observations), 
                    cov=self.noise_variance*np.eye(observation_operator.num_observations)
                )         
            else:
                self.likelihood_distribution = norm(loc=0, scale=np.sqrt(self.noise_variance))   

    def compute_likelihood(
        self, 
        state, 
        observations,
    ):
        
        model_observations = self.observation_operator.get_observations(
            state=state,
            ensemble=True
        )    
        
        residual = observations - model_observations


        if self.backend == 'torch':
            if self.multivariate:
                likelihood = torch.exp(self.likelihood_distribution.log_prob(residual))
            else:
                likelihood = torch.exp(self.likelihood_distribution.log_prob(residual).sum(axis=1))

        else:
            if self.multivariate:
                likelihood = self.likelihood_distribution.pdf(residual)
            else:
                likelihood = self.likelihood_distribution.pdf(residual).sum(axis=1)



        return likelihood