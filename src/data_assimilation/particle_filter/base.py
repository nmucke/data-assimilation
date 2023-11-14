    
from abc import abstractmethod, ABC
import pdb

import numpy as np
import ray
import torch
import tqdm

import matplotlib.pyplot as plt

from data_assimilation.utils import create_directory


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
    def compute_likelihood(
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
        backend: str = 'numpy',
        save_folder: str = None,
        ) -> None:
        
        super().__init__()

        self.backend = backend

        self.num_particles = particle_filter_args['num_particles']
        self.ESS_threshold = particle_filter_args['ESS_threshold']

        self.forward_model = forward_model
        self.observation_operator = observation_operator
        self.likelihood = likelihood
        self.model_error = model_error
        self.model_type = self.forward_model.model_type
        self.save_folder = save_folder

        self.state_save_folder = self.save_folder + '/state'
        self.pars_save_folder = self.save_folder + '/pars'

        create_directory(self.state_save_folder)
        create_directory(self.pars_save_folder)

        if self.model_type in ['FNO', 'latent']:
            self.num_previous_steps = self.forward_model.num_previous_steps
            

    @abstractmethod
    def _get_posterior(self, **kwargs):
        """Update the weights of the particles."""
        raise NotImplementedError
    
    def _initialize_particles(self, pars=None, **kwargs):
        
        state_ensemble, pars_ensemble = self.forward_model.initialize_state(
            pars=pars
        )

        return state_ensemble, pars_ensemble
    
    def compute_filtered_solution(
        self, 
        true_solution,
        init_pars,
        save_level = 2,
        distributed = False,
        num_workers = 1,
    ):
        """Compute the filtered solution."""
        
        state_ensemble, pars_ensemble = self._initialize_particles(pars=init_pars)
        t_old = 0      


        if self.model_type != 'PDE':
            t_old = (self.num_previous_steps-1) * self.forward_model.step_size
            
        if self.model_type in ['latent']:
            if save_level == 1:
                out_state_ensemble = [state_ensemble[:, :, 0]]
                out_pars_ensemble = [pars_ensemble[:, :, 0]]
            elif save_level == 2:
                out_state_ensemble = [state_ensemble]
                out_pars_ensemble = [pars_ensemble]

        pbar = tqdm.tqdm(
            enumerate(true_solution.observation_t_vec),
            total=true_solution.observation_t_vec.shape[0],
            bar_format = "{desc}: {percentage:.2f}%|{bar:20}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"#
        )

        #state_observations = []

        for i, t_new in pbar:
            
            #if ray.is_initialized():
            #    ray.shutdown()
            #if distributed:
            #    ray.init(num_cpus=num_workers)
            posterior_state_ensemble, posterior_pars_ensemble = \
                self._get_posterior(
                    state_ensemble=state_ensemble,
                    pars_ensemble=pars_ensemble,
                    observations=true_solution.observations[:, i],
                    t_range=[t_old, t_new],
                )
            
            '''
            observations = self.observation_operator.get_observations(
                posterior_state_ensemble,
                ensemble=True,
            )

            state_observations.append(observations)
            '''

            t_old = t_new

            if self.model_type in ['latent', 'neural_network']: 
                state_ensemble = posterior_state_ensemble[:, :, -self.num_previous_steps:]
                pars_ensemble = posterior_pars_ensemble[:, :, -self.num_previous_steps:]
                
                if save_level == 1:
                    np.savez_compressed(
                        f'{self.state_save_folder}/state_{i}.npz', 
                        data=self.forward_model.transform_state(
                            posterior_state_ensemble[:, :, -1:],
                            x_points=self.observation_operator.full_space_points,
                            pars=pars_ensemble[:, :, -1]
                        ).detach().numpy()
                    )
                    np.savez_compressed(
                        f'{self.pars_save_folder}/pars_{i}.npz', 
                        data=self.forward_model.transform_pars(
                            pars_ensemble[:, :, -1:]                                                                                                                   
                        )
                    )
                    
            elif self.model_type in ['PDE', 'FNO']: 
                state_ensemble = posterior_state_ensemble[:, :, :, -1:]
                pars_ensemble = posterior_pars_ensemble[:, :, -1:]

                if save_level == 1:
                    np.savez_compressed(
                        f'{self.state_save_folder}/state_{i}.npz', 
                        data=self.forward_model.transform_state(
                            posterior_state_ensemble[:, :, :, -1:],
                            x_points=self.observation_operator.full_space_points,
                            pars=pars_ensemble[:, :, -1:]
                        )
                    )
                    np.savez_compressed(
                        f'{self.pars_save_folder}/pars_{i}.npz', 
                        data=pars_ensemble[:, :, -1:]                                                                                              
                    )
        if self.model_type in ['latent', 'neural_network']: 
            out_state_ensemble = self.forward_model.transform_state(
                posterior_state_ensemble[:, :, -1:],
                x_points=self.observation_operator.full_space_points,
                pars=pars_ensemble[:, :, -1]
            ).detach().unsqueeze(-1).numpy()

            out_pars_ensemble = self.forward_model.transform_pars(
                pars_ensemble[:, :, -1:]                                                                                                                   
            )

        elif self.model_type in ['PDE', 'FNO']:
            out_state_ensemble = self.forward_model.transform_state(
                posterior_state_ensemble[:, :, :, -1:],
                x_points=self.observation_operator.full_space_points,
                pars=pars_ensemble[:, :, -1:]
            )
            out_state_ensemble = np.expand_dims(out_state_ensemble, axis=-1)

            out_pars_ensemble = pars_ensemble[:, :, -1:]

        #if ray.is_initialized():
        #    ray.shutdown()

        return out_state_ensemble, out_pars_ensemble, 0

        '''
        if self.model_type in ['latent', 'neural_network']: 
            state_ensemble = torch.cat(
                (state_ensemble, posterior_state_ensemble), 
                dim=-1
            )
            pars_ensemble = torch.cat(
                (
                    pars_ensemble, 
                    posterior_pars_ensemble if save_level == 0 or 1 else \
                    posterior_pars_ensemble.repeat(1, 1, posterior_state_ensemble.shape[-1])
                ), 
                dim=-1
            )
            if save_level == 1:
                out_state_ensemble.append(state_ensemble[:, :, -1])
                out_pars_ensemble.append(pars_ensemble[:, :, -1])

            elif save_level == 2:
                out_state_ensemble.append(state_ensemble)
                out_pars_ensemble.append(pars_ensemble)

        if self.model_type == 'PDE':
            if save_level == 0:
                state_ensemble = posterior_state_ensemble[:, :, :, -1:]
                pars_ensemble = posterior_pars_ensemble[:, :, -1:]

            if save_level == 1:
                state_ensemble = np.concatenate(
                    (state_ensemble, posterior_state_ensemble[:, :, :, -1:]), 
                    axis=-1
                )
                pars_ensemble = np.concatenate(
                    (pars_ensemble, posterior_pars_ensemble[:, :, -1:]), 
                    axis=-1
                )
                
            if save_level == 2:
                state_ensemble = np.concatenate(
                    (state_ensemble, posterior_state_ensemble), 
                    axis=-1
                )
                pars_ensemble = np.concatenate(
                    (pars_ensemble, posterior_pars_ensemble), 
                    axis=-1
                )

        '''

        '''
        if self.model_type in ['latent', 'neural_network']:
            if save_level == 0:
                return state_ensemble[:, :, -1:], pars_ensemble[:, :, -1:]
            elif save_level == 1:
                return torch.stack(out_state_ensemble, dim=-1), torch.stack(out_pars_ensemble, dim=-1)
            elif save_level == 2:
                return torch.cat(out_state_ensemble, dim=-1), torch.cat(out_pars_ensemble, dim=-1)
        '''
        #state_observations = np.concatenate(state_observations, axis=-1)
        
        #return state_ensemble, pars_ensemble, 0 #state_observations



        

