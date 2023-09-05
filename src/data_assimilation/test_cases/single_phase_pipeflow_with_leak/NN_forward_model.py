


import pdb

import numpy as np
import torch
from data_assimilation.oracle import ObjectStorageClientWrapper
from data_assimilation.particle_filter.base import BaseForwardModel
from latent_time_stepping.utils import (
    load_trained_AE_model, 
    load_trained_time_stepping_model
)


class NNForwardModel(BaseForwardModel):

    def __init__(
        self,
        model_args: dict,
        device: str,
        distributed: bool = False,
        num_particles: int = 100,
        **kwargs
        ):

        super().__init__(**kwargs)

        self.step_size = 0.05
        self.num_particles = num_particles

        self.model_args = model_args

        self.device = device

        object_storage_client = ObjectStorageClientWrapper(
            bucket_name='trained_models'
        )

        ##### load AE model #####
        state_dict, config = object_storage_client.get_model(
            source_path=model_args['AE_model_path'],
        )   
        self.AE_model = load_trained_AE_model(
            state_dict=state_dict,
            config=config,
            model_type=model_args['AE_model_type'],
            device=device,
            )   
        self.AE_model.eval()                  
        
        ##### load time stepping model #####
        state_dict, config = object_storage_client.get_model(
            source_path=model_args['time_stepping_model_path'],
        )
        self.time_stepping_model = load_trained_time_stepping_model(
            state_dict=state_dict,
            config=config,
            device=device,
        )
        self.time_stepping_model.eval()


        ##### Load preoprocesser #####
        self.preprocesssor = object_storage_client.get_preprocessor(
            source_path=model_args['preprocessor_path'],
        )



    def update_params(self, params):
        pass
    
    def transform_state(self, state, x_points, pars):

        state = self.AE_model.decode(state, pars)
        state = self.preprocesssor.inverse_transform_state(state, ensemble=True)

        return state
            
    def initialize_state(self, pars):

        state = np.load(f'data/{self.model_args["phase"]}_phase_pipeflow_with_leak/initial_conditions/states.npz')
        state = state['data'][0:self.num_particles]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        state = self.preprocesssor.transform_state(state, ensemble=True)
        state = self.AE_model.encode(state)

        pars = np.load('data/{self.model_args["phase"]}_phase_pipeflow_with_leak/initial_conditions/pars.npz')
        pars = pars['data'][0:self.num_particles]
        pars = torch.tensor(pars, dtype=torch.float32, device=self.device)

        pars = self.preprocesssor.transform_pars(pars, ensemble=True)

        pars = pars.unsqueeze(-1)
        pars = pars.repeat(1, 1, state.shape[-1])

        return state, pars

    def compute_forward_model(self, state_ensemble, pars_ensemble, t_range):

        output_seq_len = int((t_range[-1] - t_range[0]) / self.step_size)

        sol = self.time_stepping_model.multistep_prediction(
            input=state_ensemble, 
            pars=pars_ensemble, 
            output_seq_len=output_seq_len,
            )
                
        return sol, output_seq_len*self.step_size