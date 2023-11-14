


import pdb
from scipy.io import loadmat

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
        num_skip_steps = 10,
        PDE_step_size: float = 0.01,
        space_dim: int = 512,
        x_max: float = 5000.,
        batch_size: int = 256,
        num_pars: int = 1,
        num_PDE_states: int = 3,
        initial_condition_path: str = None,
        **kwargs
        ):

        super().__init__(**kwargs)

        self.num_particles = num_particles
        self.initial_condition_path = initial_condition_path
        self.num_skip_steps = num_skip_steps
        self.PDE_step_size = PDE_step_size
        self.space_dim = space_dim
        self.x_max = x_max
        self.num_PDE_states = num_PDE_states
        self.num_pars = num_pars
        self.batch_size = batch_size
        self.step_size = num_skip_steps * PDE_step_size
        self.model_args = model_args
        self.device = device
        self.num_previous_steps = model_args['num_previous_steps']

        object_storage_client = ObjectStorageClientWrapper(
            bucket_name='trained_models'
        )

        ##### load AE model #####
        state_dict, config = object_storage_client.get_model(
            source_path=model_args['AE_model_path'],
            device=device,
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

        self.latent_dim = self.AE_model.decoder.latent_dim

    def update_params(self, params):
        pass
    
    def transform_state(self, state, x_points, pars):

        out_state = torch.zeros(
            (self.num_particles, self.num_PDE_states, self.space_dim, 1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            for batch_idx in range(0, self.num_particles, self.batch_size):
                    
                batch_state = state[batch_idx:batch_idx+self.batch_size].to(self.device)
                batch_pars = pars[batch_idx:batch_idx+self.batch_size].to(self.device)
    
                batch_state = self.AE_model.decode(batch_state, batch_pars)
    
                out_state[batch_idx:batch_idx+self.batch_size] = batch_state.cpu()

        out_state = self.preprocesssor.inverse_transform_state(out_state, ensemble=True)

        return out_state.squeeze(-1).detach().numpy()
    

    def transform_pars(self, pars):

        pars = self.preprocesssor.inverse_transform_pars(pars, ensemble=True)

        return pars.cpu().detach().numpy()
             
    def initialize_state(self, pars):

        state = np.zeros(
            (self.num_particles, self.num_PDE_states, self.space_dim, self.num_previous_steps),
            dtype=np.float32,
        )
        pars = np.zeros(
            (self.num_particles, self.num_pars),
            dtype=np.float32,
        )
        for i in range(self.num_particles):
            
            state_i = np.load(f'{self.initial_condition_path}/state/sample_{i}.npz')
            state_i = state_i['data'][:, :, -self.num_previous_steps:]
            state[i] = state_i

            pars_i = np.load(f'{self.initial_condition_path}/pars/sample_{i}.npz')
            pars[i] = pars_i['data']

        state = torch.tensor(state, dtype=torch.float32)

        state = self.preprocesssor.transform_state(state, ensemble=True)
        #with torch.cuda.amp.autocast():
        out_state = torch.zeros(
            (self.num_particles, self.latent_dim, self.num_previous_steps),
            dtype=torch.float32,
        )
        with torch.no_grad():
            for batch_idx in range(0, self.num_particles, self.batch_size):

                batch_state = state[batch_idx:batch_idx+self.batch_size].to(self.device)

                batch_state = self.AE_model.encode(batch_state)
                
                out_state[batch_idx:batch_idx+self.batch_size] = batch_state.cpu()
        

        pars = torch.tensor(pars, dtype=torch.float32)#, device=self.device)

        pars = self.preprocesssor.transform_pars(pars, ensemble=True)

        pars = pars.unsqueeze(-1)
        pars = pars.repeat(1, 1, state.shape[-1])



        return out_state, pars

    def compute_forward_model(self, state_ensemble, pars_ensemble, t_range):

        output_seq_len = int((t_range[-1] - t_range[0]) // self.step_size)

        sol = torch.zeros(
            (self.num_particles, self.latent_dim, output_seq_len),
        )

        with torch.no_grad():
            for batch_idx in range(0, self.num_particles, self.batch_size):

                batch_state_ensemble = state_ensemble[batch_idx:batch_idx+self.batch_size].to(self.device)
                batch_pars_ensemble = pars_ensemble[batch_idx:batch_idx+self.batch_size].to(self.device)

                batch_state_ensemble = self.time_stepping_model.multistep_prediction(
                    input=batch_state_ensemble, 
                    pars=batch_pars_ensemble, 
                    output_seq_len=output_seq_len,
                )

                sol[batch_idx:batch_idx+self.batch_size] = batch_state_ensemble.cpu()

        #sol = self.time_stepping_model.multistep_prediction(
        #    input=state_ensemble, 
        #    pars=pars_ensemble, 
        #    output_seq_len=output_seq_len,
        #    )
                
        return sol, output_seq_len*self.step_size