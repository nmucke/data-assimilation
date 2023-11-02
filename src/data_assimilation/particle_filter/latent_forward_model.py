


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

from torch.utils.data import DataLoader, Dataset, TensorDataset

class LoadInitialCondition(Dataset):
    
        def __init__(
            self, 
            initial_condition_path, 
            num_particles, 
            space_dim,
            num_previous_steps,
            matlab=False,
        ):
    
            self.initial_condition_path = initial_condition_path
            self.num_particles = num_particles
            self.space_dim = space_dim
            self.num_previous_steps = num_previous_steps
            self.matlab = matlab
    
        def __len__(self):
            return self.num_particles
    
        def __getitem__(self, idx):

            if self.matlab:
                state = loadmat(f'{self.initial_condition_path}/state/sample_{idx}.mat')['state']
                state = state[:, :, 0:self.num_previous_steps]
        
                pars = loadmat(f'{self.initial_condition_path}/pars/sample_{idx}.mat')['pars'][0]

            else:
                
                state = np.load(f'{self.initial_condition_path}/state/sample_{idx}.npz')
                state = state['data'][:, :, 0:self.num_previous_steps]

                pars = np.load(f'{self.initial_condition_path}/pars/sample_{idx}.npz')
                pars = pars['data']

            state = torch.tensor(state, dtype=torch.float32)
            pars = torch.tensor(pars, dtype=torch.float32)
    
            return state, pars


class LatentForwardModel(BaseForwardModel):

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
        matlab=False,
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
        self.matlab = matlab
        self.model_type = 'latent'

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
    
    def transform_state(self, state, x_points, pars, numpy=False):
        out_state = torch.zeros(
            (self.num_particles, self.num_PDE_states, self.space_dim, 1),
            dtype=torch.float32,
        )

        for batch_idx in range(0, self.num_particles, self.batch_size):
            
            batch_state = state[batch_idx:batch_idx+self.batch_size].to(self.device)
            batch_pars = pars[batch_idx:batch_idx+self.batch_size].to(self.device)

            batch_state = self.AE_model.decode(batch_state, batch_pars)

            out_state[batch_idx:batch_idx+self.batch_size] = batch_state.cpu()

        out_state = self.preprocesssor.inverse_transform_state(out_state, ensemble=True)

        if numpy:
            return out_state.squeeze(-1).detach().numpy()
        else:
            return out_state.squeeze(-1)
    

    def transform_pars(self, pars):

        pars = self.preprocesssor.inverse_transform_pars(pars, ensemble=True)

        return pars.cpu().detach().numpy()
            
    def initialize_state(self, pars):

        initial_condition_dataset = LoadInitialCondition(
            initial_condition_path=self.initial_condition_path,
            num_particles=self.num_particles,
            space_dim=self.space_dim,
            num_previous_steps=self.num_previous_steps,
            matlab=self.matlab,
        )
        initial_condition_loader = DataLoader(
            initial_condition_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

        out_state = torch.zeros(
            (self.num_particles, self.latent_dim, self.num_previous_steps),
            dtype=torch.float32,
        )
        pars = np.zeros(
            (self.num_particles, self.num_pars),
            dtype=np.float32,
        )
        with torch.no_grad():
            for batch_idx, (batch_state, batch_pars) in enumerate(initial_condition_loader):
                    
                batch_state = batch_state.to(self.device)
                batch_pars = batch_pars.to(self.device)

                batch_state = self.preprocesssor.transform_state(batch_state, ensemble=True)

                batch_state = self.AE_model.encode(batch_state)

                out_state[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size] = batch_state.cpu()
                pars[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size] = batch_pars.cpu()

        pars = torch.tensor(pars, dtype=torch.float32)#, device=self.device)
        pars = self.preprocesssor.transform_pars(pars, ensemble=True)
        pars = pars.unsqueeze(-1)
        pars = pars.repeat(1, 1, out_state.shape[-1])

        '''

        pars = np.zeros(
            (self.num_particles, self.num_pars),
            dtype=np.float32,
        )

        out_state = torch.zeros(
            (self.num_particles, self.latent_dim, self.num_previous_steps),
            dtype=torch.float32,
        )
        for batch_idx in range(0, self.num_particles, self.batch_size):

            batch_state = np.zeros(
                (self.batch_size, self.num_PDE_states, self.space_dim, self.num_previous_steps),
                dtype=np.float32,
            )

            for i, idx in enumerate(range(batch_idx, batch_idx+self.batch_size)): 
                
                state_i = loadmat(f'{self.initial_condition_path}/state/sample_{idx}.mat')['state']
                batch_state[i] = state_i[:, :, 0:self.num_previous_steps]

                pars_i = loadmat(f'{self.initial_condition_path}/pars/sample_{idx}.mat')['pars'][0]
                pars[i] = pars_i
            
            batch_state = torch.tensor(batch_state, dtype=torch.float32)
            batch_state = self.preprocesssor.transform_state(batch_state, ensemble=True)
            pdb.set_trace()
            batch_state = batch_state.to(self.device)
            batch_state = self.AE_model.encode(batch_state)
            
            out_state[batch_idx:batch_idx+self.batch_size] = batch_state.cpu()
        
        pars = torch.tensor(pars, dtype=torch.float32)#, device=self.device)
        pars = self.preprocesssor.transform_pars(pars, ensemble=True)
        pars = pars.unsqueeze(-1)
        pars = pars.repeat(1, 1, out_state.shape[-1])
        '''

        return out_state, pars

    def compute_forward_model(
        self, 
        state_ensemble, 
        pars_ensemble, 
        t_range=None, 
        output_seq_len=None,
        with_grad=False,
        ):
        
        if output_seq_len is None:
            output_seq_len = int((t_range[-1] - t_range[0]) // self.step_size)

        out_state = torch.zeros(
            (self.num_particles, self.latent_dim, output_seq_len),
            dtype=torch.float32,
        )

        with torch.no_grad():
            for batch_idx in range(0, self.num_particles, self.batch_size):
                batch_state = state_ensemble[batch_idx:batch_idx+self.batch_size].to(self.device)
                batch_pars = pars_ensemble[batch_idx:batch_idx+self.batch_size].to(self.device)

                batch_state_ensemble = self.time_stepping_model.multistep_prediction(
                    input=batch_state, 
                    pars=batch_pars, 
                    output_seq_len=output_seq_len,
                )

                out_state[batch_idx:batch_idx+self.batch_size] = batch_state_ensemble.cpu()

        return out_state, output_seq_len*self.step_size