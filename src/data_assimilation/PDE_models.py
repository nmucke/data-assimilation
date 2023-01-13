from discontinuous_galerkin.base.base_model import BaseModel
import numpy as np

class AdvectionEquation(BaseModel):
    """Advection equation model class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_params = {
            'advection_velocity': 0.5*np.pi,
            'inflow_frequence': 2 * np.pi,
        }
    
    def initial_condition(self, x):
        """Compute the initial condition."""

        init = np.sin(x)
        init = np.expand_dims(init, 0)

        return init
    
    def boundary_conditions(self, t, q):
        """Compute the boundary conditions."""

        BC_state_1 = {
            'left': -np.sin(self.model_params['inflow_frequence'] * t),
            'right': None,
        }

        BCs = [BC_state_1]
        
        return BCs
        
    def flux(self, q):
        """Compute the flux."""

        return self.model_params['advection_velocity']*q
    
    def velocity(self, q):
        return self.model_params['advection_velocity']

    def source(self, t, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))
