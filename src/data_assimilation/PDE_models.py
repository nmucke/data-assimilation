import pdb
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


class PipeflowEquations(BaseModel):
    """Advection equation model class."""

    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)

        self.L = 2000
        self.d = 0.508
        self.A = np.pi*self.d**2/4
        self.c = 308.
        self.p_amb = 101325.
        self.p_ref = 5016390.
        self.rho_ref = 52.67
        self.e = 1e-8
        self.mu = 1.2e-5
        self.Cd = 5e-4
        self.xl = 500

        self.model_params = {
            'leak_location': self.xl,
            'leak_size': self.Cd,
        }


    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.c**2*(rho - self.rho_ref) + self.p_ref

    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return (p - self.p_ref)/self.c**2 + self.rho_ref

    def friction_factor(self, q):
        """Compute the friction factor."""

        rho = q[0]/self.A
        u = q[1]/q[0]

        Re = rho * u * self.d / self.mu
        
        f = (self.e/self.d/3.7)**(1.11) + 6.9/Re
        f *= -1/4*1.8*np.log10(f)**(-2)
        f *= -1/2/self.d * rho * u*u 

        return f


    def system_jacobian(self, q):

        u = q[1]/q[0]

        J =np.array(
            [[0, 1],
            [-u*u + self.c*self.c/np.sqrt(self.A), 2*u]]
            )

        return J

    def initial_condition(self, x):
        """Compute the initial condition."""
        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0] = self.pressure_to_density(self.p_ref) * self.A
        init[1] = init[0] * 4.0

        return init
    
    def boundary_conditions(self, t, q=None):
        """Compute the boundary conditions."""

        rho_out = self.pressure_to_density(self.p_ref)

        BC_state_1 = {
            'left': None,
            'right': rho_out * self.A,
        }
        BC_state_2 = {
            'left': q[0, 0] * (4.0 + 0.5*np.sin(0.2*t)),
            'right': None
        }

        BCs = [BC_state_1, BC_state_2]
        
        return BCs
    
    def velocity(self, q):
        """Compute the wave speed."""
        
        u = q[1]/q[0]

        c = np.abs(u) + self.c/np.sqrt(self.A)

        return c
        
    def flux(self, q):
        """Compute the flux."""


        p = self.density_to_pressure(q[0]/self.A)

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        flux[0] = q[1]
        flux[1] = q[1]*q[1]/q[0] + p * self.A

        return flux
    
    def source(self, t, q):
        """Compute the source."""

        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))

        xl = self.model_params['leak_location']
        if t>0:
            x = self.DG_vars.x.flatten('F')
            width = 50
            point_source = \
                (np.heaviside(x-xl + width/2, 1) - np.heaviside(x-xl-width/2, 1))
            point_source *= 1/width

        rho = q[0]/self.A
        p = self.density_to_pressure(rho)

        s[0] = - self.model_params['leak_size'] * np.sqrt(rho * (p - self.p_amb)) * point_source

        s[1] = -self.friction_factor(q)

        return s