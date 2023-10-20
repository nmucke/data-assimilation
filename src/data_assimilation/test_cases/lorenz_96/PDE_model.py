import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.polynomials.jacobi_polynomials import JacobiP
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

class PipeflowEquations(BaseModel):
    """Single phase equation model class."""

    def __init__(
        self, 
        model_parameters: dict,
        **kwargs,
        ):
        super().__init__(**kwargs)

        for k, v in model_parameters.items():
            setattr(self, k, v)

        self.xElementL = np.int32(self.leak_location / self.basic_args['xmax'] * self.DG_vars.K)

        self.lagrange = []
        l = np.zeros(self.DG_vars.N + 1)
        rl = 2 * (self.leak_location - self.DG_vars.VX[self.xElementL]) / self.DG_vars.deltax - 1
        for i in range(0, self.DG_vars.N + 1):
            l[i] = JacobiP(np.array([rl]), 0, 0, i)
        self.lagrange = np.linalg.solve(np.transpose(self.DG_vars.V), l)    

        self.D_orifice = 0.03
        self.A_orifice = np.pi*(self.D_orifice/2)**2

    def update_parameters(self, pars):

        leak_location = pars[0]
        Cd = pars[1]

        self.Cv = self.A/np.sqrt(self.rho_ref/2 * ((self.A/(self.A_orifice*Cd))**2-1))
        self.leak_location = leak_location
        
        self.xElementL = np.int32(self.leak_location / self.basic_args['xmax'] * self.DG_vars.K)

        self.lagrange = []
        l = np.zeros(self.DG_vars.N + 1)
        rl = 2 * (self.leak_location - self.DG_vars.VX[self.xElementL]) / self.DG_vars.deltax - 1
        for i in range(0, self.DG_vars.N + 1):
            l[i] = JacobiP(np.array([rl]), 0, 0, i)
        self.lagrange = np.linalg.solve(np.transpose(self.DG_vars.V), l)    


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

    def initial_condition(self, x):
        """Compute the initial condition."""

        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0] = self.pressure_to_density(self.p_ref) * self.A
        init[1] = init[0] * 4.5

        return init
    
    def boundary_conditions(self, t, q=None):
        """Compute the boundary conditions."""
        
        BC_state_1 = {
            'left': None,
            'right': self.pressure_to_density(self.p_ref) * self.A#(p-self.p_ref)/self.step_size
        }
        BC_state_2 = {
            'left': q[0, 0]*4.,#(u-4.5)/self.step_size,#4.0 + 0.5,#*np.sin(0.2*t)),
            'right': None
        }

        BCs = [BC_state_1, BC_state_2]

        BCs = {
            'state': BCs,
            'flux': None,
        }
        
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
    
    def leakage(self, pressure=0, rho_m=0):
        """Compute leakage"""

        f_l = np.zeros((self.DG_vars.x.shape))

        pressureL = self.evaluate_solution(np.array([self.leak_location]), pressure)[0]
        rhoL = self.evaluate_solution(np.array([self.leak_location]), rho_m)[0]

        discharge_sqrt_coef = (pressureL - self.p_amb) * rhoL
        f_l[:, self.xElementL] = self.Cv * np.sqrt(discharge_sqrt_coef) * self.lagrange
        f_l[:, self.xElementL] = self.DG_vars.invMk @ f_l[:, self.xElementL]

        return f_l
    
    def source(self, t, q):
        """Compute the source."""

        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))
        rho = q[0]/self.A
        p = self.density_to_pressure(rho)

        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
        if t>0:
            leak = self.leakage(pressure=p, rho_m=rho).flatten('F')

            s[0] = -leak
        #s[0] *= 0.
        s[1] = -self.friction_factor(q)


        return s