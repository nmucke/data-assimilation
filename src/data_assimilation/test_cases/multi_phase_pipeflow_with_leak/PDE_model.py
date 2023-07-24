import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.polynomials.jacobi_polynomials import JacobiP
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import fsolve, least_squares

from matplotlib.animation import FuncAnimation

from scipy.linalg import eig

class PipeflowEquations(BaseModel):
    """Multiphase equation model class."""

    def __init__(
        self, 
        model_parameters: dict,
        **kwargs
        ):
        super().__init__(**kwargs)


        for k, v in model_parameters.items():
            setattr(self, k, v)
        

        self.D_orifice = 0.01
        self.A_orifice = np.pi*(self.D_orifice/2)**2
        self.Cv = self.A/np.sqrt(self.rho_g_norm/2 * ((self.A/(self.A_orifice*self.Cd))**2-1))

        self.conservative_or_primitive = 'primitive'

        self.added_boundary_noise = 0.0
        self._t = 0.0

        self.xElementL = np.int32(self.leak_location / self.basic_args['xmax'] * self.DG_vars.K)

        self.lagrange = []
        l = np.zeros(self.DG_vars.N + 1)
        rl = 2 * (self.leak_location - self.DG_vars.VX[self.xElementL]) / self.DG_vars.deltax - 1
        for i in range(0, self.DG_vars.N + 1):
            l[i] = JacobiP(np.array([rl]), 0, 0, i)
        self.lagrange = np.linalg.solve(np.transpose(self.DG_vars.V), l)    

    def update_parameters(self, pars):

        leak_location = pars[0]
        Cd = pars[1]

        self.Cv = self.A/np.sqrt(self.rho_g_norm/2 * ((self.A/(self.A_orifice*Cd))**2-1))
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

        return self.p_norm * rho * self.T / self.rho_g_norm / self.T_norm


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return self.rho_g_norm * self.T_norm / self.p_norm * p / self.T
    
    def conservative_to_primitive(self, q):
        """Compute the primitive variables from the conservative variables."""

        A_l = q[1]/self.rho_l
        A_g = self.A - A_l

        rho_g = q[0]/A_g

        p = self.density_to_pressure(rho_g)

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

        u_m = q[2]/rho_m/self.A

        return np.array([A_l, p/self.p_outlet, u_m])
    
    def primitive_to_conservative(self, q):
        """Compute the conservative variables from the primitive variables."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

        rho_g_A_g = rho_g * A_g
        rho_l_A_l = self.rho_l * A_l
        rho_m_u_m_A = rho_m * u_m * self.A

        return np.array([rho_g_A_g, rho_l_A_l, rho_m_u_m_A])
    

    def friction_factor(self, q):
        """Compute the friction factor."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        A_g = self.A - A_l

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_g = self.pressure_to_density(p)

        #rho_m = rho_g * A_g + self.rho_l * A_l
        rho_m = rho_g * alpha_g + self.rho_l * alpha_l
        mu_m = self.mu_g * alpha_g + self.mu_l * alpha_l

        Re = rho_m * np.abs(u_m) * self.d / mu_m

        a = (-2.457 * np.log((7/Re)**0.9 + 0.27*self.e/self.d))**16
        b = (37530/Re)**16
        
        f_w = 8 * ((8/Re)**12 + (a + b)**(-1.5))**(1/12)

        T_w = f_w * rho_m * u_m * np.abs(u_m) / (2 * self.d) * self.A

        return T_w

    def initial_condition(self, x):
        """Compute the initial condition.
        
        initial condition:
        rho_g * A_g * u_m = a
        rho_l * A_l * u_m = b
        rho_m * u_m * A = a + b
        
        Conservative:
        q[0] = rho_g * A_g
        q[1] = rho_l * A_l
        q[2] = rho_m * u_m * A
        
        Primitive:
        q[0] = A_l
        q[1] = p
        q[2] = u_m

        rho_g = self.p_outlet * (rho_g_norm * self.T_norm) / (self.p_norm * self.T_norm)
        alpha_l  = b/self.rho_l / (a/rho_g + b/self.rho_l)
        u_m = b /(self.rho_l * self.A * self.alpha_l)
        """

        a = 0.2
        b = 20

        rho_g = self.pressure_to_density(self.p_outlet)#self.p_outlet * (self.rho_g_norm * self.T_norm) / (self.p_norm * self.T)
        alpha_l  = b/self.rho_l / (a/rho_g + b/self.rho_l)
        u_m = b /(self.rho_l * self.A * alpha_l)

        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0, :] = alpha_l*self.A
        init[1, :] = 1.#self.p_outlet/self.p_outlet
        init[2, :] = u_m

        return init
    
    def start_solver(self):

        pass

        '''

        brownian = Brownian()

        window_size = 600
        self.inflow_boundary_noise = brownian.gen_normal(n_step=np.int64(self.t_final/self.step_size + window_size + 1))
        self.inflow_boundary_noise = moving_average(self.inflow_boundary_noise, n=window_size)
        self.inflow_boundary_noise = np.abs(self.inflow_boundary_noise)
        '''


    def BC_eqs(self, q, gas_mass_inflow, liquid_mass_inflow):

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        gas_mass_flow = rho_g * A_g * u_m
        liquid_mass_flow = self.rho_l * A_l * u_m

        return np.array([gas_mass_flow - gas_mass_inflow, liquid_mass_flow - liquid_mass_inflow, 0.0])


    def boundary_conditions(self, t=0, q=None):
        """Compute the boundary conditions."""


        #inflow_noise = self.inflow_boundary_noise[len(self.t_vec)]*15

        
        t_start = 10000000.
        t_end = 200000000.

        # a increases linearly from 0.2 to 0.4 over 10 seconds
        if t < t_end and t > t_start:
            t_ = t - t_start
            gas_mass_inflow = 0.2 + 0.02 * t_ / 10
        elif t > t_end:
            gas_mass_inflow = 0.4
        else:
            gas_mass_inflow = 0.2 #+ inflow_noise#+ self.added_boundary_noise# * np.sin(2*np.pi*t/50)

        gas_mass_inflow = 0.2# + inflow_noise
        liquid_mass_inflow = 20.0# + inflow_noise

        if len(q.shape) == 1:
            rho_g = self.pressure_to_density(q[1]*self.p_outlet)
        else:
            rho_g = self.pressure_to_density(q[1]*self.p_outlet)[0]
        _alpha_l  = liquid_mass_inflow/self.rho_l / (gas_mass_inflow/rho_g + liquid_mass_inflow/self.rho_l)
        _u_m = liquid_mass_inflow /(self.rho_l * self.A * _alpha_l)

        func = lambda q: self.BC_eqs(q, gas_mass_inflow, liquid_mass_inflow)
        q_guess = np.array([_alpha_l*self.A, self.density_to_pressure(rho_g)/1e6, _u_m])
        q_sol = fsolve(func, q_guess)

        A_l = q_sol[0]
        p = q_sol[1]*self.p_outlet
        u_m = q_sol[2]

        #print('A_l: ', A_l-_alpha_l*self.A, 'u_m: ', u_m-_u_m)

        if self.steady_state_solve or self.BC_args['treatment'] == 'naive':
            BC_state_1 = {
                'left': A_l,
                'right': None,
            }
            BC_state_2 = {
                'left': None,
                'right': 1. #+ outflow_noise#self.p_outlet,
            }
            BC_state_3 = {
                'left': u_m,#,
                'right': None
            }
        else:
            BC_state_1 = {
                'left': (q[0] - A_l) / self.step_size,
                'right': None,
            }
            BC_state_2 = {
                'left': None,
                'right': 0.,#(q[1] - 1.0) / self.step_size,
            }
            BC_state_3 = {
                'left': (q[2] - u_m) / self.step_size,
                'right': None
            }


        BC_flux_1 = {
            'left': gas_mass_inflow,
            'right': None,
        }
        BC_flux_2 = {
            'left': liquid_mass_inflow,
            'right': None,
        }
        BC_flux_3 = {
            'left': None,
            'right': None
        }

        BCs_state = [BC_state_1, BC_state_2, BC_state_3]
        BCs_flux = [BC_flux_1, BC_flux_2, BC_flux_3]

        BCs = {
            'state': BCs_state,
            'flux': BCs_flux,
        }

        return BCs


    
    def velocity(self, q):
        """Compute the wave speed."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        return np.maximum(np.abs(u_m + self.c), np.abs(u_m - self.c))

        
    def flux(self, q):
        """Compute the flux."""

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        A_g = self.A - A_l

        rho_g = self.pressure_to_density(p)

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

        flux[0] = rho_g * A_g * u_m
        flux[1] = self.rho_l * A_l * u_m
        
        flux[2] = rho_m * u_m**2 * self.A + p * self.A
        
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

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        alpha_l = A_l/self.A
        alpha_g = 1 - alpha_l

        rho_g = self.pressure_to_density(p)
        rho_m = rho_g * alpha_g + self.rho_l * alpha_l
        
        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
        if t>0.:

            '''
            x = self.DG_vars.x.flatten('F')
            width = 50
            point_source = \
                (np.heaviside(x-self.leak_location + width/2, 1) - np.heaviside(x-self.leak_location-width/2, 1))
            point_source *= 1/width

            leak_mass = self.Cv * np.sqrt(rho_m * (p - self.p_amb)) * point_source
            s[0] = -alpha_g * leak_mass
            s[1] = -alpha_l * leak_mass
            '''
            leak = self.leakage(pressure=p, rho_m=rho_m).flatten('F')
            s[0] = -alpha_g * leak
            s[1] = -alpha_l * leak

        s[-1] = -self.friction_factor(q)

        return s
