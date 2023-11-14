import time
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.integrate import odeint

class Burgers():
    def __init__(
        self,
        N: int = 256,
    ) -> None:
        self.N = N

        self.nu = 1/150

        self.x = np.linspace(0, 2, N)

        # Second derivative finite difference matrix
        self.second_deriv_matrix = -2*np.eye(self.N-2)
        self.second_deriv_matrix += np.diag(np.ones(self.N-2-1), k=1)
        self.second_deriv_matrix += np.diag(np.ones(self.N-2-1), k=-1)
        self.second_deriv_matrix = np.concatenate(
            (
                np.zeros((self.N-2, 1)),
                self.second_deriv_matrix,
                np.zeros((self.N-2, 1)),
            ),  
            axis=1     
        ) 
        self.second_deriv_matrix[0, 0] = 1
        self.second_deriv_matrix[-1, -1] = 1
        self.second_deriv_matrix /= (self.x[1] - self.x[0])**2

        self.first_derivative_matrix = np.diag(np.ones(self.N-2-1), k=1)
        self.first_derivative_matrix -= np.diag(np.ones(self.N-2-1), k=-1)
        self.first_derivative_matrix = np.concatenate(
            (
                np.zeros((self.N-2, 1)),
                self.first_derivative_matrix,
                np.zeros((self.N-2, 1)),
            ),          
            axis=1
        )
        self.first_derivative_matrix[0, 0] = -1
        self.first_derivative_matrix[-1, -1] = 1
        self.first_derivative_matrix /= 4*(self.x[1] - self.x[0])

    def update_parameters(self, pars):
        pass

    def initialize_state(self, pars):
        state = pars*np.sin(2*np.pi*self.x/2)

        return state, pars

    
    def rhs(self, q, t):
        """ Space discretization of Burgers equations """

        dudt = - self.first_derivative_matrix @ (q*q) + self.nu*self.second_deriv_matrix @ q

        # Add zero boundary conditions
        dudt = np.concatenate(
            (
                np.zeros(1),
                dudt,
                np.zeros(1),
            )
        )
        return dudt
    
    def solve(
        self, 
        t, 
        q_init, 
        t_final, 
        print_progress=False
    ):

        sol = odeint(
            self.rhs, 
            q_init[0], 
            [t, t_final],
        )
        
        sol = sol[-1, :]

        return sol, 0
    