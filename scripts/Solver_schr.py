# Script for solving Schrödinger 1 dimensional equation #
# and animating the solution both in the free case      #
# and the case with a potential barrier                 #
# Import this module in a notebook to the the animation #
# Contact : lorenzo.spera@studenti.unipg.it             #

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scipy.constants 
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
import numpy as np


# ------------------ global style for plots ------------------
mpl.rcParams.update({
    "text.usetex": False,                                                       # LaTeX labels (comment this if you don't have LaTeX)
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "legend.fontsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


class schrodinger_solver:
    def __init__(self, N, nsteps):

        # contants and initial values 
        self.m = scipy.constants.m_e                                            # mass of the electron
        self.hbar = scipy.constants.hbar                                        # planck's constant 
        self.sigma = 1e-10                                                      # standard deviation of the wave packet
        self.k  = 5*1e10                                                        # wave vector of the electron
        self.L = 1e-8                                                           # lenght of the box 
        self.x0 = self.L/2                                                      # midpoint
        self.Npoints = N                                                        # number of spatial points 
        self.a = self.L/self.Npoints                                            # spatial spacing between points 
        self.h = 1e-18                                                          # time step
        self.V0 = 3*scipy.constants.e                                           # height of the potential in ev       
        self.steps = nsteps

        # elements of tridiagonal matrix to solve the equation using the implicit Cranck - Nicolson scheme 

        self.a1 = complex(real = 1, imag = self.h*self.hbar/(2*self.m*self.a**2))
        self.a2 = complex(real = 0, imag = -self.h*self.hbar/(4*self.m*self.a**2))
        self.b1 = complex(real = 1, imag = -self.h*self.hbar/(2*self.m*self.a**2))
        self.b2 = complex(real = 0, imag = self.h*self.hbar/(4*self.m*self.a**2))

        self.x_range = np.arange (0, self.L, self.a)                             # spatial range 

        # build tridiagonal matrices for the free case

        self.A, self.B = self.build_AB_matrices_free_case()

        # build the tridiagonal matrices for the potential barrier 

        self.AV, self.BV = self.build_AB_matrices_potential_barrier()

    
    def psi_initial(self, x):
        """ 
            Returns the intial profile for the wave function at t = 0.
            Input : x (np.float)
            Output : initial_psi (np.complex) 
        """
    
        term1  = np.exp((-(x-self.x0)**2)/(2*self.sigma**2))
        term2 = complex(real = np.cos(self.k*x), imag = np.sin(self.k*x))
        return term1*term2
    
    def initialise_psi(self):
        """ Initializes the wave functiona at t = 0 according to the form of psi_initial."""

        initial_psi = np.array([self.psi_initial(x) for x in self.x_range], dtype = complex)
        return initial_psi
    
    def build_AB_matrices_free_case(self):
        """
        Function that creates the two tridiagonal matrices A and B for the free case.
        """
        A = np.zeros(shape = [self.Npoints, self.Npoints], dtype = complex)
        B = np.zeros(shape = [self.Npoints, self.Npoints], dtype = complex)
        for i in range(self.Npoints-1):
            A[i,i] = self.a1
            A[i, i+1] = self.a2
            A[i+1, i] = self.a2
            B[i,i] = self.b1
            B[i, i+1] = self.b2
            B[i+1, i] = self.b2
        
        return A, B

    def single_step(self, psi_to_evolve):
        """
        Function that implements the implicit Crank-Nicolson method for a single time step.
        psi_to_evolve is the wave function to evolve to the next time step.

        Output : wave function at the next time step
        """
        psi0 = psi_to_evolve
        v = self.B @ psi0
        psi_next = np.linalg.solve(self.A, v)

        return psi_next
    
    def evolve_wave_function(self):
        """
        Function that evolves the initial wave function up to time t = nsteps*h

        Output : list containing wave function at each time step
        """
        psi0 = self.initialise_psi()
        psi_list = [psi0]

        for i in range(self.steps):
            next_psi = self.single_step(psi0)                                           # compute evolution
            psi_list.append(next_psi)                                                   # save the wave function
            psi0 = next_psi                                                             # swap the wave function to have the right one at next steps
        
        return psi_list
    
    def plot_wave_function(self, psi):
        """
        Function that plots the real part, imaginary part and squared module of the wave function
        """

        fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = [12,6], sharex = True)
        fig.suptitle(r"Plot for $\psi(x,t)$")
        ax[0].plot(self.x_range, np.real(psi), c = " red ", label = r"$\Re{\psi(x,t)}$" )
        ax[1].plot(self.x_range, np.imag(psi), c = " blue ", label = r"$\Im{\psi(x,t)}$" )
        ax[2].plot(self.x_range, (np.abs(psi))**2, c = " red", label = r"$|{\psi(x,t)|^2$")
        for i in range(3):
            ax[i].legend(loc = "best")
        fig.tight_layout()
        plt.show()
    

    def animate_wave_function(self):
        """
        Function that implements the animation of the real part of the wave function over time.
        """

        psi_history = self.evolve_wave_function()                                             # list of wave functions over time 
        fig, ax = plt.subplots(figsize = [12,4])
        fig.suptitle(r"Evolution of $\Re{\psi(x,t)}$ over time")
        line, = ax.plot([], [], lw = 2)

        ax.set_xlim(self.x_range[0], self.x_range[-1])

        max_real = np.max(np.abs(np.real(psi_history)))
        ax.set_ylim(-1.1 * max_real, 1.1 * max_real)

        ax.set_xlabel(r"$x\ \mathrm{(m)}$")
        ax.set_ylabel(r"$\Re[\psi(x,t)]$")
        fig.tight_layout()

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            y = np.real(psi_history[frame])
            line.set_data(self.x_range, y)
            return line,

        ani = FuncAnimation(
            fig,
            update,
            frames = len(psi_history),                          # number of time snapshots
            init_func = init,
            blit=True,
            interval = 5                                        # ms between frames (~rate analog)
        )

        plt.close(fig)                                          # avoid static duplicate


        # return an HTML animation 
        return HTML(ani.to_jshtml())
    
    def potential_barrier(self):
        """
        Function that defines the potential barrier.
        self.V0   : height of the potential in eV
        """
        V = np.zeros((len(self.x_range)))
        left_edge = 0.4*self.L
        right_edge = 0.6*self.L

        mask = (left_edge <= self.x_range) & (right_edge >= self.x_range)                           # select the region whhere there is the barrier 
        V[mask] = self.V0

        return V

    def build_AB_matrices_potential_barrier(self):
        """
        Function that creates the two tridiagonal matrices A and B for the potential barrier.
        
        """
        alpha = self.h * self.hbar / (2 * self.m * self.a**2)      # from the kinetic term (same as before)
        gamma = self.h * self.potential_barrier()/ (2 * self.hbar)                  # array: from the potential term

        A = np.zeros((self.Npoints, self.Npoints), dtype=complex)
        B = np.zeros((self.Npoints, self.Npoints), dtype=complex)

        for j in range(self.Npoints):
            # diagonal terms: 1 + i (alpha + gamma_j)   and   1 - i (alpha + gamma_j)
            A[j, j] = complex(1.0,  alpha + gamma[j])
            B[j, j] = complex(1.0, -alpha - gamma[j])

            # off-diagonal kinetic terms (same as free case)
            if j < self.Npoints - 1:
                off_A = complex(0.0, -alpha / 2.0)
                off_B = complex(0.0,  alpha / 2.0)

                A[j,     j+1] = off_A
                A[j+1,   j  ] = off_A

                B[j,     j+1] = off_B
                B[j+1,   j  ] = off_B
    
        
        return A, B
    
    def plot_potential(self):
        V = self.potential_barrier()
        plt.figure(figsize = [10,6])
        plt.title(r"Plot of the potential barrier")
        plt.plot(self.x_range, V, label = r"$V(x)$")
        plt.xlabel(r"$x(m)$")
        plt.ylabel(r"$V(x) [eV]$")
        plt.tight_layout()
        plt.show()

    def single_step_potential(self, psi_to_evolve):
        """
        Function that implements the implicit Crank-Nicolson method for a single time step in the case of a potential barrier.
        psi_to_evolve is the wave function to evolve to the next time step.

        Output : wave function at the next time step
        """
        psi0 = psi_to_evolve
        v = self.BV @ psi0
        psi_next = np.linalg.solve(self.AV, v)

        return psi_next
    
    def evolve_wave_function_potential(self):
        """
        Function that evolves the initial wave function up to time t = nsteps*h

        Output : list containing wave function at each time step
        """
        psi0 = self.initialise_psi()
        psi_list = [psi0]

        for i in range(self.steps):
            next_psi = self.single_step_potential(psi0)                                           # compute evolution
            psi_list.append(next_psi)                                                             # save the wave function
            psi0 = next_psi                                                                       # swap the wave function to have the right one at next steps
        
        return psi_list
    

    def animate_wave_function_potential(self):
        """
        Function that implements the animation of the real part of the wave function over time.
        """

        psi_history = self.evolve_wave_function_potential()                                             # list of wave functions over time 
        fig, ax = plt.subplots(figsize = [12,4])
        fig.suptitle(r"Evolution of $|{\psi(x,t)}|^2$ over time")
        line, = ax.plot([], [], lw = 2)

        ax.set_xlim(self.x_range[0], self.x_range[-1])
        V = self.potential_barrier()
        ax.plot(self.x_range, V/scipy.constants.e, label = r"$V(x)$")

        max_real = np.max(np.abs((psi_history))**2)
        ax.set_ylim(-1.1 * max_real, 1.1 * max_real)

        ax.set_xlabel(r"$x\ \mathrm{(m)}$")
        ax.set_ylabel(r"$|[\psi(x,t)]|^2$")
        fig.tight_layout()

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            y = np.abs(psi_history[frame])**2
            line.set_data(self.x_range, y)
            return line,

        ani = FuncAnimation(
            fig,
            update,
            frames = len(psi_history),                          # number of time snapshots
            init_func = init,
            blit=True,
            interval = 5                                        # ms between frames (~rate analog)
        )

        plt.close(fig)                                          # avoid static duplicate


        # return an HTML animation 
        return HTML(ani.to_jshtml())
    

