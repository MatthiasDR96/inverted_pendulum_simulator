from math import *
import numpy as np


class BalancingRobot:
    
    def __init__(self):
        
        # System parameters
        self.name = 'Robot'
        self.m = 0.5  # mass of bob (kg)
        self.l = 0.5  # length of pendulum (m)
        self.g = 9.81  # gravitational acceleration (m/s^2)
        self.b_theta = 0.5  # damping pendulum (Ns/m)
        
        # Sample time
        self.control_freq = 100  # frequency at which the used micro controller (Arduino) loops
        self.dt = 1 / self.control_freq  # time period of one time step
        
        # State: [theta, thetadot]
        self.state = np.mat([[0.0], [0.0]])
        
        # Continuous linearized state space representation
        self.A_cont = np.mat([[0.0, 1.0], [self.g / self.l, 0.0]])
        self.B_cont = np.mat([[0.0], [-1.0 / self.l]])

        # Discrete linearized state space representation without stepper (Eulers method)
        self.A_disc = np.identity(len(self.state)) + self.A_cont * self.dt
        self.B_disc = self.dt * self.B_cont
        
        # Measurement matrix
        self.C = np.array([1, 0])
        
        # Simulation time
        self.time_elapsed = 0.0

    # Set an initial state
    def set_state(self, state):
        assert np.shape(state) == (2, 1)
        self.state = state

    # Computes the real non-linearized dynamics and returns the derivative of the state
    def dynamics(self, state, u):
        ds = np.zeros((2, 1))
        ds[0] = state[1]
        ds[1] = (self.g / self.l) * sin(state[0]) - (u / self.l) * cos(state[0]) - self.b_theta * self.state[1]
        return ds

    # Proceed a timestep in simulation by applying discrete dynamics
    def step_disc(self, u):

        # Compute the discrete linearized dynamics
        self.state = self.A_disc * self.state + self.B_disc * u
        self.time_elapsed += self.dt

    # Proceed a timestep in simulation by applying discrete dynamics
    def step_cont(self, u):

        # Compute the continuous non-linearized dynamics
        ds = self.dynamics(self.state, u)

        # Compute new state via Euler iteration
        new_state = np.zeros((2, 1))
        new_state[1] = self.state[1] + ds[1] * self.dt
        new_state[0] = self.state[0] + new_state[1] * self.dt

        # Set new state and evolve time
        self.state = new_state
        self.time_elapsed += self.dt
