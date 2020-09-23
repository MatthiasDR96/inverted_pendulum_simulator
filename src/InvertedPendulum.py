from math import *

import numpy as np


class InvertedPendulum:
    
    def __init__(self):
        
        # System parameters
        self.m = 0.5  # mass of bob (kg)
        self.M = 1.0  # mass of cart (kg)
        self.l = 0.5  # length of pendulum (m)
        self.g = 9.81  # gravitational acceleration (m/s^2)
        self.b_theta = 0.5  # damping pendulum (Ns/m)
        self.b_x = 1.0  # damping cart (Ns/m)
        self.cart_width = 0.1  # width of the cart (m)
        self.cart_height = 0.2  # height of the cart (m)
        
        # Sample time
        self.control_freq = 100  # frequency at which the used micro controller (Arduino) loops
        self.dt = 1 / self.control_freq  # time period of one time step
        
        # State: [x, xdot, theta, thetadot]
        self.state = np.mat([[0.0], [0.0], [0.0], [0.0]])
        
        # Continuous linearized state space representation without stepper
        self.A_cont = np.mat([[0, 1, 0, 0], [0, -self.b_theta, -self.g * self.m / self.M, 0], [0, 0, 0, 1.],
                              [0, self.b_theta / self.l, (self.m + self.M) * self.g / (self.M * self.l), -self.b_x]])
        self.B_cont = np.mat([[0], [1.0 / self.M], [0], [-1 / (self.M * self.l)]])

        # Discrete linearized state space representation without stepper (Eulers method)
        self.A_disc = np.identity(len(self.state)) + self.A_cont * self.dt
        self.B_disc = self.dt * self.B_cont
        
        # Continuous linearized state space representation with stepper
        self.A_step_cont = np.mat([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1.], [0, 0, self.g / self.l, 0]])
        self.B_step_cont = np.mat([[0], [1.0], [0], [-1 / self.l]])
        
        # Discrete linearized state space representation with stepper (Eulers method)
        self.A_step_disc = np.identity(len(self.state)) + self.A_step_cont * self.dt
        self.B_step_disc = self.dt * self.B_step_cont
        
        # Measurement matrix
        self.C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        
        # Simulation time
        self.time_elapsed = 0.0

        # Reinforcement Learning: Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * pi / 360
        self.x_threshold = 2.4
        self.steps_beyond_done = 0

    # Set an initial state
    def set_state(self, state):
        assert np.shape(state) == (4, 1)
        self.state = state

    # Computes the real non-linearized dynamics and returns the derivative of the state
    def dynamics(self, state, u):
        ds = np.zeros((4, 1))
        ds[0] = state[1]
        ds[1] = (u + self.m * self.l * state[3] ** 2 * sin(state[2]) + self.m * self.g * cos(state[2]) *
                 sin(state[2])) / (self.M + self.m - self.m * cos(state[2]) ** 2) - self.b_x * self.state[1]
        ds[2] = state[3]
        ds[3] = (-cos(state[2]) / self.l) * ds[1] + (self.g / self.l) * sin(state[2]) - self.b_theta * self.state[3]
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
        new_state = np.zeros((4, 1))
        new_state[1] = self.state[1] + ds[1] * self.dt
        new_state[0] = self.state[0] + new_state[1] * self.dt
        new_state[3] = self.state[3] + ds[3] * self.dt
        new_state[2] = self.state[2] + new_state[3] * self.dt

        # Set new state and evolve time
        self.state = new_state
        self.time_elapsed += self.dt

    # Proceed a timestep in simulation and calculating the reward of the action
    def step_rl(self, u):

        # Apply dynamics
        self.step_cont(u)

        # End criterium
        done = bool(
            self.state[0] < - self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < - self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        # Rewards
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}
