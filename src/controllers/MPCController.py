import cvxpy
import numpy as np


class Control:
    
    def __init__(self, model):
        
        # Bind model
        self.model = model
        
        # Desired x_pos
        self.xd = 0.0
        
        # Control parameters
        self.N = 100
        
        # Control limits
        self.umax = np.reshape(np.repeat(10, self.N), (self.N, 1))
        self.xmax = np.reshape(np.repeat(1, 4 * self.N), (4 * self.N, 1))

        # Control parameters
        if self.model.name == 'Pendulum':
            self.Q = np.mat([[100, 0, 0, 0], [0, 10, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            self.R = np.mat(np.identity(1))
            self.P = np.mat([[1000, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        else:
            self.Q = np.mat([[1.0, 0.0], [0.0, 1.0]])
            self.R = np.mat(np.identity(1))
            self.P = np.mat([[1.0, 0.0], [0.0, 1.0]])

        # Get dynamics
        A = np.mat(self.model.A_disc)
        B = np.mat(self.model.B_disc)

        # Alternative to calculating Abar, Bbar, Cbar, and Ahat
        Abar = np.vstack((np.zeros((len(A), self.N*len(A))), np.hstack((np.kron(np.eye(self.N-1), A),
        np.zeros((len(A)*(self.N-1), len(A)))))))
        Bbar = np.kron(np.eye(self.N), B)
        self.Ahat = (np.identity(np.shape(Abar)[0]) - Abar).I * np.kron(np.identity(self.N), A)[:, 0:len(A)]
        self.Cbar = (np.identity(np.shape(Abar)[0]) - Abar).I * Bbar

        # Calculate penalty matrices
        tm1 = np.eye(self.N)
        tm1[self.N - 1, self.N - 1] = 0
        tm2 = np.zeros((self.N, self.N))
        tm2[self.N - 1, self.N - 1] = 1
        self.Qbar = np.kron(tm1, self.Q) + np.kron(tm2, self.P)
        self.Rbar = np.kron(np.eye(self.N), self.R)
    
    def set_desired_position(self, x):
        self.xd = x
    
    def control(self, state):

        # Initial state
        x0 = np.reshape(np.mat(state), (len(state), 1))

        # Set optimization variables
        u = cvxpy.Variable((self.N, 1))
        x = cvxpy.Variable((self.N * len(state), 1))
        
        # Ccompute cost
        cost = 0.5 * cvxpy.quad_form(x, self.Qbar)
        cost += 0.5 * cvxpy.quad_form(u, self.Rbar)
        
        # Create state constraints
        constr = [x == self.Cbar @ u + self.Ahat @ x0]
        
        # Create control constraints
        constr += [np.vstack((np.identity(self.N), -np.identity(self.N))) @ u <= np.vstack((self.umax, self.umax))]

        # Create problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
        
        # Solve problem
        prob.solve(verbose=False)
        
        # Get optimal result
        if prob.status == cvxpy.OPTIMAL:
            ou = np.array(u.value[0, :]).flatten()
        else:
            ou = [0.0]
        
        # Get only first control signal
        u = ou[0]
        
        return u
