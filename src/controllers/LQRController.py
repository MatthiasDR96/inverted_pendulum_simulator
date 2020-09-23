import numpy as np
import scipy.linalg


class Control:
    
    def __init__(self, model):
        
        # Bind model
        self.model = model
        
        # Desired x_pos
        self.xd = 0.0
        
        # Control parameters
        if self.model.name == 'Pendulum':
            self.Q = np.array([[100, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10, 0], [0, 0, 0, 100]])
        else:
            self.Q = np.array([[1, 0], [0, 1]])
        self.R = np.identity(1)
        
        # Compute optimal K
        self.K = self.lqr(self.model.A_cont, self.model.B_cont, self.Q, self.R)

        ### Only for pendulum on cart, to track desired x position ###

        # Closed loop system matrix
        Acl = self.model.A_cont - self.model.B_cont * self.K

        # DC gain
        Kdc = self.model.C * np.linalg.inv(Acl) * self.model.B_cont

        # Reference gain
        self.Kr = - 1 / Kdc[0]
    
    def set_desired_position(self, x):
        self.xd = x
    
    def lqr(self, A, B, Q, R):
        
        # Solve the ricatti equation
        X = np.mat(scipy.linalg.solve_continuous_are(A, B, Q, R))
        
        # Compute the LQR gain
        K = np.mat(scipy.linalg.inv(R) * (B.T * X))
        
        return K
    
    def control(self, state):

        # Control signal
        if self.model.name == "Pendulum":
            u = - self.K * state + self.xd * self.Kr
        else:
            u = - self.K * state
        return u
