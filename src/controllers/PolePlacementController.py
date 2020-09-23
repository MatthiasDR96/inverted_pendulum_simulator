import control
import numpy as np


class Control:
    
    def __init__(self, model):
        # Bind model
        self.model = model
        
        # Desired x_pos
        self.xd = 0.0
        
        # Compute desired K
        desired_eigenvalues = [-2, -8, -9, -10]
        self.K = control.place(self.model.A_cont, self.model.B_cont, desired_eigenvalues)

        # Closed loop system matrix
        Acl = self.model.A_cont - self.model.B_cont * self.K

        # DC gain
        Kdc = self.model.C * np.linalg.inv(Acl) * self.model.B_cont

        # Reference gain
        self.Kr = - 1 / Kdc[0]
    
    def set_desired_position(self, x):
        self.xd = x
    
    def control(self, state):
        
        # Control signal
        u = - self.K * state + self.xd * self.Kr
        
        return u
