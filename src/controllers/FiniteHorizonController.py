import numpy as np


class Control:
    
    def __init__(self, model):
        
        # Bind model
        self.model = model
        
        # Desired x_pos
        self.xd = 0.0
        
        # Control parameters
        self.N = 100  # Prediction and control horizon

        # Control parameters
        self.Q = np.mat([[100, 0, 0, 0], [0, 10, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.R = np.mat(np.identity(1))
        self.P = np.mat([[1000, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        # Get dynamics
        A = np.mat(self.model.A_disc)
        B = np.mat(self.model.B_disc)

        # Calculate matrix relating initial state to all successive states
        Ahat = np.eye(4)
        for i in range(self.N):
            An = np.linalg.matrix_power(A, i + 1)
            Ahat = np.r_[Ahat, An]
        Ahat = Ahat[4:, :]

        # Calculate matrix relating control signals to successive states
        Cbar = np.zeros((4, self.N))
        for i in range(self.N):
            tmp = None
            for ii in range(i + 1):
                tm = np.linalg.matrix_power(A, ii) * B
                if tmp is None:
                    tmp = tm
                else:
                    tmp = np.c_[tm, tmp]
            for _ in np.arange(i, self.N - 1):
                tm = np.zeros(B.shape)
                if tmp is None:
                    tmp = tm
                else:
                    tmp = np.c_[tmp, tm]
            Cbar = np.r_[Cbar, tmp]
        Cbar = Cbar[4:, :]

        # Calculate penalty matrices
        tm1 = np.eye(self.N)
        tm1[self.N - 1, self.N - 1] = 0
        tm2 = np.zeros((self.N, self.N))
        tm2[self.N - 1, self.N - 1] = 1
        Qbar = np.kron(tm1, self.Q) + np.kron(tm2, self.P)
        Rbar = np.kron(np.eye(self.N), self.R)

        # Calculate objective derivative solution
        self.H = Cbar.T * Qbar * Cbar + Rbar
        self.F_trans = Ahat.T * Qbar * Cbar
    
    def set_desired_position(self, x):
        self.xd = x
    
    def control(self, state):

        # Initial state
        x0 = np.reshape(np.mat(state), (4, 1))
        
        # Solve for optimal control
        uopt = -self.H.I * self.F_trans.T * x0

        # Take only first control signal
        u = uopt[0]

        return u
