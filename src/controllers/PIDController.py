class Control:
    
    def __init__(self, model):

        # Bind model
        self.model = model
        
        # Desired x_pos
        self.xd = 0.0
        
        # Control parameters
        self.Kp_th = 50
        self.Kd_th = 15
        self.Kp_x = 5.0
        self.Kd_x = 4.8
    
    def set_desired_position(self, x):
        self.xd = x
    
    def control(self, state):

        if self.model.name == 'Pendulum':
            # Theta control
            error_theta = (state[2] - 0)
            u_theta_p = self.Kp_th * error_theta
            u_theta_d = self.Kd_th * state[3]

            # X control
            error_x = (state[0] - self.xd)
            u_x_p = self.Kp_x * error_x
            u_x_d = self.Kd_x * state[1]

            # Control signal
            u = u_theta_p + u_theta_d + u_x_p + u_x_d

        else:

            # Theta control
            error_theta = (0 - state[0])
            u_theta_p = self.Kp_th * error_theta
            u_theta_d = self.Kd_th * state[1]

            # Control signal
            u = u_theta_p + u_theta_d
        
        return u
