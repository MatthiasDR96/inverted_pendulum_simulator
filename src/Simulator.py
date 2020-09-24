import matplotlib
import time
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np


class Simulator:
    
    def __init__(self, model, control=None, frames=1000):
        
        # Model
        self.model = model
        
        # Control
        self.control = control
        self.model.controller = control
        
        # Animation params
        self.frames = frames
        self.delta_t = model.dt
        self.model.dt = self.delta_t
        self.sim_time = self.frames * self.delta_t
        
        # Data lists
        self.time_axis = []  # Time axis
        self.u_list = []  # Total joint torques
        
        # Set up figure and animation
        self.fig = plt.figure()
        
        # Plot axis
        ax1 = self.fig.add_subplot(211, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-1.0, 1.0))
        self.cart_plot = ax1.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))
        self.pendulum_plot, = ax1.plot([], [], 'o-', lw=2)
        self.xd_plot, = ax1.plot([], [], 'r.')
        self.time_plot = ax1.text(0.01, 0.9, '', transform=ax1.transAxes)
        ax1.set_title("Inverted pendulum plot")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.grid()

        # Control axis
        ax4 = self.fig.add_subplot(212, autoscale_on=False, xlim=(0, self.sim_time), ylim=(-20, 20))
        self.u_plot, = ax4.plot([], [], '-', lw=1, )
        ax4.set_title("Control commands")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Control signal (m/s)")
        ax4.grid()
        
        # State lists
        self.state_list = []

        # Choose the interval based on dt and the time to animate one step
        t0 = time.time()
        self.model.step_cont(0)  # Modulate one simulation step
        t1 = time.time()
        self.interval = 1000 * self.delta_t - (t1 - t0)
    
    # Initialize animation
    def init(self):
        self.pendulum_plot.set_data([], [])
        self.xd_plot.set_data([], [])
        self.time_plot.set_text('')
        if self.model.name == "Pendulum":
            self.cart_plot.set_xy((-self.model.cart_width / 2, -self.model.cart_height / 2))
            self.cart_plot.set_width(self.model.cart_width)
            self.cart_plot.set_height(self.model.cart_height)
        self.u_plot.set_data([], [])
        return self.pendulum_plot, self.xd_plot, self.u_plot, self.time_plot, self.cart_plot
    
    # Animation step
    def animate(self, i):
    
        # Compute control signal
        if self.control:
            u = self.control.control(self.model.state)
        else:
            u = 0
    
        # Apply dynamics
        self.model.step_cont(u)
        
        # Capture state data
        self.state_list.append(self.model.state)
        
        # Compute pendulum mass position
        if self.model.name == "Pendulum":
            x_pos_pend = self.model.l * np.sin(self.model.state[2]) + self.model.state[0]
            y_pos_pend = self.model.l * np.cos(self.model.state[2])
            thisx = [self.model.state[0], x_pos_pend]
            thisy = [0, y_pos_pend]
        else:
            x_pos_pend = self.model.l * np.sin(self.model.state[0])
            y_pos_pend = self.model.l * np.cos(self.model.state[0])
            thisx = [0, x_pos_pend]
            thisy = [0, y_pos_pend]

        # Plot pendulum
        self.pendulum_plot.set_data(thisx, thisy)

        # Plot time
        self.time_plot.set_text('time = %.1fs' % (i * self.delta_t))

        # Plot cart
        if self.model.name == 'Pendulum':
            self.cart_plot.set_x(self.model.state[0] - self.model.cart_width / 2)

        # Plot desired position
        if self.control:
            self.xd_plot.set_data(self.control.xd, 0)
    
        # Update plot data
        self.time_axis.append(i * self.delta_t)
        self.u_list.append(u)
    
        # Plot data lists
        self.u_plot.set_data(self.time_axis, self.u_list)
    
        return self.pendulum_plot, self.xd_plot, self.u_plot, self.time_plot, self.cart_plot

    def simulate(self):
    
        # Animate
        _ = animation.FuncAnimation(self.fig, self.animate, repeat=False, interval=self.interval, frames=self.frames,
                                        blit=True, init_func=self.init)

        plt.tight_layout()
        plt.show()
        
        return self.time_axis, np.array(self.state_list)