from src.InvertedPendulum import *
import cvxpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation


# Initialize animation
def init():
    line_plot.set_data([], [])
    line_time.set_text('')
    patch.set_xy((-model.cart_width / 2, -model.cart_height / 2))
    patch.set_width(model.cart_width)
    patch.set_height(model.cart_height)
    return line_plot, line_time, patch


# Animation step
def animate(i):
    
    u = u_t[i]
    
    # Apply dynamics
    model.step_cont(u)
    
    # Plot pendulum
    x_pos_pend = model.l * np.sin(model.state[2]) + model.state[0]
    y_pos_pend = model.l * np.cos(model.state[2])
    thisx = [model.state[0], x_pos_pend]
    thisy = [0, y_pos_pend]
    line_plot.set_data(thisx, thisy)
    line_time.set_text('time = %.1fs' % (i * T/N))
    patch.set_x(model.state[0] - model.cart_width / 2)
    
    return line_plot, line_time, patch


def simulate():
    
    # Animate
    _ = animation.FuncAnimation(fig, animate, repeat=False, interval=50, frames=N*4-4,
                                blit=True, init_func=init)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # Set up figure and animation
    fig = plt.figure()
    
    # Plot axis
    ax1 = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-1.0, 1.0))
    patch = ax1.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))
    line_plot, = ax1.plot([], [], 'o-', lw=2)
    line_time = ax1.text(0.01, 0.9, '', transform=ax1.transAxes)
    ax1.set_title("Inverted pendulum plot")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid()
    
    # Trajectory parameters
    T = 2
    N = 25
    hk = T / N
    umax = 20
    dmax = 2
    d = 1
    
    # Initial state
    x0 = [0, 0, -pi/4, 0]
    xd = [1, 0, 0, 0]

    # Import pendulum model
    model = InvertedPendulum()
    model.set_state(np.reshape(x0, (4, 1)))
    model.dt = hk
    
    # Define decision variables
    u = cvxpy.Variable((1, N))
    x = cvxpy.Variable((4, N))
    
    # Define objective
    cost = 0
    for k in range(N-1):
        cost += (hk / 2) * (u[:, k] ** 2 + u[:, k+1] ** 2)
    objective = cvxpy.Minimize(cost)
    
    # Create collocation constraints
    constr = []
    for k in range(N-1):
        # dx = x[:, k][1]
        # ddx = (u + model.m * model.l * x[:, k][3] ** 2 * sin(x[:, k][2]) + model.m * model.g * cos(x[:, k][2]) *
                 # sin(x[:, k][2])) / (model.M + model.m - model.m * cos(x[:, k][2]) ** 2) - model.b_x * model.state[1]
        # dtheta = x[:, k][3]
        # ddtheta = (-cos(x[:, k][2]) / model.l) * f_k[1] + (model.g / model.l) * sin(x[:, k][2]) - model.b_theta * model.state[3]
        # f_k = np.array([dx, ddx, dtheta, ddtheta])
        # f_k1 = model.dynamics(x[:, k+1], u[:, k+1])
        constr += [(hk / 2) * (model.A_cont @ (x[:, k]) + model.A_cont @ (x[:, k+1])) == x[:, k+1] - x[:, k]]
    
    # Create path constraints
    constr += [x[:, k][0] <= dmax for k in range(N-1)]
    constr += [x[:, k][0] >= -dmax for k in range(N-1)]
    
    # Create control constraints
    constr += [u[:, i] <= umax for i in range(N-1)]
    constr += [u[:, i] >= -umax for i in range(N-1)]
    
    # Create bound constraints
    constr += [x[:, 0] == x0]
    constr += [x[:, N-1] == xd]
    
    # Create initial guess
    u_guess = np.zeros((1, N))
    x_guess = np.zeros((4, N))
    for k in range(N):
        x_guess[:, k] = k * (T / N) * np.array(xd)
    
    # Create problem
    prob = cvxpy.Problem(objective, constr)
    
    # Solve problem
    u.value = u_guess
    x.value = x_guess
    prob.solve(warm_start=True, verbose=False)
    
    # Get optimal result
    if prob.status == cvxpy.OPTIMAL:
        ou = u.value
        ox = x.value
    else:
        print("No solution found")

    u_t = []
    t = 0
    for k in range(N-1):
        t_array = np.linspace(t, t+hk, 5)[:-1]
        u_t.append(ou[0][k] + ((t_array - t) / hk) * (ou[0][k + 1] - ou[0][k]))
        t += hk
    u_t = np.ravel(u_t)

    x_t = []
    t = 0
    for k in range(N-1):
        t_array = np.linspace(t, t + hk, 5)[:-1]
        f_k = np.array(model.A_cont @ (ox[:, k]) + model.B_cont @ ou[:, k])
        f_k1 = np.array(model.A_cont @ (ox[:, k + 1]) + model.B_cont @ ou[:, k + 1])
        x_t.append(ox[:, k] + f_k.T * (t_array - t) + (f_k1 - f_k).T * ((t_array - t) ** 2 / hk))
        # print(ox[:, k] + np.multiply(f_k, (t_array - t)) + ((t_array - t) ** 2 / hk) * (f_k1 - f_k))
        t += hk
    # x_t = np.reshape(x_t, (4, len(x_t)))
    
    # Plot collocation points
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(np.linspace(0, T, 25), ou[0], '.r')
    # plt.plot(np.linspace(0, T, 4*24), u_t, '-')
    # plt.xlabel("Time (s)")
    # plt.ylabel('Control output')
    # plt.subplot(3, 1, 2)
    # plt.plot(np.linspace(0, T, 25), ox[0, :], '.r')
    # plt.plot(np.linspace(0, T, 4 * 24), x_t[0, :], '-')
    # plt.xlabel("Time (s)")
    # plt.ylabel('Cart position (m)')
    # plt.subplot(3, 1, 3)
    # plt.plot(np.linspace(0, T, 25), ox[2, :], '.r')
    # plt.plot(np.linspace(0, T, 4 * 24), x_t[2, :], '-')
    # plt.xlabel("Time (s)")
    # plt.ylabel('Pendulum angle (rad)')
    #plt.show()
    
    simulate()