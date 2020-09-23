from src.InvertedPendulum import *
from src.Simulator import *

# This script shows the normal behavior of the pendulum without control

if __name__ == "__main__":
    
    # Import model
    model = InvertedPendulum()

    # Set initial state [m, m/s, rad, rad/s]
    model.set_state(np.mat([[0.0], [0.0], [0.1], [0.0]]))

    # Define controller
    controller = None

    # Simulate
    sim = Simulator(model, controller)
    t, state_list = sim.simulate()

    # Plot data
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, state_list[:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.title("Cart position")
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(t, state_list[:, 1])
    plt.xlabel("Time (s)")
    plt.ylabel("xdot (m/s)")
    plt.title("Cart velocity")
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(t, state_list[:, 2])
    plt.xlabel("Time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Pendulum angle")
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(t, state_list[:, 3])
    plt.xlabel("Time (s)")
    plt.ylabel("thetadot (rad/s)")
    plt.title("Pendulum angular velocity")
    plt.grid()
    plt.tight_layout()

    # Save figure in data
    plt.savefig("../data/Dynamics_result.png")
    plt.show()

