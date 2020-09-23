from src.BalancingRobot import *
from src.Simulator import *
from src.controllers.FiniteHorizonController import *

# This script shows the behavior of the robot controlled by a finite horizon controller

if __name__ == "__main__":
    
    # Import model
    model = BalancingRobot()
    
    # Set initial state [m, m/s, rad, rad/s]
    model.set_state(np.mat([[0.1], [0.0]]))
    
    # Define controller
    controller = Control(model)

    # Simulate
    sim = Simulator(model, controller)
    t, state_list = sim.simulate()

    # Plot data
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t, state_list[:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Pendulum angle")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(t, state_list[:, 1])
    plt.xlabel("Time (s)")
    plt.ylabel("thetadot (rad/s)")
    plt.title("Pendulum angular velocity")
    plt.grid()
    plt.tight_layout()

    # Save figure in data
    plt.savefig("../data/robot_finiteHorizon_result.png")
    plt.show()
