# Inverted Pendulum Simulator

This is a Python package for the simulation of an inverted pendulum.

## Installation

It is sufficient to just download Python (3.6) and to download the package as a .zip folder from github.
Once the package is opened in your favorite IDE (I recommend Pycharm), you're ready to go.

## Simulator Structure

### Src

In the src folder, all the controllers for stabilizing the pendulum are situated as well as the
inverted pendulum model class and the main simulator class.

#### Simulator class

The Simulator class is the main class of the simulator and contains all functions for rendering the simulator.
The constructor accepts a robot model and a controller. The simulator is set to run for 1000 frames at 0.05
frames per second.

#### Controllers

In the controllers folder, all the controllers are implemented:

-   pid controller
-   pole placement controller
-   LQR controller
-   finite horizon controller
-   MPC controller

Each controller has a functions to set a target and implements a function 'control'
which inputs the full state and outputs the control signal.

### Scripts

The scripts folder contains all the examples of using different controllers.
When you run each script you should be able to see the simulation run. You can adapt the start state.
Pay attention when changing variables.

## Errata

It is definitely possible that bugs are present in the code or that some theory pieces are wrongly implemented.
If you notice any of these, please contact me.


