from src.InvertedPendulum import *
import casadi as ca
import matplotlib.pyplot as plt

# Trajectory parameters
T = 2
N = 25
hk = T / N
umax = 20
dmax = 2
d = 1

# Import pendulum model
model = InvertedPendulum()

# Initial state
x0 = [0, 0, 0, 0]
xd = [0, 0, -pi/4, 0]

# Define decision variables
u = ca.SX.sym('u', 1, N)
x = ca.SX.sym('x', 4, N)
w = ca.vertcat(u, x)

# Define objective
cost = 0
for k in range(N-1):
    cost += (hk / 2) * (u[:, k] ** 2 + u[:, k+1] ** 2)

# Create collocation constraints
constr = []
constr += [(hk / 2) * (model.A_cont @ (x[:, k]) + model.A_cont @ (x[:, k+1])) == x[:, k+1] - x[:, k] for k in range(N-1)]

# Create path constraints
# constr += [x[:, k][0] <= dmax for k in range(N-1)]
# constr += [x[:, k][0] >= -dmax for k in range(N-1)]

# Create control constraints
# constr += [u[:, i] <= umax for i in range(N-1)]
# constr += [u[:, i] >= -umax for i in range(N-1)]

# Create bound constraints
constr += [x[:, 0] == x0]
constr += [x[:, N-1] == xd]

# Create initial guess
u_guess = np.zeros((1, N))
x_guess = np.zeros((4, N))
for k in range(N):
    x_guess[:, k] = k * (T / N) * np.array(xd)
w_guess = ca.vertcat(u_guess, x_guess)

# Create an NLP solver
prob = {'f': cost, 'x': w, 'g': constr}
solver = ca.nlpsol('solver', 'ipopt', prob)

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

print(u.value)
print(x.value)
ou = np.array(u.value[0, :]).flatten()
ox = np.array(x.value[0, :]).flatten()
    
# Plot collocation points
t = np.arange(N)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, ou)
plt.xlabel("Time (s)")
plt.ylabel('Control output')
plt.subplot(3, 1, 2)
plt.plot(t, ox)
plt.xlabel("Time (s)")
plt.ylabel('Cart position (m)')
plt.subplot(3, 1, 3)
plt.plot(t, ox)
plt.xlabel("Time (s)")
plt.ylabel('Pendulum angle (rad)')
