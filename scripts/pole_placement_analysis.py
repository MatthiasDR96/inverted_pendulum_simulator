import control
from src.InvertedPendulum import *

# This script computes the control gain matrix K to place the system's poles at desired locations

# Insert model
model = InvertedPendulum()

# Get system and input matrices
A = model.A_cont
B = model.B_cont

# Check stability of the system by calculating eigenvalues and eigenvectors, mention positive eigenvalues (unstable)
w, v = np.linalg.eig(A)
print("Eigenvalues of the system matrix: \n" + str(w))

# Controllability
print('\nControllability')
print('Rank of ctrb(A,b): ', np.linalg.matrix_rank(control.ctrb(A, B)))

# Pole Placement, compute control gain
desired_eigenvalues = [-1, -2, -4, -5]
K = control.place(A, B, desired_eigenvalues)
print('\nPole Placement to eigenvalues: ' + str(desired_eigenvalues))
print('K= ', K)

# Verification of Eigen values of A-BK
w_, v_ = np.linalg.eig(A - B * K)
print("\nEigenvalues of A-BK: \n" + str(w_))
