import control
import scipy
from src.InvertedPendulum import *

# This script computes the optimal control gain matrix K for LQR control


def lqr(A, B, Q, R):

    """Solve the continuous time lqr controller.
    
    dx/dt = A x + B u
    
    cost = integral x.T*Q*x + u.T*R*u
    """
    # first, try to solve the ricatti equation
    X = np.mat(scipy.linalg.solve_continuous_are(A, B, Q, R))
    
    # compute the LQR gain
    K = np.mat(scipy.linalg.inv(R) * (B.T * X))
    
    return K


if __name__ == "__main__":

    # Insert model
    model = InvertedPendulum()

    # Get system and input matrices
    A = model.A_cont
    B = model.B_cont

    # Control parameters, Q = state penalty, R = control penalty
    Q = np.array([[500, 0, 0, 0], [0, 250, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.identity(1)

    # Check stability of the system by calculating eigenvalues and eigenvectors, mention positive eigenvalues (unstable)
    w, v = np.linalg.eig(A)
    print("Eigenvalues of the system matrix: \n" + str(w))

    # Controllability
    print('\nControllability')
    print('Rank of ctrb(A,b): ', np.linalg.matrix_rank(control.ctrb(A, B)))

    # Pole placement, compute optimal control gain
    K = lqr(A, B, Q, R)
    print('\nK= ', K)

    # Verification of Eigen values of A-BK
    w_, v_ = np.linalg.eig(A - B * K)
    print("\nEigenvalues of A-BK: \n" + str(w_))
