import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def greens_function_cholesky(omega, H):
    """
    Compute Green's function using Cholesky decomposition
    
    Parameters:
    -----------
    omega : float
        Frequency parameter
    H : numpy.ndarray
        Hermitian matrix (positive definite)
    
    Returns:
    --------
    G : numpy.ndarray
        Green's function matrix
    """
    # Create identity matrix of same size as H
    N = H.shape[0]
    I = np.eye(N)
    
    # Construct (ωI - H) matrix
    matrix = omega * I - H
    
    try:
        # Attempt Cholesky decomposition
        # This works only for positive definite matrices
        L = la.cholesky(matrix, lower=True)
        
        # Solve linear system using Cholesky decomposition
        G = la.cho_solve((L, True), I)
        
        return G
    except la.LinAlgError:
        # Fallback to standard solve if matrix is not positive definite
        print(f"Warning: Matrix not positive definite at ω = {omega}")
        return la.solve(matrix, I)

def generate_positive_definite_matrix(N):
    """
    Generate a random positive definite matrix
    
    Parameters:
    -----------
    N : int
        Size of the matrix
    
    Returns:
    --------
    H : numpy.ndarray
        Positive definite matrix
    """
    # Create a random matrix
    A = np.random.rand(N, N)
    
    # Ensure positive definiteness
    return A.T @ A + N * np.eye(N)

def plot_greens_function_cholesky(N=30):
    """
    Plot Green's function for different frequencies using Cholesky
    
    Parameters:
    -----------
    N : int, optional
        Size of the Hamiltonian matrix (default 30)
    """
    # Generate a positive definite matrix
    H = generate_positive_definite_matrix(N)
    
    # Range of frequencies to explore
    frequencies = np.linspace(-10, 10, 200)
    
    # Compute Green's function for each frequency
    greens_values = [np.trace(greens_function_cholesky(omega, H)) for omega in frequencies]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, greens_values, label='Green\'s Function Trace')
    plt.title(f'Green\'s Function for {N}x{N} Matrix (Cholesky)')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('Trace of Green\'s Function')
    plt.grid(True)
    plt.legend()
    plt.show()

# Demonstrate the Green's function computation
plot_greens_function_cholesky()