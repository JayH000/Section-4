import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def greens_function_lu(omega, H):
    
    # Create identity matrix of same size as H
    N = H.shape[0]
    I = np.eye(N)
    
    # Construct (ωI - H) matrix
    matrix = omega * I - H
    
    # Use LU decomposition to solve linear system
    # Equivalent to solving (ωI - H)G = I
    G = la.solve(matrix, I)
    
    return G

def generate_hamiltonian(N):
   
    # Create a random matrix
    H = np.random.rand(N, N)
    
    # Make the matrix Hermitian
    H = (H + H.conj().T) / 2
    
    return H

def plot_greens_function(N=30):
   
    # Generate a random Hermitian Hamiltonian
    H = generate_hamiltonian(N)
    
    # Range of frequencies to explore
    frequencies = np.linspace(-10, 10, 200)
    
    # Compute Green's function for each frequency
    greens_values = [np.trace(greens_function_lu(omega, H)) for omega in frequencies]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, greens_values, label='Green\'s Function Trace')
    plt.title(f'Green\'s Function for {N}x{N} Hamiltonian')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('Trace of Green\'s Function')
    plt.grid(True)
    plt.legend()
    plt.show()

# Demonstrate the Green's function computation
plot_greens_function()