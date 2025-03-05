import numpy as np
import matplotlib.pyplot as plt

def qr_algorithm(H, max_iterations=100, tolerance=1e-10):
    """
    Implement the QR algorithm for eigenvalue decomposition.
    
    Parameters:
    -----------
    H : numpy.ndarray
        Initial square matrix to decompose
    max_iterations : int, optional
        Maximum number of iterations to prevent infinite loops
    tolerance : float, optional
        Convergence threshold for detecting diagonal-like matrix
    
    Returns:
    --------
    tuple: (final_matrix, iterations, convergence_history)
        - final_matrix: The nearly diagonal matrix after QR iterations
        - iterations: Number of iterations performed
        - convergence_history: List tracking matrix changes
    """
    # Ensure input is a numpy array and float type for precision
    H_o = np.array(H, dtype=float)
    
    # Validate input matrix is square
    if H_o.ndim != 2 or H_o.shape[0] != H_o.shape[1]:
        raise ValueError("Input must be a square matrix")
    
    # Initialize tracking variables
    convergence_history = []
    off_diagonal_norms = []
    
    for iteration in range(max_iterations):
        # QR Decomposition
        Q_o, R_o = np.linalg.qr(H_o)
        
        # Next iteration matrix: H_1 = R_o * Q_o
        H_1 = R_o @ Q_o
        
        # Track changes and check convergence
        off_diagonal_norm = np.linalg.norm(
            np.triu(H_1, k=1)  # Upper triangular part above main diagonal
        )
        off_diagonal_norms.append(off_diagonal_norm)
        convergence_history.append(H_1.copy())
        
        # Convergence check: are off-diagonal elements sufficiently small?
        if off_diagonal_norm < tolerance:
            return H_1, iteration + 1, convergence_history
        
        # Update for next iteration
        H_o = H_1
    
    # If max iterations reached without convergence
    print("Warning: Maximum iterations reached without full convergence")
    return H_o, max_iterations, convergence_history

def visualize_convergence(convergence_history):
    """
    Visualize the convergence of the QR algorithm.
    
    Parameters:
    -----------
    convergence_history : list
        List of matrices representing each iteration's state
    """
    # Convert convergence history to a matrix for visualization
    conv_array = np.array(convergence_history)
    
    plt.figure(figsize=(15, 5))
    
    # Plot heatmap of matrix evolution
    plt.subplot(131)
    plt.imshow(conv_array[-1], cmap='viridis')
    plt.title('Final Matrix State')
    plt.colorbar()
    
    # Plot off-diagonal norm convergence
    plt.subplot(132)
    off_diagonal_norms = [np.linalg.norm(np.triu(mat, k=1)) for mat in convergence_history]
    plt.plot(off_diagonal_norms)
    plt.title('Off-Diagonal Norm Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Off-Diagonal Norm')
    plt.yscale('log')
    
    # Plot diagonal elements progression
    plt.subplot(133)
    diagonals = [np.diag(mat) for mat in convergence_history]
    diag_array = np.array(diagonals)
    
    for i in range(diag_array.shape[1]):
        plt.plot(diag_array[:, i], label=f'Diagonal Element {i+1}')
    
    plt.title('Diagonal Elements Progression')
    plt.xlabel('Iterations')
    plt.ylabel('Diagonal Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create a random symmetric matrix for demonstration
    np.random.seed(42)
    N = 5  # Matrix size
    H = np.random.rand(N, N)
    H = (H + H.T) / 2  # Symmetrize
    
    print("Original Matrix:")
    print(H)
    
    # Run QR algorithm
    diagonalized_matrix, iterations, convergence_history = qr_algorithm(H)
    
    print(f"\nConverged after {iterations} iterations")
    print("\nDiagonalized Matrix:")
    print(diagonalized_matrix)
    
    # Visualize convergence
    visualize_convergence(convergence_history)
    
    # Verify near-diagonal nature and eigenvalues
    eigenvalues = np.diag(diagonalized_matrix)
    print("\nComputed Eigenvalues:")
    print(eigenvalues)
    
    # Compare with numpy's eigenvalue computation
    np_eigenvalues = np.linalg.eigvals(H)
    print("\nNumPy Eigenvalues:")
    print(sorted(np_eigenvalues))