import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def create_sparse_hamiltonian(N, J=1.0):
    """
    Create a sparse Hamiltonian matrix for the spin chain
    
    Args:
    N (int): Number of spin sites
    J (float): Coupling constant
    
    Returns:
    scipy.sparse.csr_matrix: Sparse Hamiltonian matrix
    """
    from scipy.sparse import lil_matrix
    
    # Smaller dimension for sparse matrix
    dim = 2**N
    H = lil_matrix((dim, dim), dtype=complex)
    
    # Iterate through all sites
    for n in range(N):
        next_site = (n + 1) % N
        
        # Iterate through all possible states
        for state in range(dim):
            # Check if spins can be flipped
            if not (state & (1 << n)) and not (state & (1 << next_site)):
                # Flip both spins
                flipped_state = state ^ (1 << n) ^ (1 << next_site)
                
                # Add coupling term
                H[flipped_state, state] += -0.5 * J
    
    return H.tocsr()

def calculate_magnon_energy(N, p, J=1.0):
    """
    Calculate magnon energy using sparse matrix techniques
    
    Args:
    N (int): Number of spin sites
    p (float): Momentum value
    J (float): Coupling constant
    
    Returns:
    float: Energy of the magnon state
    """
    # Create sparse Hamiltonian
    H = create_sparse_hamiltonian(N, J)
    
    # Create magnon state
    magnon_state = np.zeros(2**N, dtype=complex)
    
    # Construct magnon state
    for n in range(N):
        # Apply phase factor and spin lowering
        phase = np.exp(1j * p * n)
        
        for state in range(2**N):
            if not (state & (1 << n)):  # Check if spin is up
                flipped_state = state ^ (1 << n)
                magnon_state[flipped_state] += phase
    
    # Normalize the state
    magnon_state /= np.sqrt(N)
    
    # Compute energy using sparse matrix multiplication
    energy = np.dot(magnon_state.conj(), H.dot(magnon_state)).real
    
    return energy

def theoretical_energy(p, J=1.0):
    """
    Calculate theoretical magnon energy from dispersion relation
    
    Args:
    p (float): Momentum value
    J (float): Coupling constant
    
    Returns:
    float: Theoretical energy
    """
    return 2 * J * np.sin(p/2)**2

def plot_energy_comparison(N):
    """
    Plot and compare computed and theoretical energies
    
    Args:
    N (int): Number of spin sites
    """
    # Generate momentum values
    k_values = np.arange(N)
    p_values = 2 * np.pi * k_values / N
    
    # Compute energies
    computed_energies = [calculate_magnon_energy(N, p) for p in p_values]
    theoretical_energies = [theoretical_energy(p) for p in p_values]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, computed_energies, 'ro', label='Computed Energy')
    plt.plot(k_values, theoretical_energies, 'b-', label='Theoretical Energy')
    plt.xlabel('k')
    plt.ylabel('Energy')
    plt.title(f'Magnon Energy Levels for Spin Chain (N = {N})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print detailed comparison
    print("\nDetailed Energy Comparison:")
    for k in [0, 1, 5, 10, 15, 20, 25]:
        p = 2 * np.pi * k / N
        computed = calculate_magnon_energy(N, p)
        theoretical = theoretical_energy(p)
        print(f"k = {k}: Computed = {computed:.4f}, Theoretical = {theoretical:.4f}")

# Run the example for N = 30
plot_energy_comparison(30)
