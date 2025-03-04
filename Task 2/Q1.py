import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import matplotlib.pyplot as plt
def heisenberg_xxx(N, J=1.0):
    """
    Constructs the Heisenberg XXX Hamiltonian matrix for N spins with periodic boundary conditions.
    
    Parameters:
    N (int): Number of spins in the chain.
    J (float): Interaction strength (default is 1.0).
    
    Returns:
    H (scipy.sparse.csr_matrix): The sparse Hamiltonian matrix.
    """
    dim = 2**N  # Hilbert space dimension
    H = sp.lil_matrix((dim, dim), dtype=np.float64)  # Use LIL format for easy element assignment
    
    # Basis states are represented by integers (bitstrings)
    for i in range(N):  # Iterate over each site
        j = (i + 1) % N  # Next neighbor with periodic boundary condition
        
        for state in range(dim):
            # Get spin values at sites i and j (0 = |↓⟩, 1 = |↑⟩)
            si = (state >> i) & 1  # Extract i-th bit
            sj = (state >> j) & 1  # Extract j-th bit
            
            # Sz_i Sz_j term
            H[state, state] += J * (0.25 if si == sj else -0.25)
            
            # S+_i S-_j + S-_i S+_j terms (hopping terms)
            if si != sj:  # Flip spins if they are different
                flipped_state = state ^ ((1 << i) | (1 << j))  # Flip bits i and j
                H[state, flipped_state] += J * 0.5
    
    return H.tocsr()  # Convert to CSR format for efficient calculations

def analyze_time_complexity(max_N=10):
    """
    Analyze the runtime and compare with theoretical time complexity.
    
    Parameters:
    max_N (int): Maximum number of spins to test.
    """
    # Lists to store experimental measurements
    N_values = list(range(2, max_N + 1))
    runtimes = []
    
    # Measure actual runtimes
    for N in N_values:
        start = time.time()
        H = heisenberg_xxx(N)
        end = time.time()
        runtimes.append(end - start)
    
    # Theoretical complexity analysis
    # For Heisenberg XXX matrix: O(N * 2^N)
    theoretical_times = [N * (2**N) * 1e-6 for N in N_values]  # Scale factor added for visualization
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, runtimes, 'bo-', label='Experimental Runtime')
    plt.plot(N_values, theoretical_times, 'r--', label='Theoretical Complexity')
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Time (seconds)')
    plt.title('Heisenberg XXX Matrix Construction: Runtime vs Theoretical Complexity')
    plt.legend()
    plt.yscale('log')  # Use log scale to better visualize exponential growth
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\nDetailed Runtime and Complexity Analysis:")
    print("N\tExperimental Runtime\tTheoretical Complexity")
    for N, runtime, theo_time in zip(N_values, runtimes, theoretical_times):
        print(f"{N}\t{runtime:.6f} s\t\t{theo_time:.6f} s")

# Run the complexity analysis
analyze_time_complexity(max_N=8)





"""""
 Example/ test usage
N = 3  # Number of spins
H = heisenberg_xxx(N)
print("Hamiltonian matrix (dense representation):\n", H.toarray())


printed result;
Hamiltonian matrix (dense representation):
 [[ 0.75  0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.   -0.25  0.5   0.    0.5   0.    0.    0.  ]
 [ 0.    0.5  -0.25  0.    0.5   0.    0.    0.  ]
 [ 0.    0.    0.   -0.25  0.    0.5   0.5   0.  ]
 [ 0.    0.5   0.5   0.   -0.25  0.    0.    0.  ]
 [ 0.    0.    0.    0.5   0.   -0.25  0.5   0.  ]
 [ 0.    0.    0.    0.5   0.    0.5  -0.25  0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.75]]
"""
