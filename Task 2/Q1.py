import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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

# Example usage
N = 3  # Number of spins
H = heisenberg_xxx(N)
print("Hamiltonian matrix (dense representation):\n", H.toarray())

""""
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
