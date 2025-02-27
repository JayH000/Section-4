import numpy as np

def naive_matrix_multiplication(A, B):
    """ Bottom-up divide-and-conquer matrix multiplication (iterative). """
    n = A.shape[0]
    C = np.zeros((n, n))

    # Step size doubling (bottom-up approach)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(0, n, 2 * step):
                for k in range(0, n, 2 * step):
                    # Extract submatrices
                    A11, A12, A21, A22 = A[i:i+step, k:k+step], A[i:i+step, k+step:k+2*step], A[i+step:i+2*step, k:k+step], A[i+step:i+2*step, k+step:k+2*step]
                    B11, B12, B21, B22 = B[k:k+step, j:j+step], B[k:k+step, j+step:j+2*step], B[k+step:k+2*step, j:j+step], B[k+step:k+2*step, j+step:j+2*step]

                    # Compute submatrix multiplications
                    C[i:i+step, j:j+step] += A11 @ B11 + A12 @ B21
                    C[i:i+step, j+step:j+2*step] += A11 @ B12 + A12 @ B22
                    C[i+step:i+2*step, j:j+step] += A21 @ B11 + A22 @ B21
                    C[i+step:i+2*step, j+step:j+2*step] += A21 @ B12 + A22 @ B22

        step *= 2  # Increase the block size

    return C

# Example usage
n = 4  # Matrix size (must be power of 2)
A = np.random.randint(1, 10, (n, n))
B = np.random.randint(1, 10, (n, n))

C = naive_matrix_multiplication(A, B)
print("Result Matrix C:\n", C)
