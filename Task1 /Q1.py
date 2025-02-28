import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
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

# Measure execution time for different matrix sizes
sizes = [2**i for i in range(2, 8)]  # Matrix sizes: 4, 8, 16, 32, 64
execution_times = []

for n in sizes:
    A = np.random.randint(1, 10, (n, n))
    B = np.random.randint(1, 10, (n, n))
    
    start_time = time.time()
    naive_matrix_multiplication(A, B)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    print(f"Size {n}x{n}: {execution_time:.6f} seconds")
    log_sizes = np.log(sizes)
log_times = np.log(execution_times)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_times)
# Plot results
plt.figure(figsize=(8,5))
plt.plot(sizes, execution_times, marker='o', linestyle='-', label="Measured Time")
plt.plot(sizes, [t**3 for t in sizes], linestyle='--', label="O(n³) Trend")  
plt.xlabel("Matrix Size (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Time Complexity of Naive Divide-and-Conquer Matrix Multiplication")
plt.legend()
plt.grid()
plt.show()

print(f"Calculated scaling exponent: {slope:.4f}")
print(f"R-squared: {r_value**2:.4f}")

#gives calculates scaling exponent of 2.8839
#evaluates R-squared as 0.9999