import numpy as np

def split_matrix(matrix):
    """Split a matrix into four equal quadrants."""
    n = len(matrix)
    mid = n // 2
    a11 = [row[:mid] for row in matrix[:mid]]
    a12 = [row[mid:] for row in matrix[:mid]]
    a21 = [row[:mid] for row in matrix[mid:]]
    a22 = [row[mid:] for row in matrix[mid:]]
    return a11, a12, a21, a22

def add_matrices(a, b):
    """Add two matrices."""
    n = len(a)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = a[i][j] + b[i][j]
    return result

def subtract_matrices(a, b):
    """Subtract matrix b from matrix a."""
    n = len(a)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = a[i][j] - b[i][j]
    return result

def strassen(a, b):
    """
    Strassen's algorithm for matrix multiplication.
    Input: n x n matrices a and b where n is a power of 2
    Output: n x n matrix, product of a and b
    """
    n = len(a)
    
    # Base case: 1x1 matrix
    if n == 1:
        return [[a[0][0] * b[0][0]]]
    
    # Split matrices into quadrants
    a11, a12, a21, a22 = split_matrix(a)
    b11, b12, b21, b22 = split_matrix(b)
    
    # Calculate 7 products (these are the key to Strassen's algorithm)
    p1 = strassen(a11, subtract_matrices(b12, b22))
    p2 = strassen(add_matrices(a11, a12), b22)
    p3 = strassen(add_matrices(a21, a22), b11)
    p4 = strassen(a22, subtract_matrices(b21, b11))
    p5 = strassen(add_matrices(a11, a22), add_matrices(b11, b22))
    p6 = strassen(subtract_matrices(a12, a22), add_matrices(b21, b22))
    p7 = strassen(subtract_matrices(a11, a21), add_matrices(b11, b12))
    
    # Calculate the quadrants of the result
    c11 = add_matrices(subtract_matrices(add_matrices(p5, p4), p2), p6)
    c12 = add_matrices(p1, p2)
    c21 = add_matrices(p3, p4)
    c22 = subtract_matrices(subtract_matrices(add_matrices(p5, p1), p3), p7)
    
    # Combine the quadrants into a single matrix
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n//2):
        for j in range(n//2):
            result[i][j] = c11[i][j]
            result[i][j + n//2] = c12[i][j]
            result[i + n//2][j] = c21[i][j]
            result[i + n//2][j + n//2] = c22[i][j]
    
    return result

def pad_matrix(matrix):
    """Pad a matrix with zeros to make its dimensions a power of 2."""
    n = len(matrix)
    m = max(n, len(matrix[0]))
    
    # Find the next power of 2
    power_of_two = 1
    while power_of_two < m:
        power_of_two *= 2
    
    # Create a padded matrix filled with zeros
    padded = [[0 for _ in range(power_of_two)] for _ in range(power_of_two)]
    
    # Copy the original matrix elements
    for i in range(n):
        for j in range(len(matrix[i])):
            padded[i][j] = matrix[i][j]
    
    return padded, n, len(matrix[0])

def strassen_multiply(a, b):
    """
    Wrapper for Strassen's algorithm that handles non-square matrices
    and matrices with dimensions that are not powers of 2.
    """
    # Check if matrices can be multiplied
    if len(a[0]) != len(b):
        raise ValueError("Matrix dimensions do not match for multiplication")
    
    # Pad matrices to make dimensions powers of 2
    padded_a, orig_rows_a, orig_cols_a = pad_matrix(a)
    padded_b, orig_rows_b, orig_cols_b = pad_matrix(b)
    
    # Multiply using Strassen's algorithm
    result_padded = strassen(padded_a, padded_b)
    
    # Extract the relevant part of the result
    result = [[result_padded[i][j] for j in range(orig_cols_b)] for i in range(orig_rows_a)]
    
    return result

# Example usage
if __name__ == "__main__":
    # Example matrices
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    
    B = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
    
    # Using Strassen's algorithm
    result = strassen_multiply(A, B)
    
    # Print the result
    print("Result using Strassen's algorithm:")
    for row in result:
        print(row)
    
    # Verify with NumPy
    numpy_result = np.dot(np.array(A), np.array(B))
    print("\nVerification with NumPy:")
    print(numpy_result)

    #Result using Strassen's algorithm:
#[58, 64]
#[139, 154]

#Verification with NumPy:
#[[ 58  64]
 #[139 154]]S