import numpy as np

def generate_random_matrix(size, diagonally_dominant=False):
    np.random.seed(42)  # For reproducibility
    A = np.random.rand(size, size)
    if diagonally_dominant:
        D = np.diag(np.abs(A).sum(axis=1))
        A = A + D
    return A

def generate_hilbert_matrix(size):
    H = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            H[i, j] = 1 / (i + j + 1)
    return H
