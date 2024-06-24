import numpy as np

def generate_rhs_vector(A):
    x_true = np.ones(A.shape[0])
    b = np.dot(A, x_true)
    return b, x_true
