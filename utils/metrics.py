import numpy as np

def calculate_metrics(A, x, b, x_true):
    residual = np.linalg.norm(np.dot(A, x) - b)
    distance_to_solution = np.linalg.norm(x - x_true)
    return residual, distance_to_solution
