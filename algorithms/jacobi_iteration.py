import numpy as np
import time

def jacobi_iteration(A, b, tol=1e-10, max_iterations=500):
    x = np.zeros_like(b)
    n = len(b)
    iterations = 0
    converged = False

    start_time = time.time()

    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            try:
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            except ZeroDivisionError:
                return x, time.time() - start_time, iterations, False

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            converged = True
            break

        x = x_new
        iterations += 1

    elapsed_time = time.time() - start_time
    return x, elapsed_time, iterations, converged
