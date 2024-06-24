import numpy as np
import time

def gaussian_elimination(A, b):
    start_time = time.time()
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    elapsed_time = time.time() - start_time
    return x, elapsed_time
