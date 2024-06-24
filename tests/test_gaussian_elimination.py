import unittest
import numpy as np
from algorithms.gaussian_elimination import gaussian_elimination

class TestGaussianElimination(unittest.TestCase):
    def test_small_system(self):
        A = np.array([[3, 2], [1, 2]], dtype=float)
        b = np.array([7, 5], dtype=float)
        x_expected = np.linalg.solve(A, b)
        x, _ = gaussian_elimination(A, b)
        np.testing.assert_array_almost_equal(x, x_expected, decimal=5)

if __name__ == '__main__':
    unittest.main()
