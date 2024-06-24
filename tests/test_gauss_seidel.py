import unittest
import numpy as np
from algorithms.gauss_seidel import gauss_seidel

class TestGaussSeidel(unittest.TestCase):
    def test_small_diagonally_dominant(self):
        A = np.array([[4, 1], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        x_expected = np.linalg.solve(A, b)
        x, _, _, converged = gauss_seidel(A, b)
        self.assertTrue(converged)
        np.testing.assert_array_almost_equal(x, x_expected, decimal=5)

    def test_larger_system(self):
        A = np.array([[10, 1, 1], [2, 10, 1], [2, 2, 10]], dtype=float)
        b = np.array([12, 13, 14], dtype=float)
        x_expected = np.linalg.solve(A, b)
        x, _, _, converged = gauss_seidel(A, b)
        self.assertTrue(converged)
        np.testing.assert_array_almost_equal(x, x_expected, decimal=5)

    def test_non_convergence(self):
        A = np.array([[1, 2], [2, 1]], dtype=float)
        b = np.array([1, 1], dtype=float)
        x, _, _, converged = gauss_seidel(A, b, max_iterations=100)
        self.assertFalse(converged)

if __name__ == '__main__':
    unittest.main()
