import unittest
import numpy as np
from utils.metrics import calculate_metrics

class TestMetrics(unittest.TestCase):
    def test_calculate_metrics(self):
        A = np.array([[3, 1], [1, 2]], dtype=float)
        x_true = np.array([2, 3], dtype=float)
        b = np.dot(A, x_true)
        x = np.linalg.solve(A, b)
        residual, distance = calculate_metrics(A, x, b, x_true)
        self.assertAlmostEqual(residual, 0.0, places=5)
        self.assertAlmostEqual(distance, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
