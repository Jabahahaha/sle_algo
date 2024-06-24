import unittest
import numpy as np
from generators.matrix_generators import generate_random_matrix, generate_hilbert_matrix

class TestMatrixGenerators(unittest.TestCase):
    def test_generate_random_matrix(self):
        A = generate_random_matrix(5, diagonally_dominant=True)
        self.assertEqual(A.shape, (5, 5))
        for i in range(5):
            self.assertTrue(A[i, i] > np.sum(np.abs(A[i, :])) - A[i, i])  # Diagonally dominant check

    def test_generate_hilbert_matrix(self):
        H = generate_hilbert_matrix(5)
        self.assertEqual(H.shape, (5, 5))
        self.assertTrue(np.allclose(H, np.linalg.inv(np.linalg.inv(H))))

if __name__ == '__main__':
    unittest.main()
