import unittest
import numpy as np
from generators.matrix_generators import generate_random_matrix
from generators.rhs_vector import generate_rhs_vector

class TestRHSVector(unittest.TestCase):
    def test_generate_rhs_vector(self):
        A = generate_random_matrix(3)
        b, x_true = generate_rhs_vector(A)
        self.assertEqual(b.shape[0], A.shape[0])
        self.assertEqual(x_true.shape[0], A.shape[0])

if __name__ == '__main__':
    unittest.main()
