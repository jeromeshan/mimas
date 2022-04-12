import unittest
from pbgca import CubeGenerator
import numpy as np

class CubeGeneratorTestCase(unittest.TestCase):


    def test_2d(self):
        """Test 2d cube generation"""

        cube = CubeGenerator.n_dim_cube(2)
        self.assertEqual(cube.tolist(), np.transpose(np.array([[ -0.5, -0.5, 0.5, 0.5], [ -0.5, 0.5, 0.5, -0.5]])).tolist())

    def test_3d(self):
            """Test 3d cube generation"""

            cube = CubeGenerator.n_dim_cube(3)
            self.assertEqual(cube.tolist(), [[-0.5, -0.5, -0.5],[-0.5,  0.5, -0.5],[ 0.5,  0.5, -0.5],[ 0.5, -0.5, -0.5],[-0.5, -0.5,  0.5],[-0.5,  0.5,  0.5],[ 0.5,  0.5,  0.5],[ 0.5, -0.5,  0.5]])

if __name__ == '__main__':
    unittest.main()