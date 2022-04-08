import unittest
import numpy as np
from pbgca import Cluster


class ClusterTestCase(unittest.TestCase):

    def setUp(self):
        self.cluster = Cluster([1,1,1])

    def test_unit_galaxy_assign(self):
        """Test cluster created with one center parameter assign center to the galaxy"""

        result = self.cluster.galaxies
        self.assertEqual(result.tolist()[0], [1,1,1])

    def test_zero_coors(self):
        """Test with zero center in coords"""

        result = Cluster([0,0,0]).galaxies
        self.assertEqual(result.tolist()[0], [0,0,0])




if __name__ == '__main__':
    unittest.main()