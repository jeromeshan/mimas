import unittest
import pandas as pd
from pbgca import Cluster
from pbgca import Clusterer
import random
from sklearn.metrics.cluster import adjusted_rand_score

class ClustererTestCase(unittest.TestCase):

    def setUp(self):
        data = []
        true_labels = []
        for i in range(10,50):
            random.seed(42)
            rand1 = random.uniform(-5, 5)
            random.seed(42)
            rand2 = random.uniform(-5, 5)
            random.seed(42)
            rand3 = random.uniform(-5, 5)
            random.seed(42)
            rand4 = random.uniform(-5, 5)

            data.append([i+rand1,i+rand2])
            data.append([-1*i+rand3,-1*i+rand4])
            true_labels.append(0)
            true_labels.append(1)
        self.data = pd.DataFrame(data)
        self.true_labels = true_labels

    def test_simple_case(self):
        """Test cluster created with one center parameter assign center to the galaxy"""

        clusterer = Clusterer(epsilon= 10, lr = 1, max_iter = 10, limit_radian = 0.1, grow_limit= 5, min_diff = -1,parallel = False)
        labels=clusterer.fit(self.data)
        self.assertEqual(adjusted_rand_score(self.true_labels,labels), 1)


if __name__ == '__main__':
    unittest.main()