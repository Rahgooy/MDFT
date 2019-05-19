import unittest
import numpy as np
from helpers.distances import hotaling
import math


class distances_tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_hotaling(self):
        o1 = np.array([1, 3])
        o2 = np.array([2, 5])
        b = 10
        calculated = hotaling(o1, o2, b)

        e1 = o1[0]
        e2 = o2[0]
        q1 = o1[1]
        q2 = o2[1]
        ΔE = e2 - e1
        ΔQ = q2 - q1
        ΔI = (ΔQ - ΔE) / math.sqrt(2)
        ΔD = (ΔQ + ΔE) / math.sqrt(2)
        actual = ΔI ** 2 + b * ΔD ** 2

        self.assertTrue(np.allclose(actual, calculated))


if __name__ == "__main__":
    unittest.main()
