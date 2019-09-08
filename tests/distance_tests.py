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

        # Paper's example:
        # A = (1, 3), B = (2, 2), and C = (0, 2)
        # b  10, then Dist(A, B) = 2, Dist(A, C) = 20,
        # and Dist(B, C) = 22
        A = np.array([1, 3])
        B = np.array([2, 2])
        C = np.array([0, 2])
        b = 10
        ab = hotaling(A, B, b)
        ac = hotaling(A, C, b)
        bc = hotaling(B, C, b)

        self.assertEqual(ab, 2.0)
        self.assertEqual(ac, 20.0)
        self.assertEqual(bc, 22.0)


if __name__ == "__main__":
    unittest.main()
