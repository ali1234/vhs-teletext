import io
import os
import unittest

import numpy as np

from teletext.vbi.pattern import Pattern
from teletext.vbi.patterncuda import PatternCUDA


class PatternCUDATestCase(unittest.TestCase):

    def setUp(self):
        p = os.path.dirname(__file__) + '/../../vbi/data/parity.dat'
        self.pattern = Pattern(p)
        self.patterncuda = PatternCUDA(p)

    def test_equal_to_cpu(self):
        arr = np.arange(256, dtype=np.uint8)
        a = self.pattern.match(arr)
        b = self.patterncuda.match(arr)

        self.assertTrue(all(a==b), 'CPU and CUDA pattern matching produced different results.')
