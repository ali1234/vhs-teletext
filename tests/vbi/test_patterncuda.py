import pathlib
import unittest

import numpy as np

from teletext.vbi.pattern import Pattern
try:
    from teletext.vbi.patterncuda import PatternCUDA


    class PatternCUDATestCase(unittest.TestCase):

        def setUp(self):
            p = pathlib.Path(__file__).parent.parent.parent / 'teletext' / 'vbi' / 'data' / 'vhs' / 'parity.dat'
            self.pattern = Pattern(p)
            self.patterncuda = PatternCUDA(p)

        def test_equal_to_cpu(self):
            arr = np.arange(256, dtype=np.uint8)
            a = self.pattern.match(arr)
            b = self.patterncuda.match(arr)

            self.assertTrue(all(a==b), 'CPU and CUDA pattern matching produced different results.')

except ModuleNotFoundError as e:
    if e.name != 'pycuda':
        raise

