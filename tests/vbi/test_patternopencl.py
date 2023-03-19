import pathlib
import unittest

import numpy as np

from teletext.vbi.pattern import Pattern
try:
    from teletext.vbi.patternopencl import PatternOpenCL


    class PatternOpenCLTestCase(unittest.TestCase):

        def setUp(self):
            p = pathlib.Path(__file__).parent.parent.parent / 'teletext' / 'vbi' / 'data' / 'vhs' / 'parity.dat'
            self.pattern = Pattern(p)
            self.patternopencl = PatternOpenCL(p)

        def test_equal_to_cpu(self):
            arr = np.arange(256, dtype=np.uint8)
            a = self.pattern.match(arr)
            b = self.patternopencl.match(arr)

            #self.assertTrue(all(a==b), 'CPU and OpenCL pattern matching produced different results.')
            self.assertEqual(a.tolist(), b.tolist(), 'CPU and OpenCL pattern matching produced different results.')

except ModuleNotFoundError as e:
    if e.name != 'pyopencl':
        raise

