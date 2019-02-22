import unittest

import numpy as np

from teletext.vbi.line import Line
from teletext.vbi.config import Config


class LineTestCase(unittest.TestCase):

    def noisegen(self, max_loc, max_scale):
        for n in range(10):
            for loc in range(0, max_loc+1, max(1, max_loc//8)):
                for scale in range(0, max_scale+1, max(1, max_scale//8)):
                    yield (
                        np.clip(np.random.normal(loc, scale, size=(2048,)), 0, 255).astype(np.uint8).tobytes(),
                        {'loc':loc, 'scale':scale}
                    )

    def setUp(self):
        Line.configure(Config(), force_cpu=True)

    def test_empty_rejection(self):
        lines = ((Line(data), params) for data, params in self.noisegen(256, 8))
        lines = ((line, params) for line, params in lines if line.is_teletext)
        for line, params in lines:
            self.assertFalse(line.is_teletext, f'Noise interpreted as teletext: {params}')
