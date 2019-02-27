import os
import unittest

import numpy as np

from teletext.file import FileChunker
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

    @unittest.expectedFailure
    def test_known_teletext(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), 'data', 'teletext.vbi'), 'rb') as f:
                lines = (Line(data, number) for number, data in FileChunker(f, 2048))
                for line in lines:
                    self.assertTrue(line.is_teletext, f'Line {line._number} false negative.')
        except FileNotFoundError:
            self.skipTest('Known teletext data not available.')

    @unittest.expectedFailure
    def test_known_reject(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), 'data', 'reject.vbi'), 'rb') as f:
                lines = (Line(data, number) for number, data in FileChunker(f, 2048))
                for line in lines:
                    self.assertFalse(line.is_teletext, f'Line {line._number} false positive.')
        except FileNotFoundError:
            self.skipTest('Known reject data not available.')

