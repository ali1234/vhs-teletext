import unittest

import numpy as np

from teletext.vbi.line import Line
from teletext.vbi.config import Config


class LineTestCase(unittest.TestCase):

    def noisegen(self):
        for loc in range(0, 256):
            for scale in range(0, 256):
                for n in range(10):
                    yield (np.clip(np.random.normal(loc, scale, size=(2048,)), 0, 255).astype(np.uint8).tobytes(), {'loc':loc, 'scale':scale, 'n':n})

    def setUp(self):
        Line.set_config(Config())
        Line.try_cuda = False

    def test_noise_rejection(self):
        for data, params in self.noisegen():
            line = Line(data)
            self.assertFalse(line.is_teletext, f'Noise interpreted as teletext: {params}.')
