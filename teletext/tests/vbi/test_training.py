import io
import unittest

import numpy as np

from teletext.vbi.training import *
from teletext.vbi.config import Config


class TrainingTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_split(self):
        files = [io.BytesIO() for _ in range(256)]
        pattern = load_pattern()
        max_n = 10

        data = [(n, np.unpackbits(pattern[n:n+pattern_length][::-1])[::-1]) for n in range(max_n)]
        split(data, files)

        pattern_bits = np.unpackbits(pattern[:max_n+pattern_length][::-1])[::-1]
        patterns_present = set()
        for x in range(len(pattern_bits) - 23):
            patterns_present.add(
                np.packbits(pattern_bits[x:x+24][::-1])[::-1].tobytes()
            )

        for f in files[:1]:
            arr = np.fromstring(f.getvalue(), dtype=np.uint8).reshape(-1, 27)
            for l in arr:
                # Assert that pattern matches samples.
                self.assertTrue(all(l[:3] == np.packbits(l[3:][::-1])[::-1]))
                # Assert that pattern is a pattern we actually put in to split.
                self.assertIn(l[:3].tobytes(), patterns_present)

