import io
import unittest

import numpy as np

from teletext.file import FileChunker

class TestChunker(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(0, 256, dtype=np.uint8)
        self.file = io.BytesIO(self.data.tobytes())

    def test_basic(self):
        result = list(FileChunker(self.file, 1))
        self.assertEqual(len(result), len(self.data))
        for n in range(256):
            self.assertEqual(result[n], (n, bytes([n])))

    def test_step(self):
        result = list(FileChunker(self.file, 1, step=2))
        self.assertEqual(len(result), len(self.data[::2]))
        for n in range(128):
            self.assertEqual(result[n], (n*2, bytes([n*2])))
