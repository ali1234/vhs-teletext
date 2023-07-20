import unittest

import numpy as np

from teletext.subpage import Subpage


class SubpageTestCase(unittest.TestCase):

    def test_checksum(self):
        p = Subpage()
        self.assertEqual(0xe23d, p.checksum)
        p = Subpage(prefill=True)
        self.assertEqual(0xe23d, p.checksum)

