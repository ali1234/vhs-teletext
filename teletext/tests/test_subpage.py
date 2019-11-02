import unittest

import numpy as np

from teletext.subpage import Subpage


class SubpageTestCase(unittest.TestCase):

    def test_checksum(self):
        p = Subpage()
        self.assertEqual(p.checksum, 0xe23d)
        p = Subpage(prefill=True)
        self.assertEqual(p.checksum, 0xe23d)

