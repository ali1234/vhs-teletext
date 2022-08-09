import unittest

from teletext.stats import *


class TestStatsList(unittest.TestCase):

    def test_str(self):
        l = StatsList()
        l.append('a')
        l.append('b')
        self.assertEqual(str(l), 'ab')
