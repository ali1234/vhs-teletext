import unittest

from teletext.mp import itermap


def func(it):
    for x in it:
        yield x*x


class TestStatsList(unittest.TestCase):

    def setUp(self):
        self.input = list(range(512))
        self.result = list(func(self.input))

    def test_single(self):
        result = list(itermap(func, self.input, processes=1))
        self.assertListEqual(result, self.result)

    #@unittest.skip # breaks coverage - https://github.com/nedbat/coveragepy/issues/745
    def test_multi(self):
        result = list(itermap(func, self.input, processes=2))
        self.assertListEqual(result, self.result)
