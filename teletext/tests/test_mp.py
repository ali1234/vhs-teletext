import unittest

from teletext.mp import itermap, PureGeneratorPool


def func(it):
    for x in it:
        yield x*x


class TestMP(unittest.TestCase):

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

    def test_reuse(self):
        with PureGeneratorPool(func, processes=2) as pool:
            result = list(pool.apply(self.input[:100]))
            self.assertListEqual(result, self.result[:100])
            result = list(pool.apply(self.input[100:]))
            self.assertListEqual(result, self.result[100:])
