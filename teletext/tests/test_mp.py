import unittest

from teletext.mp import itermap, PureGeneratorPool, _PureGeneratorPoolSingle, _PureGeneratorPoolMP


def square(it):
    for x in it:
        yield x*x


callcounter = 0
def callcount(it):
    global callcounter
    callcounter += 1
    for x in it:
        yield callcounter


def crashy(it):
    for x in it:
        if x:
            raise ValueError('Crashed on purpose.')
        else:
            yield x


def crashy_quiet(it):
    import sys
    sys.stderr = open('/dev/null') # so as not to spam all over the tests
    crashy(it)



class TestMPSingle(unittest.TestCase):
    procs = 1
    desired_type = _PureGeneratorPoolSingle

    def setUp(self):
        self.input = list(range(200))
        self.result = list(square(self.input))
        global callcounter
        callcounter = 0

    def test_single(self):
        result = list(itermap(square, self.input, processes=self.procs))
        self.assertListEqual(result, self.result)

    def test_called_once_single(self):
        result = list(itermap(callcount, [None] * (self.procs + 1), processes=self.procs))
        self.assertListEqual(result, [1] * (self.procs + 1))

    def test_reuse(self):
        with PureGeneratorPool(square, processes=self.procs) as pool:
            self.assertIsInstance(pool, self.desired_type)
            result = list(pool.apply(self.input[:100]))
            self.assertListEqual(result, self.result[:100])
            result = list(pool.apply(self.input[100:]))
            self.assertListEqual(result, self.result[100:])

    def test_called_once_reuse(self):
        with PureGeneratorPool(callcount, processes=self.procs) as pool:
            for n in range(self.procs + 1): # ensure at least one process is used twice
                result = list(pool.apply([None]))
                self.assertListEqual(result, [1])

    def _crashing_iter(self, n):
        with self.assertRaises(ValueError):
            list(itermap(crashy, ([False]*n) + [True], processes=self.procs))

    def test_crashing_iter(self):
        self._crashing_iter(0)
        self._crashing_iter(1)
        self._crashing_iter(self.procs + 1)
        self._crashing_iter(105)


class TestMPMulti(TestMPSingle):
    procs = 2
    desired_type = _PureGeneratorPoolMP

    def _crashing_iter(self, n):
        with self.assertRaises(ChildProcessError):
            list(itermap(crashy_quiet, ([False]*n) + [True], processes=self.procs))
