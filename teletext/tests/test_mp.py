from multiprocessing import current_process
import unittest
from functools import wraps
from itertools import count, islice
import os
import sys
import time

from teletext.mp import itermap, PureGeneratorPool, _PureGeneratorPoolSingle, _PureGeneratorPoolMP

from .test_sigint import ctrl_c


def multiply(it, a):
    for x in it:
        yield x*a


def null(it, a):
    for x in it:
        yield (x, a)


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


def early_crash(it):
    raise ValueError('Crashed early on purpose.')


def not_generator(it):
    return 23


class TestMPSingle(unittest.TestCase):
    procs = 1
    desired_type = _PureGeneratorPoolSingle

    def setUp(self):
        global callcounter
        callcounter = 0

    def test_single(self):
        input = list(range(100))
        expected = list(multiply(input, 3))
        result = list(itermap(multiply, input, self.procs, 3))
        self.assertListEqual(result, expected)

    def test_called_once_single(self):
        result = list(itermap(callcount, [None] * (self.procs + 1), processes=self.procs))
        self.assertListEqual(result, [1] * (self.procs + 1))

    def test_reuse(self):
        input = list(range(100))
        expected = list(multiply(input, 3))
        with PureGeneratorPool(multiply, self.procs, 3) as pool:
            self.assertIsInstance(pool, self.desired_type)
            result = list(pool.apply(input[:50]))
            self.assertListEqual(result, expected[:50])
            result = list(pool.apply(input[50:]))
            self.assertListEqual(result, expected[50:])

    def test_called_once_reuse(self):
        with PureGeneratorPool(callcount, processes=self.procs) as pool:
            for n in range(self.procs + 1): # ensure at least one process is used twice
                result = list(pool.apply([None]))
                self.assertListEqual(result, [1])

    def _crashing_iter(self, n):
        with self.assertRaises(ChildProcessError if self.procs > 1 else ValueError):
            list(itermap(crashy, ([False]*n) + [True], processes=self.procs))

    def test_crashing_iter(self):
        self._crashing_iter(0)
        self._crashing_iter(self.procs + 1)
        self._crashing_iter(40)

    def test_early_crash(self):
        with self.assertRaises(ChildProcessError if self.procs > 1 else ValueError):
            list(itermap(early_crash, ([False]*3), self.procs))

    def test_not_generator(self):
        with self.assertRaises(ChildProcessError if self.procs > 1 else TypeError):
            list(itermap(not_generator, ([False]*3), self.procs))

    def test_too_many_args(self):
        with self.assertRaises(ChildProcessError if self.procs > 1 else TypeError):
            list(itermap(multiply, ([False]*3), self.procs, 3, 4))


class TestMPMulti(TestMPSingle):
    procs = 2
    desired_type = _PureGeneratorPoolMP

    def test_unpickleable_function(self):
        with self.assertRaises(AttributeError):
            list(itermap(lambda x: x, ([False] * 3), self.procs))

    def test_unpickleable_item_in_args(self):
        with self.assertRaises(AttributeError):
            list(itermap(null, ([None]*10), self.procs, lambda x: x))

    def test_unpickleable_item_in_iter(self):
        with self.assertRaises(AttributeError):
            list(itermap(null, ([None]*10) + [lambda x: x], self.procs, None))


class TestMPMultiSigInt(unittest.TestCase):

    pool_size = 4

    def items(self):
        with PureGeneratorPool(multiply, self.pool_size, 1) as pool:
            self.pool = pool
            yield from pool.apply(islice(count(), 100))

    def test_sigint_to_self(self):
        result = self.items()
        with self.assertRaises(KeyboardInterrupt):
            for r in result:
                ctrl_c(os.getpid())

    @unittest.skipIf(sys.platform.startswith('win'), "Can't send ctrl-c to an individual process on Windows")
    def test_sigint_to_child(self):
        result = self.items()
        for r in result:
            ctrl_c(self.pool._pool[r%self.pool_size][0].pid)
