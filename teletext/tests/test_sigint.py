import os
import signal
import unittest

from teletext.sigint import SigIntDefer


class TestSigInt(unittest.TestCase):

    def test_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.assertRaises(ValueError):
                with SigIntDefer() as s:
                    os.kill(os.getpid(), signal.SIGINT)
                    self.assertTrue((s.fired))
                    raise ValueError
