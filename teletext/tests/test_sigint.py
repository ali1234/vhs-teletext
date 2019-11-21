import os
import signal
import sys
import time
import unittest

from teletext.sigint import SigIntDefer


def ctrl_c(pid):
    if sys.platform.startswith('win'):
        # Note: on Windows this doesn't get delivered immediately.
        os.kill(pid, signal.CTRL_C_EVENT)
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGINT)


class TestSigInt(unittest.TestCase):

    def test_ctrl_c(self):
        with self.assertRaises(KeyboardInterrupt):
            ctrl_c(os.getpid())

    def test_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.assertRaises(ValueError):
                with SigIntDefer() as s:
                    self.assertFalse((s.fired))
                    ctrl_c(os.getpid())
                    self.assertTrue((s.fired))
                    raise ValueError
