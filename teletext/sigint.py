import signal

class SigIntDefer(object):

    """
    SigIntDefer is a context manager which catches SIGINT (aka KeyboardInterrupt)
    and allows the code to check if it has happened or not, and then exit at an
    appropriate time, instead of in the middle of doing something important.
    """

    def __init__(self, times=1):
        self._times = 1
        self._fired = None

    def __enter__(self):
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)
        return self

    def handler(self, f, n):
        self._fired = (f, n)
        self._times -= 1
        if self._times == 0:
            signal.signal(signal.SIGINT, self._old_handler)

    @property
    def fired(self):
        return self._fired is not None

    def __exit__(self, *args, **kwargs):
        signal.signal(signal.SIGINT, self._old_handler)
        if self._fired:
            self._old_handler(*self._fired)



if __name__ == '__main__':

    # Example: the goal here is to make sure that if we print "hello", we always
    # print "goodbye", even if a KeyboardInterrupt happens. If a KeyboardInterrupt
    # did happen, SigIntDefer will re-fire it upon exiting the context.

    import time

    def loop():
        with SigIntDefer() as sigint:
            while True:
                if sigint.fired:
                    return
                print("hello")
                time.sleep(0.5)
                print("goodbye")

    loop()
