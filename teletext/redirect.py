import os
import sys

class DropStderr:
    def __enter__(self):
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(self.devnull, 2)
        os.close(self.devnull)
        del self.devnull

    def __exit__(self, *args):
        os.dup2(self.old_stderr, 2)
        os.close(self.old_stderr)
        del self.old_stderr
