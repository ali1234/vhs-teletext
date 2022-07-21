import io
import itertools
import os
import stat


def PossiblyInfiniteRange(start=0, stop=None, step=1, limit=None):
    if stop is None:
        if limit is None:
            return itertools.count(start, step)
        else:
            return range(start, start + (limit * step), step)
    else:
        if limit is None:
            return range(start, stop, step)
        else:
            return range(start, min(stop, start + (limit * step)), step)


class LenWrapper(object):
    def __init__(self, i, l):
        self.i = i
        self.l = l

    def __iter__(self):
        return self.i

    def __len__(self):
        return self.l


def _chunks(f, size, flines, frange, seek):
    while True:
        if seek:
            f.seek(size * frange.start, os.SEEK_CUR)
        else:
            f.read(size * frange.start)
        for _ in frange:
            b = f.read(size)
            if len(b) < size:
                return
            yield b
        if seek:
            f.seek(size * (flines - frange.stop), os.SEEK_CUR)
        else:
            f.read(size * (flines - frange.stop))


def chunks(f, size, start, step, flines=16, frange=(0, 16), seek=True):
    while True:
        c = _chunks(f, size, flines, frange, seek)
        try:
            for _ in range(start):
                next(c)
            while True:
                yield next(c)
                for i in range(step-1):
                    next(c)
        except StopIteration:
            if seek:
                f.seek(0, os.SEEK_SET)
            else:
                return

def FileChunker(f, size, start=0, stop=None, step=1, limit=None, flines=16, frange=range(0, 16), loop=False):
    seekable = False
    try:
        if hasattr(f, 'fileno') and stat.S_ISFIFO(os.fstat(f.fileno()).st_mode):
            raise io.UnsupportedOperation

        f.seek(0, os.SEEK_END)
        total_lines = f.tell() // size
        total_fields = total_lines // flines
        remainder = max(min((total_lines % flines) - frange.start, len(frange)), 0)
        useful_lines = (total_fields * len(frange)) + remainder

        if stop is None:
            stop = useful_lines
        else:
            stop = min(stop, useful_lines)

        seekable = True
        f.seek(0, os.SEEK_SET)

    except io.UnsupportedOperation:
        # chunks() always seeks to the start
        pass

    r = PossiblyInfiniteRange(start, None if loop else stop, step, limit)
    i = zip(r, chunks(f, size, start, step, flines, frange, seek=seekable))
    if hasattr(r, '__len__'):
        return LenWrapper(i, len(r))
    else:
        return i
