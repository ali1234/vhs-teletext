import io
import itertools
import os


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


def chunks(f, size, step, seek=True):
    while True:
        b = f.read(size)
        if len(b) < size:
            return
        yield b
        if step > 1:
            if seek:
                f.seek(size * (step - 1), os.SEEK_CUR)
            else:
                f.read(size * (step - 1))


def FileChunker(f, size, start=0, stop=None, step=1, limit=None):
    seekable = False
    try:
        f.seek(0, os.SEEK_END)
        f_len = f.tell()//size

        if stop is None:
            stop = f_len
        else:
            stop = min(stop, f_len)

        f.seek(size * start, os.SEEK_SET)
        seekable = True

    except io.UnsupportedOperation:
        f.read(size * start)

    r = PossiblyInfiniteRange(start, stop, step, limit)
    i = zip(r, chunks(f, size, step, seek=seekable))
    if hasattr(r, '__len__'):
        return LenWrapper(i, len(r))
    else:
        return i
