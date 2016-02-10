import numpy

from coding import mrag_decode
from functools import partial
from operator import itemgetter

def reader(infile):
    """Helper to read t42 lines from a file-like object."""
    lines = iter(partial(infile.read, 42), b'')
    for l in lines:
        if len(l) < 42:
            return
        else:
            yield numpy.fromstring(l, dtype=numpy.uint8)


def demux(line_iter, magazines=None, rows=None):
    """Filters t42 stream to a subset of magazines and packets."""
    for l in line_iter:
        ((m, r), e) = mrag_decode( l[:2] )
        if magazines is None or m in magazines:
            if rows is None or r in rows:
                yield l


def paginate(line_iter):
    """Reorders lines in a t42 stream so that pages are continuous."""
    magbuffers = [[],[],[],[],[],[],[],[]]
    for l in line_iter:
        ((m, r), e) = mrag_decode( l[:2] )
        if r == 0:
            magbuffers[m].sort(key=itemgetter(0))
            for br,bl in magbuffers[m]:
                yield bl
            magbuffers[m] = []
        magbuffers[m].append((r,l))
