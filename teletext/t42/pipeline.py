import numpy

from collections import defaultdict

from scipy.stats.mstats import mode

from coding import mrag_decode, page_decode, subcode_bcd_decode
from functools import partial
from operator import itemgetter
from teletext.misc.all import All

def reader(infile):
    """Helper to read t42 lines from a file-like object."""
    lines = iter(partial(infile.read, 42), b'')
    for l in lines:
        if len(l) < 42:
            return
        else:
            yield numpy.fromstring(l, dtype=numpy.uint8)


def demux(line_iter, magazines=All, rows=All):
    """Filters t42 stream to a subset of magazines and packets."""
    for l in line_iter:
        ((m, r), e) = mrag_decode( l[:2] )
        if m in magazines:
            if r in rows:
                yield l


def paginate(line_iter, pages=All, yield_lines=True):
    """Reorders lines in a t42 stream so that pages are continuous."""
    magbuffers = [[],[],[],[],[],[],[],[]]
    for l in line_iter:
        ((m, r), e) = mrag_decode( l[:2] )
        if r == 0:
            magbuffers[m].sort(key=itemgetter(0))
            if len(magbuffers[m]) > 0:
                page = '%d%02x' % (m, page_decode(magbuffers[m][0][1][2:4])[0])
                if page in pages:
                    if yield_lines:
                        for br,bl in magbuffers[m]:
                            yield bl
                    else:
                        page_array = numpy.zeros((42,32), dtype=numpy.uint8)
                        for br,bl in magbuffers[m]:
                            page_array[:,br] = bl
                        yield page,page_array
            magbuffers[m] = []
        magbuffers[m].append((r,l))

class PageContainer(object):
    def __init__(self):
        self.subpages = defaultdict(list)

    def insert(self, arr):
        ((subpage, control), subpage_error) = subcode_bcd_decode(arr[4:10,0])
        self.subpages[subpage].append(arr)

def page_squash(page_iter):
    pages = defaultdict(PageContainer)
    for n,page in page_iter:
        pages[n].insert(page)
    for n,pc in pages.iteritems():
      for s,pl in pc.subpages.iteritems():
       print len(pl)
       if len(pl) > 1:
        arr = numpy.array(pl)
        m = mode(arr, axis=0)
        for i in range(25):
            #print m[0][:,i]
            yield m[0][0,:,i].astype(numpy.uint8)
