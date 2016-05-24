import numpy

from collections import defaultdict

from scipy.stats.mstats import mode

from functools import partial
from operator import itemgetter
from teletext.misc.all import All
from packet import Packet, HeaderPacket
from subpage import Subpage
from service import Service
import itertools

def reader(infile, start=0, stop=-1):
    """Helper to read t42 lines from a file-like object."""
    if start > 0:
        infile.seek(start * 42)
    lines = iter(partial(infile.read, 42), b'')
    for n,l in enumerate(lines):
        offset = n + start
        if len(l) < 42:
            return
        elif offset == stop:
            return
        else:
            p = Packet.from_bytes(l)
            p._offset = offset
            yield p



def demux(packet_iter, magazines=All, rows=All):
    """Filters t42 stream to a subset of magazines and packets."""
    for packet in packet_iter:
        if packet.mrag.magazine in magazines:
            if packet.mrag.row in rows:
                yield packet



def packets(packet_list):
    for p in packet_list:
        yield p

def packet_lists(packet_list):
    yield packet_list

def subpages(packet_list):
    yield Subpage.from_packets(packet_list)

def paginate(packet_iter, pages=All, yield_func=packets, drop_empty=False):
    """Reorders lines in a t42 stream so that pages are continuous."""
    magbuffers = [[],[],[],[],[],[],[],[]]
    for packet in packet_iter:
        mag = packet.mrag.magazine
        if type(packet) == HeaderPacket:
            if ((drop_empty==False and len(magbuffers[mag]) > 0) or len(magbuffers[mag]) > 1) and type(magbuffers[mag][0]) == HeaderPacket:
                if magbuffers[mag][0].page_str() in pages:
                    magbuffers[mag].sort(key=lambda p: p.mrag.row)
                    for item in yield_func(magbuffers[mag]):
                        yield item
            magbuffers[mag] = []
        magbuffers[mag].append(packet)


def subpage_squash(packet_iter, minimum_dups=3, pages=All, yield_func=packets):
    subpages = defaultdict(list)
    for pl in paginate(packet_iter, pages=pages, yield_func=packet_lists, drop_empty=True):
        subpagekey = (pl[0].mrag.magazine, pl[0].header.page, pl[0].header.subpage)
        arr = numpy.zeros((42, 32), dtype=numpy.uint8)
        for p in pl:
            arr[:,p.mrag.row] = p._original_bytes
        subpages[subpagekey].append(arr)

    for arrlist in subpages.itervalues():
        if len(arrlist) >= minimum_dups:
            arr = mode(numpy.array(arrlist), axis=0)[0][0].astype(numpy.uint8)
            packets = []

            for i in range(32):
                if arr[:,i].any():
                    packets.append(Packet.from_bytes(arr[:,i]))

            for item in yield_func(packets):
                yield item

def split_seq(iterable, size):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def row_squash(packet_iter, n_rows):

    for l_list in split_seq(packet_iter, n_rows):
        a = numpy.array([numpy.fromstring(l.to_bytes(), dtype=numpy.uint8) for l in l_list])
        best, counts = mode(a)
        best = best[0].astype(numpy.uint8)
        p = Packet.from_bytes(best)
        p._offset = l_list[0]._offset
        yield p


def make_service(packet_iter, pages=All):
    service = Service()
    for s in paginate(packet_iter, pages=pages, yield_func=subpages):
        service.magazines[s._original_magazine].pages[s._original_page].subpages[s._original_subpage] = s

    for k,v in service.magazines.iteritems():
        v.magazineno = k

    return service


