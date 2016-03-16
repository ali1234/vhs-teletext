import numpy

from collections import defaultdict

from scipy.stats.mstats import mode

from functools import partial
from operator import itemgetter
from teletext.misc.all import All
from packet import Packet, HeaderPacket
from subpage import Subpage
from service import Service

def reader(infile):
    """Helper to read t42 lines from a file-like object."""
    lines = iter(partial(infile.read, 42), b'')
    for l in lines:
        if len(l) < 42:
            return
        else:
            yield Packet.from_bytes(l)



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

def paginate(packet_iter, pages=All, yield_func=packets):
    """Reorders lines in a t42 stream so that pages are continuous."""
    magbuffers = [[],[],[],[],[],[],[],[]]
    for packet in packet_iter:
        mag = packet.mrag.magazine
        if type(packet) == HeaderPacket:
            if len(magbuffers[mag]) > 0 and type(magbuffers[mag][0]) == HeaderPacket:
                if magbuffers[mag][0].page_str() in pages:
                    magbuffers[mag].sort(key=lambda p: p.mrag.row)
                    for item in yield_func(magbuffers[mag]):
                        yield item
            magbuffers[mag] = []
        magbuffers[mag].append(packet)


def subpage_squash(packet_iter, pages=All, yield_func=packets):
    subpages = defaultdict(list)
    for pl in paginate(packet_iter, pages=pages, yield_func=packet_lists):
        subpagekey = (pl[0].mrag.magazine, pl[0].header.page, pl[0].header.subpage)
        arr = numpy.zeros((42, 32), dtype=numpy.uint8)
        for p in pl:
            arr[:,p.mrag.row] = p._original_bytes
        subpages[subpagekey].append(arr)

    for arrlist in subpages.itervalues():
        arr = mode(numpy.array(arrlist), axis=0)[0][0].astype(numpy.uint8)
        packets = []

        for i in range(32):
            if arr[:,i].any():
                packets.append(Packet.from_bytes(arr[:,i]))

        for item in yield_func(packets):
            yield item



def make_service(packet_iter, pages=All):
    service = Service()
    for s in paginate(packet_iter, pages=pages, yield_func=subpages):
        service.magazines[s._original_magazine].pages[s._original_page].subpages[s._original_subpage] = s

    return service

def make_squashed_service(packet_iter, pages=All):
    service = Service()
    for s in subpage_squash(packet_iter, pages=pages, yield_func=subpages):
        service.magazines[s._original_magazine].pages[s._original_page].subpages[s._original_subpage] = s

    return service
