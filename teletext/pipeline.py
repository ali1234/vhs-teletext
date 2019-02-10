import numpy

from collections import defaultdict

from scipy.stats.mstats import mode

from .packet import Packet
from .subpage import Subpage
from .service import Service
import itertools


def packets(packet_list):
    yield from packet_list


def packet_lists(packet_list):
    yield packet_list


def subpages(packet_list):
    yield Subpage.from_packets(packet_list)


def check_buffer(mb, pages, yield_func, min_rows=0):
    if (len(mb) > min_rows) and mb[0].type == 'header':
        page = mb[0].header.page | (mb[0].mrag.magazine * 0x100)
        if page in pages:
            yield from yield_func(sorted(mb, key=lambda p: p.mrag.row))


def paginate(packet_iter, pages=range(0x000, 0x900), yield_func=packets, drop_empty=False):
    """Reorders lines in a t42 stream so that pages are contiguous."""
    magbuffers = [[],[],[],[],[],[],[],[]]
    for packet in packet_iter:
        mag = packet.mrag.magazine & 0x7
        if packet.type == 'header':
            yield from check_buffer(magbuffers[mag], pages, yield_func, 1 if drop_empty else 0)
            magbuffers[mag] = []
        magbuffers[mag].append(packet)
    for mb in magbuffers:
        yield from check_buffer(mb, pages, yield_func, 1 if drop_empty else 0)


def subpage_squash(packet_iter, minimum_dups=3, pages=range(0x000, 0x900), yield_func=packets):
    """
            iter = subpage_squash(iter, pages=args.pages)
    """
    subpages = defaultdict(list)
    for pl in paginate(packet_iter, pages=pages, yield_func=packet_lists, drop_empty=True):
        subpagekey = (pl[0].mrag.magazine, pl[0].header.page, pl[0].header.subpage)
        arr = numpy.zeros((42, 32), dtype=numpy.uint8)
        for p in pl:
            arr[:,p.mrag.row] = p._original_bytes
        subpages[subpagekey].append(arr)

    for arrlist in subpages.values():
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
    """
            iter = row_squash(iter, args.squash_rows)
    """
    for l_list in split_seq(packet_iter, n_rows):
        a = numpy.array([numpy.fromstring(l.to_bytes(), dtype=numpy.uint8) for l in l_list])
        best, counts = mode(a)
        best = best[0].astype(numpy.uint8)
        p = Packet.from_bytes(best)
        p._offset = l_list[0]._offset
        yield p


def make_service(packet_iter, pages=range(0x100)):
    service = Service()
    for s in paginate(packet_iter, pages=pages, yield_func=subpages):
        service.magazines[s._original_magazine].pages[s._original_page].subpages[s._original_subpage] = s

    for k,v in service.magazines.iteritems():
        v.magazineno = k

    return service
