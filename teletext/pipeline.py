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

"""
def pipe():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('inputfile', type=str, help='Read VBI samples from this file.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--ansi',
                       help='Output lines in ANSI format suitable for console display. Default if STDOUT is a tty.',
                       action='store_true')
    group.add_argument('-t', '--t42',
                       help='Output lines in T42 format for further processing. Default if STDOUT is not a tty.',
                       action='store_true')

    parser.add_argument('-r', '--rows', type=int, metavar='R', nargs='+', help='Only pass packets from these rows.',
                        default=range(32))
    parser.add_argument('-m', '--mags', type=int, metavar='M', nargs='+',
                        help='Only pass packets from these magazines.', default=range(9))
    parser.add_argument('-n', '--numbered',
                        help='When output is ascii, number packets according to offset in input file.',
                        action='store_true')
    parser.add_argument('-p', '--pages', type=str, metavar='M', nargs='+',
                        help='Only pass packets from these magazines.', default=range(0x100))
    parser.add_argument('-P', '--paginate', help='Re-order output lines so pages are continuous.', action='store_true')
    parser.add_argument('-S', '--squash', help='Squash pages.', action='store_true')
    parser.add_argument('-s', '--squash-rows', metavar='N', type=int, help='Merge N consecutive rows to reduce output.',
                        default=1)

    parser.add_argument('--spellcheck', help='Try to fix common errors with a spell checking dictionary.',
                        action='store_true')

    parser.add_argument('-H', '--headers', help='Synonym for --rows 0 31.', action='store_true')

    parser.add_argument('-W', '--windowed', help='Output in a separate window.', action='store_true')
    parser.add_argument('-L', '--less', help='Page the output with less.', action='store_true')

    parser.add_argument('--start', type=int, metavar='N', help='Start after the Nth line of the input file.', default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stop', type=int, metavar='N', help='Stop before the Nth line of the input file.', default=-1)
    group.add_argument('--count', type=int, metavar='N', help='Stop after processing N lines from the input file.',
                       default=-1)

    args = parser.parse_args()

    if not args.t42 and not args.ansi:
        if sys.stdout.isatty():
            args.ansi = True
        else:
            args.t42 = True

    if args.stop == -1 and args.count > -1:
        args.stop = args.start + args.count

    if args.headers:
        args.rows = {0, 31}

    # this sucks but it will get removed soon
    if any(i not in args.pages for i in range(0x100)):
        args.paginate = True

    if args.windowed or args.less:
        from .terminal import termify
        termify(args.windowed, args.less)

    infile = open(args.inputfile, 'rb')

    iter = demux(reader(infile, args.start, args.stop), magazines=args.mags, rows=args.rows)

    if args.squash:
        iter = subpage_squash(iter, pages=args.pages)
    elif args.paginate:
        iter = paginate(iter, pages=args.pages)
    elif args.squash_rows > 1:
        iter = row_squash(iter, args.squash_rows)

    if args.spellcheck:
        from .spellcheck import spellcheck
    else:
        spellcheck = None

    for packet in iter:
        if spellcheck is not None:
            spellcheck(packet)
        if args.ansi:
            if args.numbered:
                print('%8d' % packet._offset, end='')
            print(packet.to_ansi())
        else:
            x = packet.to_bytes()
            if len(x) != 42 and len(x) != 0:
                raise IndexError("No" + str(type(packet)))
            sys.stdout.write(x)
"""
