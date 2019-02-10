import numpy

from collections import defaultdict

from scipy.stats.mstats import mode

from functools import partial
from teletext.all import All
from .packet import Packet, HeaderPacket
from .subpage import Subpage
from .service import Service
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
    for mb in magbuffers:
        if ((drop_empty==False and len(mb) > 0) or len(mb) > 1) and type(mb[0]) == HeaderPacket:
            if mb[0].page_str() in pages:
                mb.sort(key=lambda p: p.mrag.row)
                for item in yield_func(mb):
                    yield item


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


def pipe():
    import sys
    import argparse

    from ..misc.all import All

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
                        default=All)
    parser.add_argument('-m', '--mags', type=int, metavar='M', nargs='+',
                        help='Only pass packets from these magazines.', default=All)
    parser.add_argument('-n', '--numbered',
                        help='When output is ascii, number packets according to offset in input file.',
                        action='store_true')
    parser.add_argument('-p', '--pages', type=str, metavar='M', nargs='+',
                        help='Only pass packets from these magazines.', default=All)
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

    if args.pages is not All:
        args.paginate = True

    if args.windowed or args.less:
        import teletext.terminal as term
        if args.windowed:
            term.change_terminal(term.urxvt('Teletext',
                                            ['-geometry', '67x32', '+sb', '-fg', 'white', '-bg', 'black', '-fn',
                                             'teletext', '-fb', 'teletext']))
            if args.less:
                term.less()
        else:
            if args.less:
                term.less(['-F'])

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

    for packet in iter:
        if args.spellcheck:
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
