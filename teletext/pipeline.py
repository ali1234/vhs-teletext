from collections import defaultdict
from statistics import mode as pymode

import numpy as np

from scipy.stats.mstats import mode
from tqdm import tqdm

from .subpage import Subpage
from .packet import Packet


def check_buffer(mb, pages, subpages, min_rows=0):
    if (len(mb) > min_rows) and mb[0].type == 'header':
        page = mb[0].header.page | (mb[0].mrag.magazine * 0x100)
        if page in pages or (page & 0x7ff) in pages:
            if mb[0].header.subpage in subpages:
                yield sorted(mb, key=lambda p: p.mrag.row)


def packet_squash(packets):
    return Packet(mode(np.stack([p._array for p in packets]), axis=0)[0][0].astype(np.uint8))


def bsdp_squash_format1(packets):
    date = pymode([p.broadcast.format1.date for p in packets])
    hour = min(pymode([p.broadcast.format1.hour for p in packets]), 99)
    minute = min(pymode([p.broadcast.format1.minute for p in packets]), 99)
    second = min(pymode([p.broadcast.format1.second for p in packets]), 99)
    return f'{date} {hour:02d}:{minute:02d}:{second:02d}'


def bsdp_squash_format2(packets):
    day = min(pymode([p.broadcast.format2.day for p in packets]), 99)
    month = min(pymode([p.broadcast.format2.month for p in packets]), 99)
    hour = min(pymode([p.broadcast.format1.hour for p in packets]), 99)
    minute = min(pymode([p.broadcast.format1.minute for p in packets]), 99)
    return f'{month:02d}-{day:02d} {hour:02d}:{minute:02d}'

def paginate(packets, pages=range(0x900), subpages=range(0x3f80), drop_empty=False):

    """Yields packet lists containing contiguous rows."""

    magbuffers = [[],[],[],[],[],[],[],[]]
    for packet in packets:
        mag = packet.mrag.magazine & 0x7
        if packet.type == 'header':
            yield from check_buffer(magbuffers[mag], pages, subpages, 1 if drop_empty else 0)
            magbuffers[mag] = []
        magbuffers[mag].append(packet)
    for mb in magbuffers:
        yield from check_buffer(mb, pages, subpages, 1 if drop_empty else 0)


def subpage_squash(packet_lists, min_duplicates=3, ignore_empty=False):

    """Yields squashed subpages."""

    spdict = defaultdict(list)
    for pl in packet_lists:
        if len(pl) > 1:
            subpage = Subpage.from_packets(pl, ignore_empty=ignore_empty)
            spdict[(subpage.mrag.magazine, subpage.header.page, subpage.header.subpage)].append(subpage)

    for splist in tqdm(spdict.values(), unit=' Subpages'):
        if len(splist) >= min_duplicates:
            numbers = mode(np.stack([np.clip(sp.numbers, -100, -1) for sp in splist]), axis=0)[0][0].astype(np.int64)
            s = Subpage(numbers=numbers)
            for row in range(29):
                if row in [26, 27, 28]:
                    for dc in range(16):
                        if s.number(row, dc) > -100:
                            packets = [sp.packet(row, dc) for sp in splist if sp.number(row, dc) > -100]
                            arr = np.stack([p[3:] for p in packets])
                            s.packet(row, dc)[:3] = packets[0][:3]
                            if row == 27:
                                s.packet(row, dc)[3:] = mode(arr, axis=0)[0][0].astype(np.uint8)
                            else:
                                t = arr.astype(np.uint32)
                                t = t[:, 0::3] | (t[:, 1::3] << 8) | (t[:, 2::3] << 16)
                                result = mode(t, axis=0)[0][0].astype(np.uint32)
                                s.packet(row, dc)[3::3] = result & 0xff
                                s.packet(row, dc)[4::3] = (result >> 8) & 0xff
                                s.packet(row, dc)[5::3] = (result >> 16) & 0xff
                else:
                    if s.number(row) > -100:
                        packets = [sp.packet(row) for sp in splist if sp.number(row) > -100]
                        arr = np.stack([p[2:] for p in packets])
                        s.packet(row)[:2] = packets[0][:2]
                        s.packet(row)[2:] = mode(arr, axis=0)[0][0].astype(np.uint8)

            yield s


def to_file(packets, f, format):

    """Write packets to f as format."""

    if format == 'auto':
        format = 'debug' if f.isatty() else 'bytes'
    if f.isatty():
        for p in packets:
            with tqdm.external_write_mode():
                f.write(getattr(p, format))
            yield p
    else:
        for p in packets:
            f.write(getattr(p, format))
            yield p
