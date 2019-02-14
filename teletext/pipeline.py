from collections import defaultdict

import numpy as np

from scipy.stats.mstats import mode
from tqdm import tqdm

from .subpage import Subpage


def check_buffer(mb, pages, subpages, min_rows=0):
    if (len(mb) > min_rows) and mb[0].type == 'header':
        page = mb[0].header.page | (mb[0].mrag.magazine * 0x100)
        if page in pages or (page & 0x7fff) in pages:
            if mb[0].header.subpage in subpages:
                yield sorted(mb, key=lambda p: p.mrag.row)


def paginate(packets, pages=range(0x900), subpages=range(0x3f7f), drop_empty=False):

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


def subpage_squash(packet_lists, min_duplicates=3):

    """Yields squashed subpages."""

    spdict = defaultdict(list)
    for pl in packet_lists:
        subpage = Subpage.from_packets(pl)
        spdict[(subpage.mrag.magazine, subpage.header.page, subpage.header.subpage)].append(subpage)

    for splist in tqdm(spdict.values(), unit=' Subpages'):
        if len(splist) >= min_duplicates:
            arr = mode(np.stack([sp[:] for sp in splist]), axis=0)[0][0].astype(np.uint8)
            numbers = mode(np.stack([np.clip(sp.numbers, -100, -1) for sp in splist]), axis=0)[0][0].astype(np.int64)
            yield Subpage(arr, numbers)


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
