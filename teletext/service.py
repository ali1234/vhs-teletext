import datetime

from collections import defaultdict

import numpy as np

from .coding import parity_encode
from .packet import Packet


class Page(object):
    def __init__(self):
        self.subpages = {}
        self._iter = self._gen()

    def _gen(self):
        while True:
            if len(self.subpages) == 0:
                yield 0x3f7f, None
            else:
                yield from sorted(self.subpages.items())

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

class Magazine(object):
    def __init__(self, title='Unnamed  '):
        self.title = title
        self.pages = defaultdict(Page)
        self._iter = self._gen()

    def _gen(self):
        while True:
            for pageno, page in sorted(self.pages.items()):
                spno, subpage = next(page)
                if subpage is None:
                    p = Packet()
                    p.mrag.row = 0
                    p.header.page = 0xff
                    p.header.subpage = spno
                    yield p
                else:
                    subpage.header.page = pageno
                    subpage.header.subpage = spno
                    yield from subpage.packets

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

class Service(object):
    def __init__(self, replace_headers=False):
        self.magazines = defaultdict(Magazine)
        self.priorities = [1,1,1,1,1,1,1,1]
        self.replace_headers = replace_headers
        self._iter = self._gen()

    def fill_header(self, title, mag, page):
        t = datetime.datetime.now()
        data = '%9s%1d%02x' % (title, mag, page) + t.strftime(" %a %d %b\x03%H:%M/%S")
        return parity_encode(np.fromstring(data, dtype=np.uint8))

    def _gen(self):
        while True:
            for n,m in sorted(self.magazines.items()):
                for count in range(self.priorities[n&0x7]):
                    packet = next(m)
                    packet.mrag.magazine = n
                    if self.replace_headers and packet.type == 'header':
                        packet.header.displayable[:] = self.fill_header(m.title, n, packet.header.page)
                    yield packet

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)
