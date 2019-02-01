from typing import Any, Tuple

import numpy as np

from .elements import *
from .printer import PrinterANSI
from .finders import Finders


class Packet(object):

    def _setup(self, array):
        self._array = array
        self._elements = {}

    def __init__(self, magazine, row):
        self._setup(np.zeros((42,), dtype=np.uint8))
        self.mrag.set(magazine, row)

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, value):
        self._array[item] = value

    @element
    def mrag(self):
        return Mrag(self._array[:2])

    @classmethod
    def from_bytes(cls, data):
        """Packet factory which returns the appropriate type object for the packet."""

        if type(data) == bytes:
            array = np.fromstring(data, dtype=np.uint8)
        else:
            array = data.copy()

        if array.shape != (42, ):
            raise IndexError('Packet.from_bytes requires 42 bytes.')

        mrag = Mrag(array[:2])

        if mrag.row == 0:
            packet = HeaderPacket.__new__(HeaderPacket)
        elif mrag.row < 25:
            packet = DisplayPacket.__new__(DisplayPacket)
        elif mrag.row == 27:
            packet = FastextPacket.__new__(FastextPacket)
        #elif packet.mrag.row == 30:
        #    packet = BroadcastPacket.__new__(BroadcastPacket)
        else:
            packet = Packet.__new__(Packet)

        packet._setup(array)

        return packet

    def to_ansi(self, colour=True):
        return '%d %2d Unknown packet' % (self.mrag.magazine, self.mrag.row)

    def to_bytes(self):
        return self._array.tobytes()


class DisplayPacket(Packet):

    @element
    def displayable(self):
        return Displayable(self._array[2:])

    def to_ansi(self, colour=True):
        return self.displayable.to_ansi(colour)



class HeaderPacket(Packet):

    def __init__(self, magazine, row, page, subpage, control, displayable):
        Packet.__init__(self, magazine, row)
        self.header.set(page, subpage, control)
        self.displayable[:] = parity_encode(displayable)

    def ranks(self):
        ranks = [(f.match(self.displayable[:]),f) for f in Finders]
        ranks.sort(reverse=True, key=lambda x: x[0])
        if ranks[0][0] > 20:
            self.name = ranks[0][1].name
            self.finder = ranks[0][1]
            self.displayable_fixed = ranks[0][1].fixup(self.displayable[:].copy())
        else:
            self.name = 'Unknown'
            self.displayable_fixed = self.displayable

    @element
    def header(self):
        return PageHeader(self._array[2:10])

    @element
    def displayable(self):
        return Displayable(self._array[10:])

    def page_str(self):
        return '%1d%02x' % (self.mrag.magazine, self.header.page)

    def subpage_str(self):
        return '%04x' % (self.header.subpage)

    def to_ansi(self, colour=True):
        return '   P' + self.page_str() + ' ' + self.displayable.to_ansi(colour)


class FastextPacket(Packet):

    @element
    def links(self):
        return [PageLink(self._array[n:n+6]) for n in range(3, 39, 6)]

    def to_ansi(self, colour=True):
        return ' '.join((str(link) for link in self.links))


# borked for now

class BroadcastPacket(Packet):

    def __init__(self, mrag, dc, initial_page, displayable):
        Packet.__init__(self, mrag)
        self.dc = dc
        self.initial_page = initial_page
        self.displayable = displayable

    @classmethod
    def from_bytes(cls, mrag, bytes):
        dc = hamming8_decode(bytes[0])[0]
        return cls(mrag, dc, PageLink.from_bytes(bytes[1:7], 0), bytes[22:])

    def to_ansi(self, colour=True):
        return 'DC=' + str(self.dc) + ' ' + str(PrinterANSI(self.displayable, colour))

    def to_bytes(self):
        return self.mrag.to_bytes() + '                    ' + parity_encode(self.displayable).tostring()


