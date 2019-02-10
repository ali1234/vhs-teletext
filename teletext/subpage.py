import numpy as np

from .packet import Packet
from .elements import Element, Displayable


class Subpage(Element):

    def __init__(self, array=None, numbers=None):
        super().__init__((32, 42), array)
        if numbers is None:
            self._numbers = np.full((32,), -100, dtype=np.int64)
        else:
            self._numbers = numbers

    @property
    def numbers(self):
        return self._numbers[:]

    def row(self, row):
        return Packet(self._array[row, :])

    @property
    def mrag(self):
        return Packet(self._array[0, :]).mrag

    @property
    def header(self):
        return Packet(self._array[0, :]).header

    @property
    def fastext(self):
        return Packet(self._array[27, :]).fastext

    @property
    def displayable(self):
        return Displayable((25, 40), self._array[1:26,2:])

    @staticmethod
    def from_packets(packet_iter):
        s = Subpage()

        for p in packet_iter:
            s._array[p.mrag.row, :] = p[:]
            s._numbers[p.mrag.row] = -1 if p.number is None else -2
        return s

    @property
    def packets(self):
        for n, a in enumerate(self._array):
            if self._numbers[n] > -100:
                yield Packet(a, number=None if self._numbers[n] < 0 else self._numbers[n])

# still broken
#    def to_html(self, magazineno, pageno, subpageno, header_displayable=numpy.full((32,), 0x20, dtype=numpy.uint8), pages_set=All):
#        body = []
#
#        p = PrinterHTML(header_displayable)
#        p.anchor = '#%04X' % subpageno
#        body.append('   <span class="pgnum">P%d%02x</span> ' % (magazineno, pageno) + str(p))
#
#        for i in range(0,25):
#            if i == 0 or numpy.all(self.displayable[:,i-1] != 0x0d):
#                p = PrinterHTML(self.displayable[:,i], pages_set=pages_set)
#                if i == 23:
#                    p.fastext = True
#                    p.links = ['%d%02X' % (l.magazine, l.page) for l in self.links]
#                body.append(str(p))
#
#        head = '<div class="subpage" id="%04X">' % subpageno
#
#        return head + "".join(body) + '</div>'

