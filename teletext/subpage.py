import numpy as np

from .packet import Packet
from .elements import Displayable


class Subpage(object):

    def __init__(self):
        self._array = np.zeros((32, 42), dtype=np.uint8)
        self._numbers = np.full((32,), -100, dtype=np.int64)

    def row(self, row):
        return Packet(self._array[row, :])

    @property
    def header(self):
        return Packet(self._array[0, :]).header

    @property
    def fastext(self):
        return Packet(self._array[27, :]).fastext

    @property
    def displayable(self):
        return Displayable(self._array[1:26,2:])

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

