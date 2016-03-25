import numpy

from descriptors import *
from elements import *
from packet import *
from printer import PrinterHTML


class Subpage(object):
    control = ControlBits()

    def __init__(self, fill=0x20, links=[None for n in range(6)]):
        self.displayable = numpy.full((40, 25), fill, dtype=numpy.uint8)
        self.control = 0
        self.__links = [link if link else PageLink() for link in links]

    @property
    def links(self):
        return self.__links

    @staticmethod
    def from_packets(packet_iter):
        s = Subpage()

        for p in packet_iter:
            if type(p) == HeaderPacket:
                s._original_subpage = p.header.subpage
                s._original_page = p.header.page
                s._original_magazine = p.mrag.magazine
                s._original_displayable = p.displayable
            elif type(p) == FastextPacket:
                for i in range(6):
                    s.links[i] = p.links[i]
            elif type(p) == DisplayPacket:
                s.displayable[:,p.mrag.row-1] = p.displayable

        return s

    def to_packets(self, magazineno, pageno, subpageno, header_displayable=numpy.full((32,), 0x20, dtype=numpy.uint8)):
        yield HeaderPacket(Mrag(magazineno, 0), PageHeader(pageno, subpageno, self.control), header_displayable)
        for i in range(0, 25):
            if (self.displayable[:,i] != 0x20).any():
                yield DisplayPacket(Mrag(magazineno, i+1), self.displayable[:,i])
        yield FastextPacket(Mrag(magazineno, 27), self.links)


    def to_html(self, magazineno, pageno, subpageno, header_displayable=numpy.full((32,), 0x20, dtype=numpy.uint8)):
        body = []

        p = PrinterHTML(header_displayable)
        p.anchor = '#%04X' % subpageno
        body.append('   <span class="pgnum">P%d%02x</span> ' % (magazineno, pageno) + str(p))

        for i in range(0,25):
            if i == 0 or numpy.all(self.displayable[:,i-1] != 0x0d):
                p = PrinterHTML(self.displayable[:,i])
                if i == 23:
                    p.fastext = True
                    p.links = ['%d%02X' % (l.magazine, l.page) for l in self.links]
                body.append(str(p))

        head = '<div class="subpage" id="%04X">' % subpageno

        return head + "".join(body) + '</div>'

