import numpy

from descriptors import *
from elements import *
from packet import *


class Subpage(object):
    control = ControlBits()

    def __init__(self, fill=0x20, links=[None for n in range(6)]):
        self.displayable = numpy.full((40, 25), fill, dtype=numpy.uint8)
        self.control = 0
        self.__links = links

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


