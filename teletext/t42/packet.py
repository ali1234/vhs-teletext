import numpy

from coding import *
from descriptors import *
from elements import *
from printer import PrinterANSI


class Packet(object):

    def __init__(self, mrag):
        self.__mrag = mrag

    @property
    def mrag(self):
        return self.__mrag

    @classmethod
    def from_bytes(cls, data):
        """Packet factory which returns the appropriate type object for the packet."""
        if type(data) == str:
            bytes = numpy.fromstring(data, dtype=numpy.uint8)
        else:
            bytes = data
        if bytes.shape != (42, ):
            raise IndexError('Packet.from_bytes requires 42 bytes.')

        mrag = Mrag.from_bytes(bytes[:2])

        if mrag.row == 0:
            packet = HeaderPacket.from_bytes(mrag, bytes)
        elif mrag.row < 25:
            packet = DisplayPacket.from_bytes(mrag, bytes)
        elif mrag.row == 27:
            packet = FastextPacket.from_bytes(mrag, bytes)
        else:
            packet = Packet(mrag)

        packet._original_bytes = bytes
        return packet

    def __str__(self):
        return '%d %2d' % (self.mrag.magazine, self.mrag.row)



class DisplayPacket(Packet):

    def __init__(self, mrag, displayable):
        Packet.__init__(self, mrag)
        self.displayable = displayable

    @classmethod
    def from_bytes(cls, mrag, bytes):
        return cls(mrag, bytes[2:])

    def __str__(self):
        return Packet.__str__(self) + str(PrinterANSI(self.displayable))


class HeaderPacket(DisplayPacket):

    def __init__(self, mrag, header, displayable):
        DisplayPacket.__init__(self, mrag, displayable)
        self.__header = header

    @property
    def header(self):
        return self.__header

    @classmethod
    def from_bytes(cls, mrag, bytes):
        return cls(mrag, PageHeader.from_bytes(bytes[2:10]), bytes[10:])

    def __str__(self):
        return Packet.__str__(self) + '   P' + self.page_str() + ' ' + str(PrinterANSI(self.displayable)) + ' %04x %03x' % (self.header.subpage, self.header.control)

    def page_str(self):
        return '%1d%02x' % (self.mrag.magazine, self.header.page)

    def subpage_str(self):
        return '%04x' % (self.header.subpage)


class FastextPacket(Packet):

    def __init__(self, mrag, links=[PageLink() for n in range(6)]):
        Packet.__init__(self, mrag)
        self.__links = links

    @property
    def links(self):
        return self.__links

    @classmethod
    def from_bytes(cls, mrag, bytes):
        links = [PageLink.from_bytes(bytes[n:n+6], mrag.magazine) for n in range(3, 39, 6)]
        return cls(mrag, links)

    def __str__(self):
        return Packet.__str__(self) + ' ' + ' '.join((str(link) for link in self.links))


if __name__ == '__main__':
    p = Packet.from_bytes("ababababababababababababababababababababab")
    print str(p)
    p = Packet(Mrag(1, 0), "asdf")
    print str(p)
    try:
        p.mrag = 2
    except AttributeError:
        print "Writing to mrag failed as expected."
    p.mrag.row = 2
    print str(p)

