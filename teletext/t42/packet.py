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

    def to_ansi(self, colour=True):
        return '%d %2d' % (self.mrag.magazine, self.mrag.row)

    def to_bytes(self):
        return ''


class DisplayPacket(Packet):

    def __init__(self, mrag, displayable):
        Packet.__init__(self, mrag)
        self.displayable = displayable

    @classmethod
    def from_bytes(cls, mrag, bytes):
        return cls(mrag, bytes[2:])

    def to_ansi(self, colour=True):
        return str(PrinterANSI(self.displayable, colour))

    def to_bytes(self):
        return self.mrag.to_bytes() + parity_encode(self.displayable).tostring()



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

    def page_str(self):
        return '%1d%02x' % (self.mrag.magazine, self.header.page)

    def subpage_str(self):
        return '%04x' % (self.header.subpage)

    def to_ansi(self, colour=True):
        return '   P' + self.page_str() + ' ' + str(PrinterANSI(self.displayable, colour))

    def to_bytes(self):
        return self.mrag.to_bytes() + self.header.to_bytes() + parity_encode(self.displayable).tostring()



class FastextPacket(Packet):

    def __init__(self, mrag, links=[None for n in range(6)]):
        Packet.__init__(self, mrag)
        self.__links = [PageLink() for n in range(6)]
        for n in range(6):
            if links[n]:
                self.__links[n] = links[n]

    @property
    def links(self):
        return self.__links

    @classmethod
    def from_bytes(cls, mrag, bytes):
        links = [PageLink.from_bytes(bytes[n:n+6], mrag.magazine) for n in range(3, 39, 6)]
        return cls(mrag, links)

    def to_ansi(self, colour=True):
        return ' '.join((str(link) for link in self.links))

    def to_bytes(self):
        return self.mrag.to_bytes() + ' ' + ''.join([x.to_bytes(self.mrag.magazine) for x in self.links]) + '   '


