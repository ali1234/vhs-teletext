from .elements import *
from .printer import PrinterANSI
from .finders import Finders


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
        elif mrag.row == 30:
            packet = BroadcastPacket.from_bytes(mrag, bytes)
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
        ranks = [(f.match(displayable),f) for f in Finders]
        ranks.sort(reverse=True)
        if ranks[0][0] > 20:
            self.name = ranks[0][1].name
            self.displayable_fixed = ranks[0][1].fixup(displayable.copy())
        else:
            self.name = 'Unknown'
            self.displayable_fixed = displayable
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


import enchant
d = enchant.Dict('en_GB')


freecombos = [
    set(['e', 'i', 'j']),
    set(['r', 's', 't', 'u', 'k']),
    set(['y', 'z']),
    set(['k', 'g', 'o']),
    set(['n', 'm']),
    set(['d', 'h']),
]


def check_pair(x, y):
    x = x.lower()
    y = y.lower()
    if x == y:
        return 0
    for s in freecombos:
        if x in s and y in s:
            return 0
    return 1


def weighted_hamming(a, b):
    count = 0
    return sum([check_pair(x, y) for x,y in zip(a, b)])


def case_match(word, src):
    return ''.join([c.lower() if d.islower() else c.upper() for c, d in zip(word, src)])

def spellcheck(packet):
    if type(packet) == DisplayPacket or type(packet) == HeaderPacket:
        words = str(PrinterANSI(packet.displayable, False)).decode('utf-8')
        words = ''.join([c if c.isalnum() else ' ' for c in words])
        words = words.split(' ')

        for n,w in enumerate(words):
          if len(w) > 2 and not d.check(w.lower()):
            s = filter(lambda x: len(x) == len(w) and weighted_hamming(x, w) == 0, d.suggest(w.lower()))
            if len(s) > 0:
                words[n] = case_match(s[0], w)

        words = ' '.join(words)
        for n,c in enumerate(words):
            if c != ' ':
                packet.displayable[n] = ord(c)

