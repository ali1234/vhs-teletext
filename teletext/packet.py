from .elements import *


class Packet(Element):

    def __init__(self, array=None, number=None):
        super().__init__((42, ), array)
        self._number = number

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, value):
        self._array[item] = value

    @property
    def number(self):
        return self._number

    @property
    def type(self):
        row = self.mrag.row
        if row == 0:
            return 'header'
        elif row < 25:
            return 'display'
        elif row == 27:
            return 'fastext'
        elif row == 30:
            return 'broadcast'
        else:
            return 'unknown'

    @property
    def mrag(self):
        return Mrag(self._array[:2])

    @property
    def header(self):
        return Header(self._array[2:])

    @property
    def displayable(self):
        return Displayable((40,), self._array[2:])

    @property
    def fastext(self):
        return Fastext(self._array[2:], self.mrag)

    @property
    def broadcast(self):
        return BroadcastData(self._array[2:], self.mrag)

    def to_ansi(self, colour=True):
        t = self.type

        if t == 'header':
            return f'   P{self.mrag.magazine}{self.header.to_ansi(colour)}'
        elif t == 'display':
            return self.displayable.to_ansi(colour)
        elif t == 'fastext':
            return self.fastext.to_ansi(colour)
        elif t == 'broadcast':
            return self.broadcast.to_ansi(colour)
        else:
            return f'Unknown packet {self.mrag}'

    def to_binary(self):
        b = np.unpackbits(self._array[::-1])[::-1]
        x = b[0::2] | (b[1::2]<<1)
        return ''.join([' ', chr(0x258C), chr(0x2590), chr(0x2588)][n] for n in x)

    def to_bytes(self):
        return self._array.tobytes()

    @property
    def ansi(self):
        return self.to_ansi(colour=True).encode('utf8') + b'\n'

    @property
    def text(self):
        return self.to_ansi(colour=False).encode('utf8') + b'\n'

    @property
    def bar(self):
        return self.to_binary().encode('utf8') + b'\n'
