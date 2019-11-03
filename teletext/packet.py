from .elements import *


class Packet(Element):

    def __init__(self, array=None, number=None, original=None):
        super().__init__((42, ), array)
        self._number = number
        self._original = original

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
        elif row < 26:
            return 'display'
        elif row == 27:
            return 'fastext'
        elif row == 30 and self.mrag.magazine == 8:
            return 'broadcast'
        elif row in [26, 28]:
            return 'page enhancement'
        elif row == 29:
            return 'magazine enhancement'
        elif row == 31:
            return 'independant data'
        else:
            return 'unknown'

    @property
    def mrag(self):
        return Mrag(self._array[:2])

    @property
    def dc(self):
        return DesignationCode((1,), self._array[2:3])

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
            return f'{t}'

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

    @property
    def debug(self):
        if self.number is None:
            return f'None     {self.mrag.magazine} {self.mrag.row:2d} {self.to_ansi(colour=True)} errors: {np.sum(self.errors)}\n'.encode('utf8')
        else:
            return f'{self.number:8d} {self.mrag.magazine} {self.mrag.row:2d} {self.to_ansi(colour=True)} errors: {np.sum(self.errors)}\n'.encode('utf8')

    @property
    def vbi(self):
        if self._original is None:
            raise Exception('Original VBI data is not available. Probably we are not deconvolving.')
        return self._original

    @property
    def errors(self):
        e = np.zeros_like(self._array)
        e[:2] = self.mrag.errors
        t = self.type

        if t == 'header':
            e[2:] = self.header.errors
        elif t == 'display':
            e[2:] = self.displayable.errors
        elif t == 'fastext':
            e[2:] = self.fastext.errors
        elif t == 'broadcast':
            e[2:] = self.broadcast.errors

        return e
