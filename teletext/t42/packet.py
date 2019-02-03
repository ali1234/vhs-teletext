from typing import Any, Tuple

import numpy as np

from .elements import *
from .printer import PrinterANSI
from .finders import Finders


class Packet(object):

    def __init__(self, data):

        if type(data) == bytes:
            array = np.fromstring(data, dtype=np.uint8)
        else:
            array = data

        if array.shape != (42, ):
            raise IndexError('Packet.from_bytes requires 42 bytes.')

        self._array = array

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, value):
        self._array[item] = value

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
        return Displayable(self._array[2:])

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

    def to_bytes(self):
        return self._array.tobytes()




