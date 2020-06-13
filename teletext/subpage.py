import base64
import itertools

import numpy as np

from .coding import crc
from .packet import Packet
from .elements import Element, Displayable
from .printer import PrinterHTML
from .file import FileChunker

class Subpage(Element):

    def __init__(self, array=None, numbers=None, prefill=False):
        super().__init__((26 + (3*16), 42), array)
        if numbers is None:
            self._numbers = np.full((26 + (3*16),), fill_value=-100, dtype=np.int64)
        else:
            self._numbers = numbers

        if prefill:
            for i in range(1, 25):
                self.packet(i).mrag.row = i
                self._numbers[i] = -1
            self.displayable[:] = 0x20

    @property
    def numbers(self):
        return self._numbers[:]

    def number(self, row, dc=0):
        if row < 26:
            return self._numbers[row]
        else:
            return self._numbers[((row-26)*16)+26+dc]

    def packet(self, row, dc=0):
        if row < 26:
            return Packet(self._array[row, :])
        else:
            return Packet(self._array[((row-26)*16)+26+dc, :])

    def has_packet(self, row, dc=0):
        return self.number(row, dc) > -100

    @property
    def mrag(self):
        return self.packet(0).mrag

    @property
    def header(self):
        return self.packet(0).header

    @property
    def fastext(self):
        return self.packet(27, 0).fastext

    @property
    def displayable(self):
        return Displayable((24, 40), self._array[1:25,2:])

    @property
    def checksum(self):
        '''Calculates the actual checksum of the subpage.'''
        c = 0
        if self.has_packet(0):
            for b in self.header.displayable:
                c = crc(b, c)
        else:
            for b in [0x20] * 24:
                c = crc(b, c)

        for r in range(1, 26):
            if self.has_packet(r):
                for b in self.packet(r)[2:]:
                    c = crc(b, c)
            else:
                for b in [0x20] * 40:
                    c = crc(b, c)

        return c

    @classmethod
    def from_packets(cls, packets, ignore_empty=False):
        s = cls()

        for p in packets:
            i = None
            r = p.mrag.row
            if r < 26:
                i = r
            elif r < 29:
                i = ((r - 26)*16) + p.dc.dc + 26
            if i is not None:
                if ignore_empty and s._numbers[i] > -100:
                    # we've already seen this packet
                    # if the new one is closer to all spaces than the old one, skip it
                    if np.sum(s._array[i, :] == 0x80) < np.sum(p[:] == 0x80):
                        continue
                s._array[i, :] = p[:]
                s._numbers[i] = -1 if p.number is None else p.number
        return s

    @classmethod
    def from_file(cls, f):
        chunks = FileChunker(f, 42)
        packets = (Packet(data, number) for number, data in chunks)
        return cls.from_packets(packets)

    @property
    def packets(self):
        for n, a in enumerate(self._array):
            if self._numbers[n] > -100:
                yield Packet(a, number=None if self._numbers[n] < 0 else self._numbers[n])

    @property
    def url(self):
        parts = ['0']
        parts.append(base64.b64encode(Element((25, 40), self._array[0:25,2:]).sevenbit, b'-_').decode('ascii'))
        parts.append(f'PN={self.mrag.magazine}{self.header.page:02x}')
        # TODO: PS
        parts.append(f'SC={self.header.subpage:04x}')
        if self.has_packet(25):
            parts.append('X25=' + base64.b64encode(Element((1, 40), self._array[25:26,2:]).sevenbit, b'-_').decode('ascii'))
        if self.has_packet(26):
            pass # TODO: X26
        if self.has_packet(27, 0):
            parts.append('X270=' + ''.join([f'{l.magazine}{l.page:02x}{l.subpage:04x}' for l in self.fastext.links]) + f'{self.fastext.control:1x}')
        if self.has_packet(28, 0):
            pass # TODO: X280
        if self.has_packet(28, 4):
            pass # TODO: X284

        return ':'.join(parts)


    def to_html(self, pages_set):
        lines = []

        lines.append(f'<div class="subpage" id="{self.header.subpage:04x}">')
        p = PrinterHTML(self.header.displayable[:])
        p.anchor = f'#{self.header.subpage:04x}'
        lines.append(f'    <span class="pgnum">P{self.mrag.magazine}{self.header.page:02x}{str(p)}')

        for i in range(0,24):
            # only draw the line if previous line does not contain double height code
            if i == 0 or np.all(self.displayable[i-1,:] != 0x0d):
                fastext = [f'{l.magazine}{l.page:02x}' for l in self.fastext.links] if i == 23 else None
                p = PrinterHTML(self.displayable[i,:], fastext=fastext, pages_set=pages_set)
                lines.append(str(p))

        lines.append('</div>')

        return '\n'.join(lines)

