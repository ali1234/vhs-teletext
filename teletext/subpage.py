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
            for i in range(0, 25):
                self.init_packet(i)
            self.header.displayable[:] = 0x20
            self.displayable[:] = 0x20

        self.duplicates = []

    def diff(self, other):
        """Try to determine if two subpages are the same."""
        diff = np.sum(self._array != other._array, axis=1)
        rows = (self._numbers != -100) & (other._numbers != -100)
        return np.sum(diff * rows)

    @property
    def numbers(self):
        return self._numbers[:]

    def _slot(self, row, dc):
        if row < 26:
            return row
        else:
            return ((row-26)*16)+26+dc

    def has_packet(self, row, dc=0):
        return self._numbers[self._slot(row, dc)] > -100

    def init_packet(self, row, dc=0):
        self.packet(row, dc).mrag.row = row
        self._numbers[self._slot(row, dc)] = -1

    def packet(self, row, dc=0):
        try:
            return Packet(self._array[self._slot(row, dc), :])
        except IndexError:
            print(row, dc)
            raise

    @property
    def mrag(self):
        return self.packet(0).mrag

    @property
    def header(self):
        return self.packet(0).header

    @property
    def codepage(self):
        return self.packet(0).header.codepage

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
            for b in self.header.displayable[:24]:
                c = crc(b, c)
        else:
            for n in range(24):
                c = crc(0x20, c)

        for r in range(1, 26):
            if self.has_packet(r):
                for b in self.packet(r).displayable:
                    c = crc(b, c)
            else:
                for n in range(40):
                    c = crc(0x20, c)

        return c

    @property
    def addr(self):
        return f'{self.mrag.magazine}{self.header.page:02X}:{self.header.subpage:04X}'

    @classmethod
    def from_packets(cls, packets, ignore_empty=False):
        s = cls()

        for p in packets:
            row = p.mrag.row
            if row >= 29:
                continue
            dc = 0 if row < 26 else p.dc.dc
            i = s._slot(row, dc)
            if ignore_empty and s._numbers[i] > -100:
                # we've already seen this packet
                # if the new one is closer to all spaces than the old one, skip it
                if np.sum(s._array[i, :] == 0x80) < np.sum(p[:] == 0x80):
                    continue
            s._array[i, :] = p[:]
            s._numbers[i] = -1 if p.number is None else p.number
        return s

    @classmethod
    def from_url(cls, url):
        s = cls(prefill=True)
        parts = url.split(':')
        Element((25, 40), s._array[0:25, 2:]).sevenbit = base64.urlsafe_b64decode(parts[1]+'==')
        for p in parts[2:]:
            l, d = p.split('=', maxsplit=1)
            if l == 'PN':
                s.mrag.magazine = int(d[0], 16)
                s.header.page = int(d[1:3], 16)
            elif l == 'PS':
                pass # TODO
            elif l == 'SC':
                pass  # TODO
            elif l == 'X25':
                pass  # TODO
            elif l == 'X270':
                pass  # TODO
            elif l == 'X280':
                pass  # TODO
            elif l == 'X284':
                pass  # TODO
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
        parts.append(base64.urlsafe_b64encode(Element((25, 40), self._array[0:25,2:]).sevenbit).decode('ascii').rstrip('='))
        parts.append(f'PN={self.mrag.magazine}{self.header.page:02x}')
        c = self.header.control
        parts.append(f'PS={(c>>1) | ((c&1)<<13):x}')
        parts.append(f'SC={self.header.subpage:x}')
        if self.has_packet(25):
            parts.append('X25=' + base64.urlsafe_b64encode(Element((1, 40), self._array[25:26,2:]).sevenbit).decode('ascii').rstrip('='))
        for d in range(16):
            if self.has_packet(26, d):
                pass # TODO: X26
        if self.has_packet(27, 0):
            parts.append('X270=' + ''.join([f'{l.magazine}{l.page:02x}{l.subpage:04x}' for l in self.fastext.links]) + f'{self.fastext.control:1x}')
        if self.has_packet(28, 0):
            pass # TODO: X280
        if self.has_packet(28, 4):
            pass # TODO: X284

        return ':'.join(parts)


    def to_html(self, pages_set, localcodepage=None):
        lines = []

        lines.append(f'<div class="subpage" id="{self.header.subpage:04x}">')
        buf = np.full((40,), fill_value=0x20, dtype=np.uint8)
        buf[3:7] = np.fromstring(f'P{self.mrag.magazine}{self.header.page:02x}', dtype=np.uint8)
        buf[8:] = self.header.displayable[:]
        p = PrinterHTML(buf, localcodepage=localcodepage, codepage=self.codepage)
        p.anchor = f'#{self.header.subpage:04x}'
        lines.append(str(p))

        for i in range(0,24):
            # only draw the line if previous line does not contain double height code
            if i == 0 or np.all(self.displayable[i-1,:] != 0x0d):
                fastext = [f'{l.magazine}{l.page:02x}' for l in self.fastext.links] if i == 23 else None
                p = PrinterHTML(self.displayable[i,:], fastext=fastext, pages_set=pages_set, localcodepage=localcodepage, codepage=self.codepage)
                lines.append(str(p))

        lines.append('</div>')

        return ''.join(lines)

